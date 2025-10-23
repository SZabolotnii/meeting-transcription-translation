"""
Transcription Orchestrator - coordinates all system components.
Manages the flow from audio capture through transcription to translation and UI updates.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime
from dataclasses import dataclass
from queue import Queue, Empty
import threading
import time

from ..config import config
from ..audio_capture.audio_manager import AudioManager
from ..translation.translator import Translator
from ..utils.performance_profiler import measure_performance, profiler


@dataclass
class TranscriptionResult:
    """Result of transcription and translation process"""
    original_text: str
    translated_text: Optional[str]
    timestamp: float
    confidence: float
    language: str
    target_language: Optional[str]
    processing_time: float
    segment_id: str


@dataclass
class SessionConfig:
    """Configuration for a transcription session"""
    audio_source_type: str  # "microphone" | "system_audio"
    audio_device_id: Optional[int]
    target_language: str
    whisper_model: str = "base"
    buffer_duration: float = 5.0


class TranscriptionOrchestrator:
    """
    Main orchestrator that coordinates audio capture, transcription, and translation.
    Manages the complete pipeline from audio input to translated subtitles.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Session state
        self.session_active = False
        self.session_config: Optional[SessionConfig] = None
        self.session_start_time: Optional[datetime] = None
        
        # Components
        self.audio_manager: Optional[AudioManager] = None
        self.translator: Optional[Translator] = None
        self.whisper_transcriber = None  # Will be initialized when needed
        
        # Processing queues and threads
        self.audio_queue = Queue(maxsize=config.performance.audio_queue_size)
        self.transcription_queue = Queue(maxsize=config.performance.transcription_queue_size)
        self.result_queue = Queue(maxsize=config.performance.result_queue_size)
        
        # Processing threads
        self.audio_thread: Optional[threading.Thread] = None
        self.transcription_thread: Optional[threading.Thread] = None
        self.translation_thread: Optional[threading.Thread] = None
        
        # Callbacks for UI updates
        self.subtitle_callback: Optional[Callable[[TranscriptionResult], None]] = None
        self.status_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self.error_callback: Optional[Callable[[Exception], None]] = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_processed': 0,
            'transcription_latency': 0.0,
            'translation_latency': 0.0,
            'errors_count': 0,
            'audio_level': 0.0
        }
        
        # Session history
        self.session_history: List[TranscriptionResult] = []
        
        # Processing status
        self.processing_status = {
            'audio_capture': 'idle',
            'transcription': 'idle',
            'translation': 'idle'
        }
        
        self._shutdown_event = threading.Event()
        
    def set_callbacks(self, 
                     subtitle_callback: Optional[Callable[[TranscriptionResult], None]] = None,
                     status_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                     error_callback: Optional[Callable[[Exception], None]] = None):
        """Set callback functions for UI updates"""
        self.subtitle_callback = subtitle_callback
        self.status_callback = status_callback
        self.error_callback = error_callback
        
    def start_session(self, audio_source: str, target_language: str, 
                     audio_device: Optional[str] = None) -> bool:
        """
        Start a new transcription session.
        
        Args:
            audio_source: Type of audio source ("Мікрофон" or "Системне аудіо")
            target_language: Target language for translation
            audio_device: Specific audio device (optional)
            
        Returns:
            bool: True if session started successfully
        """
        if self.session_active:
            self.logger.warning("Session already active")
            return False
            
        try:
            # Map UI language names to config keys
            source_mapping = {
                "Мікрофон": "microphone",
                "Системне аудіо": "system_audio"
            }
            
            # Fallback to supported languages from config
            from ..config import SUPPORTED_LANGUAGES
            language_mapping = {v: k for k, v in SUPPORTED_LANGUAGES.items()}
            
            # Create session configuration
            self.session_config = SessionConfig(
                audio_source_type=source_mapping.get(audio_source, "microphone"),
                audio_device_id=None,  # Will be resolved by audio manager
                target_language=language_mapping.get(target_language, "uk"),
                whisper_model=config.whisper.model_size,
                buffer_duration=config.audio.buffer_duration
            )
            
            # Initialize components
            self._initialize_components()
            
            # Start processing threads
            self._start_processing_threads()
            
            # Mark session as active
            self.session_active = True
            self.session_start_time = datetime.now()
            self.session_history = []
            
            # Reset performance metrics
            self.performance_metrics = {
                'total_processed': 0,
                'transcription_latency': 0.0,
                'translation_latency': 0.0,
                'errors_count': 0,
                'audio_level': 0.0
            }
            
            self.logger.info(f"Session started: {audio_source} -> {target_language}")
            
            if self.status_callback:
                self.status_callback("session_started", {
                    'audio_source': audio_source,
                    'target_language': target_language,
                    'timestamp': self.session_start_time.isoformat()
                })
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start session: {e}")
            if self.error_callback:
                self.error_callback(e)
            return False
    
    def stop_session(self) -> bool:
        """
        Stop the current transcription session.
        
        Returns:
            bool: True if session stopped successfully
        """
        if not self.session_active:
            self.logger.warning("No active session to stop")
            return False
            
        try:
            self.logger.info("Stopping transcription session")
            
            # Signal shutdown to all threads
            self._shutdown_event.set()
            
            # Stop audio capture
            if self.audio_manager:
                self.audio_manager.stop_capture()
                
            # Wait for threads to finish
            self._stop_processing_threads()
            
            # Clean up components
            self._cleanup_components()
            
            # Mark session as inactive
            self.session_active = False
            
            # Reset processing status
            self.processing_status = {
                'audio_capture': 'idle',
                'transcription': 'idle',
                'translation': 'idle'
            }
            
            self.logger.info("Session stopped successfully")
            
            if self.status_callback:
                self.status_callback("session_stopped", {
                    'duration': (datetime.now() - self.session_start_time).total_seconds() if self.session_start_time else 0,
                    'total_processed': self.performance_metrics['total_processed']
                })
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping session: {e}")
            if self.error_callback:
                self.error_callback(e)
            return False
        finally:
            self._shutdown_event.clear()
    
    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get the complete session history"""
        return [
            {
                'timestamp': result.timestamp,
                'original_text': result.original_text,
                'translated_text': result.translated_text,
                'language': result.language,
                'target_language': result.target_language,
                'confidence': result.confidence,
                'processing_time': result.processing_time
            }
            for result in self.session_history
        ]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def get_processing_status(self) -> Dict[str, str]:
        """Get current processing status"""
        return self.processing_status.copy()
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Initialize audio manager
            self.audio_manager = AudioManager()
            self.logger.info("Audio manager initialized successfully")
            
            # Initialize translator if translation is needed
            self.translator = None
            if (self.session_config and 
                config.translation.api_key and 
                self.session_config.target_language != "en"):  # Assuming source is English
                try:
                    self.translator = Translator()
                    self.logger.info("Translator initialized successfully")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize translator: {e}. Translation will be disabled.")
                    self.translator = None
            else:
                self.logger.info("Translation disabled (no API key or same language)")
            
            # Whisper transcriber will be initialized in the transcription thread
            # to avoid blocking the main thread
            
            self.logger.info("Components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _start_processing_threads(self):
        """Start all processing threads"""
        self._shutdown_event.clear()
        
        # Audio capture thread
        self.audio_thread = threading.Thread(
            target=self._audio_capture_worker,
            name="AudioCaptureThread",
            daemon=True
        )
        self.audio_thread.start()
        
        # Transcription thread
        self.transcription_thread = threading.Thread(
            target=self._transcription_worker,
            name="TranscriptionThread",
            daemon=True
        )
        self.transcription_thread.start()
        
        # Translation thread (if translator is available)
        if self.translator:
            self.translation_thread = threading.Thread(
                target=self._translation_worker,
                name="TranslationThread",
                daemon=True
            )
            self.translation_thread.start()
        
        self.logger.info("Processing threads started")
    
    def _stop_processing_threads(self):
        """Stop all processing threads"""
        threads = [self.audio_thread, self.transcription_thread, self.translation_thread]
        
        # Wait for threads to finish (with timeout)
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
                if thread.is_alive():
                    self.logger.warning(f"Thread {thread.name} did not stop gracefully")
        
        self.audio_thread = None
        self.transcription_thread = None
        self.translation_thread = None
        
        self.logger.info("Processing threads stopped")
    
    def _cleanup_components(self):
        """Clean up all components"""
        if self.audio_manager:
            # AudioManager doesn't have cleanup method, just stop capture
            try:
                self.audio_manager.stop_capture()
            except:
                pass
            self.audio_manager = None
            
        # Clear queues
        self._clear_queue(self.audio_queue)
        self._clear_queue(self.transcription_queue)
        self._clear_queue(self.result_queue)
        
        self.logger.info("Components cleaned up")
    
    def _clear_queue(self, queue: Queue):
        """Clear all items from a queue"""
        try:
            while True:
                queue.get_nowait()
        except Empty:
            pass
    
    @measure_performance('audio_capture')
    def _audio_capture_worker(self):
        """Worker thread for audio capture"""
        self.logger.info("Audio capture worker started")
        
        try:
            if not self.audio_manager or not self.session_config:
                return
                
            self.processing_status['audio_capture'] = 'active'
            
            # Set up audio callback to receive audio segments
            def audio_callback(audio_segment):
                """Callback to handle incoming audio segments"""
                try:
                    # Update audio level for UI
                    import numpy as np
                    if len(audio_segment.data) > 0:
                        self.performance_metrics['audio_level'] = float(np.abs(audio_segment.data).mean())
                    
                    # Put audio data in transcription queue
                    if not self.audio_queue.full():
                        self.audio_queue.put((audio_segment.data, audio_segment.timestamp), timeout=1.0)
                        
                        # Track audio capture performance
                        profiler.add_metric('audio_capture_success', 1, 'count')
                    else:
                        self.logger.warning("Audio queue full, dropping audio data")
                        profiler.add_metric('audio_queue_drops', 1, 'count')
                        
                except Exception as e:
                    self.logger.error(f"Error in audio callback: {e}")
                    self.performance_metrics['errors_count'] += 1
                    profiler.add_metric('audio_capture_errors', 1, 'count')
            
            # Set the callback
            self.audio_manager.set_audio_callback(audio_callback)
            
            # Start audio capture
            self.audio_manager.start_capture(
                source_type=self.session_config.audio_source_type,
                device_id=self.session_config.audio_device_id
            )
            
            # Keep the worker thread alive while capturing
            while not self._shutdown_event.is_set() and self.audio_manager.is_capturing():
                time.sleep(0.5)  # Check every 500ms
                    
        except Exception as e:
            self.logger.error(f"Audio capture worker failed: {e}")
            if self.error_callback:
                self.error_callback(e)
        finally:
            self.processing_status['audio_capture'] = 'idle'
            self.logger.info("Audio capture worker stopped")
    
    @measure_performance('transcription')
    def _transcription_worker(self):
        """Worker thread for transcription"""
        self.logger.info("Transcription worker started")
        
        try:
            # Initialize Whisper transcriber in this thread
            from ..transcription.whisper_transcriber import WhisperTranscriber
            self.whisper_transcriber = WhisperTranscriber(
                model_size=self.session_config.whisper_model if self.session_config else "base"
            )
            
            self.processing_status['transcription'] = 'active'
            
            while not self._shutdown_event.is_set():
                try:
                    # Get audio data from queue
                    audio_data, timestamp = self.audio_queue.get(timeout=1.0)
                    
                    self.processing_status['transcription'] = 'processing'
                    
                    # Transcribe audio with performance tracking
                    start_time = time.time()
                    transcription_result = self.whisper_transcriber.transcribe_segment(audio_data)
                    transcription_time = time.time() - start_time
                    
                    self.performance_metrics['transcription_latency'] = transcription_time
                    profiler.add_metric('transcription_latency', transcription_time)
                    profiler.add_metric('transcription_queue_size', self.transcription_queue.qsize(), 'count')
                    
                    if transcription_result and transcription_result.strip():
                        # Track successful transcription
                        profiler.add_metric('transcription_success', 1, 'count')
                        profiler.add_metric('transcription_text_length', len(transcription_result), 'characters')
                        
                        # Put result in translation queue or directly to results
                        if self.translator:
                            self.transcription_queue.put((transcription_result, timestamp, transcription_time), timeout=1.0)
                        else:
                            # No translation needed, create final result
                            result = TranscriptionResult(
                                original_text=transcription_result,
                                translated_text=None,
                                timestamp=timestamp,
                                confidence=0.8,  # Default confidence
                                language="auto",
                                target_language=None,
                                processing_time=transcription_time,
                                segment_id=f"seg_{int(timestamp)}"
                            )
                            self._add_result(result)
                    else:
                        # Track empty transcription (silence or noise)
                        profiler.add_metric('transcription_empty', 1, 'count')
                    
                    self.processing_status['transcription'] = 'active'
                    
                except Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in transcription: {e}")
                    self.performance_metrics['errors_count'] += 1
                    profiler.add_metric('transcription_errors', 1, 'count')
                    if self.error_callback:
                        self.error_callback(e)
                    
        except Exception as e:
            self.logger.error(f"Transcription worker failed: {e}")
            if self.error_callback:
                self.error_callback(e)
        finally:
            self.processing_status['transcription'] = 'idle'
            self.logger.info("Transcription worker stopped")
    
    @measure_performance('translation')
    def _translation_worker(self):
        """Worker thread for translation"""
        self.logger.info("Translation worker started")
        
        try:
            self.processing_status['translation'] = 'active'
            
            while not self._shutdown_event.is_set():
                try:
                    # Get transcription result from queue
                    original_text, timestamp, transcription_time = self.transcription_queue.get(timeout=1.0)
                    
                    self.processing_status['translation'] = 'processing'
                    
                    # Translate text with performance tracking
                    start_time = time.time()
                    translation_result = self.translator.translate_text(
                        original_text, 
                        self.session_config.target_language if self.session_config else "uk"
                    )
                    translation_time = time.time() - start_time
                    
                    self.performance_metrics['translation_latency'] = translation_time
                    profiler.add_metric('translation_latency', translation_time)
                    profiler.add_metric('translation_text_length', len(original_text), 'characters')
                    
                    if translation_result and translation_result.translated_text:
                        profiler.add_metric('translation_success', 1, 'count')
                        profiler.add_metric('translated_text_length', len(translation_result.translated_text), 'characters')
                        translated_text = translation_result.translated_text
                    else:
                        profiler.add_metric('translation_failed', 1, 'count')
                        translated_text = original_text  # Fallback to original text
                    
                    # Calculate total processing time
                    total_time = transcription_time + translation_time
                    profiler.add_metric('total_processing_latency', total_time)
                    
                    # Create final result
                    result = TranscriptionResult(
                        original_text=original_text,
                        translated_text=translated_text,
                        timestamp=timestamp,
                        confidence=0.8,  # Default confidence
                        language="auto",
                        target_language=self.session_config.target_language if self.session_config else "uk",
                        processing_time=total_time,
                        segment_id=f"seg_{int(timestamp)}"
                    )
                    
                    self._add_result(result)
                    
                    self.processing_status['translation'] = 'active'
                    
                except Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in translation: {e}")
                    self.performance_metrics['errors_count'] += 1
                    profiler.add_metric('translation_errors', 1, 'count')
                    if self.error_callback:
                        self.error_callback(e)
                    
        except Exception as e:
            self.logger.error(f"Translation worker failed: {e}")
            if self.error_callback:
                self.error_callback(e)
        finally:
            self.processing_status['translation'] = 'idle'
            self.logger.info("Translation worker stopped")
    
    def _add_result(self, result: TranscriptionResult):
        """Add a transcription result to history and notify UI"""
        try:
            # Add to session history
            self.session_history.append(result)
            
            # Update performance metrics
            self.performance_metrics['total_processed'] += 1
            
            # Notify UI callback
            if self.subtitle_callback:
                self.subtitle_callback(result)
                
            # Keep history size manageable
            if len(self.session_history) > config.performance.max_history_entries:
                self.session_history = self.session_history[-config.performance.history_cleanup_threshold:]
                
        except Exception as e:
            self.logger.error(f"Error adding result: {e}")
            if self.error_callback:
                self.error_callback(e)