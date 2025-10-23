"""
Whisper Transcriber - handles speech-to-text conversion using OpenAI Whisper.
Optimized for real-time transcription with minimal latency.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any
import tempfile
import os

from ..config import config


class WhisperTranscriber:
    """
    Whisper-based transcriber for converting audio segments to text.
    Optimized for real-time processing with configurable model sizes.
    """
    
    def __init__(self, model_size: str = "base"):
        self.logger = logging.getLogger(__name__)
        self.model_size = model_size
        self.model = None
        self.device = config.whisper.device
        self.language = config.whisper.language
        
        # Performance tracking
        self.processing_count = 0
        self.total_processing_time = 0.0
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Whisper model"""
        try:
            import whisper
            
            self.logger.info(f"Loading Whisper model: {self.model_size}")
            
            # Load model with device optimization
            if self.device == "auto":
                # Auto-detect best device
                import torch
                if torch.backends.mps.is_available():
                    device = "mps"  # Apple Silicon GPU
                elif torch.cuda.is_available():
                    device = "cuda"  # NVIDIA GPU
                else:
                    device = "cpu"
            else:
                device = self.device
            
            self.model = whisper.load_model(self.model_size, device=device)
            self.logger.info(f"Whisper model loaded successfully on device: {device}")
            
        except ImportError:
            self.logger.error("OpenAI Whisper not installed. Please install with: pip install openai-whisper")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Whisper model: {e}")
            raise
    
    def transcribe_segment(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Transcribe an audio segment to text.
        
        Args:
            audio_data: Audio data as numpy array (16kHz, mono)
            
        Returns:
            Transcribed text or None if transcription failed
        """
        if self.model is None:
            self.logger.error("Whisper model not initialized")
            return None
            
        try:
            import time
            start_time = time.time()
            
            # Ensure audio data is in the correct format
            if len(audio_data) == 0:
                return None
            
            # Quick silence detection to avoid processing silent segments
            audio_level = np.abs(audio_data).mean()
            if audio_level < 0.001:  # Very quiet audio, likely silence
                self.logger.debug("Skipping silent audio segment")
                return None
                
            # Convert to float32 and normalize if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                
            # Normalize audio to [-1, 1] range if it's in int16 range
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / 32768.0
            
            # Ensure minimum length for Whisper (at least 0.1 seconds at 16kHz)
            min_samples = int(0.1 * config.audio.sample_rate)
            if len(audio_data) < min_samples:
                # Pad with zeros
                audio_data = np.pad(audio_data, (0, min_samples - len(audio_data)))
            
            # Optimize Whisper parameters based on model size
            whisper_options = {
                "task": "transcribe",
                "fp16": False,  # Use fp32 for better compatibility
                "verbose": False
            }
            
            # Only set language if it's specified and valid
            if self.language and self.language.strip():
                whisper_options["language"] = self.language
            
            # Add performance optimizations for smaller models
            if self.model_size in ["tiny", "base"]:
                whisper_options.update({
                    "beam_size": 1,  # Faster decoding
                    "best_of": 1,    # Single pass
                    "temperature": 0.0  # Deterministic output
                })
            
            # Transcribe using Whisper
            result = self.model.transcribe(audio_data, **whisper_options)
            
            # Extract text
            text = result.get("text", "").strip()
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_count += 1
            self.total_processing_time += processing_time
            
            # Log performance for optimization
            if processing_time > 3.0:  # Slow transcription
                self.logger.warning(f"Slow transcription: {processing_time:.2f}s for {len(audio_data)} samples")
            
            if text:
                self.logger.debug(f"Transcribed: '{text}' (took {processing_time:.2f}s)")
                return text
            else:
                self.logger.debug("No speech detected in audio segment")
                return None
                
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return None
    
    def transcribe_file(self, audio_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Transcribe an audio file (for testing or batch processing).
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Full transcription result with timestamps
        """
        if self.model is None:
            self.logger.error("Whisper model not initialized")
            return None
            
        try:
            result = self.model.transcribe(
                audio_file_path,
                language=self.language,
                task="transcribe",
                verbose=True
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"File transcription failed: {e}")
            return None
    
    def set_language(self, language: Optional[str]):
        """Set the transcription language"""
        self.language = language
        self.logger.info(f"Transcription language set to: {language or 'auto-detect'}")
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        try:
            import whisper
            return list(whisper.tokenizer.LANGUAGES.keys())
        except:
            return []
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if self.processing_count == 0:
            return {
                'average_processing_time': 0.0,
                'total_processed': 0,
                'total_processing_time': 0.0
            }
            
        return {
            'average_processing_time': self.total_processing_time / self.processing_count,
            'total_processed': self.processing_count,
            'total_processing_time': self.total_processing_time
        }
    
    def is_processing(self) -> bool:
        """Check if currently processing (always False for this implementation)"""
        return False
    
    def cleanup(self):
        """Clean up resources"""
        if self.model is not None:
            # Whisper models don't need explicit cleanup
            self.model = None
            self.logger.info("Whisper transcriber cleaned up")