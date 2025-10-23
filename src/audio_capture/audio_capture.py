"""
Audio capture module for real-time meeting transcription system.
Provides interface for capturing audio from different sources on macOS.
"""

import pyaudio
import sounddevice as sd
import numpy as np
import threading
import queue
import time
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AudioSourceType(Enum):
    """Enumeration of supported audio source types."""
    MICROPHONE = "microphone"
    SYSTEM_AUDIO = "system_audio"


@dataclass
class AudioDevice:
    """Represents an audio device with its properties."""
    id: int
    name: str
    max_input_channels: int
    default_sample_rate: float
    is_default: bool = False


@dataclass
class AudioSegment:
    """Represents a segment of captured audio data."""
    data: np.ndarray
    timestamp: float
    duration: float
    sample_rate: int
    segment_id: str


class AudioCapture:
    """
    Base class for audio capture from different sources.
    
    Supports capturing audio from microphone and system audio (via BlackHole on macOS).
    Provides real-time buffering and device management capabilities.
    """
    
    def __init__(self, source_type: str = AudioSourceType.MICROPHONE.value, 
                 device_id: Optional[int] = None):
        """
        Initialize AudioCapture with specified source type and device.
        
        Args:
            source_type: Type of audio source ("microphone" or "system_audio")
            device_id: Specific device ID to use (None for default)
        """
        self.source_type = AudioSourceType(source_type)
        self.device_id = device_id
        self.sample_rate = 16000  # Optimal for Whisper
        self.channels = 1  # Mono audio
        self.chunk_size = 1024
        self.buffer_duration = 5.0  # 5 seconds buffer
        
        # Audio stream and threading
        self.stream = None
        self.is_capturing = False
        self.capture_thread = None
        self.audio_queue = queue.Queue()
        
        # PyAudio instance
        self.pyaudio_instance = None
        
        # Callback for processed audio segments
        self.audio_callback: Optional[Callable[[AudioSegment], None]] = None
        
        # Initialize PyAudio
        self._initialize_pyaudio()
        
        # Set default device if not specified
        if self.device_id is None:
            self.device_id = self._get_default_device_id()
    
    def _initialize_pyaudio(self) -> None:
        """Initialize PyAudio instance."""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            logger.info("PyAudio initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PyAudio: {e}")
            raise
    
    def _get_default_device_id(self) -> int:
        """Get the default input device ID for the current source type."""
        try:
            if self.source_type == AudioSourceType.MICROPHONE:
                device_info = self.pyaudio_instance.get_default_input_device_info()
                return device_info['index']
            elif self.source_type == AudioSourceType.SYSTEM_AUDIO:
                # Look for BlackHole device for system audio capture
                devices = self.get_audio_devices()
                for device in devices:
                    if 'blackhole' in device.name.lower():
                        return device.id
                # Fallback to default input device
                device_info = self.pyaudio_instance.get_default_input_device_info()
                return device_info['index']
        except Exception as e:
            logger.error(f"Failed to get default device: {e}")
            return 0  # Fallback to device 0
    
    def get_audio_devices(self) -> List[AudioDevice]:
        """
        Get list of available audio input devices.
        
        Returns:
            List of AudioDevice objects representing available input devices
        """
        devices = []
        
        try:
            device_count = self.pyaudio_instance.get_device_count()
            default_device = self.pyaudio_instance.get_default_input_device_info()
            
            for i in range(device_count):
                try:
                    device_info = self.pyaudio_instance.get_device_info_by_index(i)
                    
                    # Only include devices with input channels
                    if device_info['maxInputChannels'] > 0:
                        device = AudioDevice(
                            id=i,
                            name=device_info['name'],
                            max_input_channels=device_info['maxInputChannels'],
                            default_sample_rate=device_info['defaultSampleRate'],
                            is_default=(i == default_device['index'])
                        )
                        devices.append(device)
                        
                except Exception as e:
                    logger.warning(f"Could not get info for device {i}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to enumerate audio devices: {e}")
            
        return devices
    
    def set_audio_source(self, source_type: str, device_id: Optional[int] = None) -> None:
        """
        Set the audio source type and device.
        
        Args:
            source_type: Type of audio source ("microphone" or "system_audio")
            device_id: Specific device ID to use (None for default)
        """
        # Stop current capture if running
        was_capturing = self.is_capturing
        if was_capturing:
            self.stop_capture()
        
        # Update source configuration
        self.source_type = AudioSourceType(source_type)
        self.device_id = device_id if device_id is not None else self._get_default_device_id()
        
        logger.info(f"Audio source set to {source_type} with device ID {self.device_id}")
        
        # Restart capture if it was running
        if was_capturing:
            self.start_capture()
    
    def set_audio_callback(self, callback: Callable[[AudioSegment], None]) -> None:
        """
        Set callback function to be called when audio segments are ready.
        
        Args:
            callback: Function to call with AudioSegment objects
        """
        self.audio_callback = callback
    
    def start_capture(self) -> None:
        """Start audio capture from the configured source."""
        if self.is_capturing:
            logger.warning("Audio capture is already running")
            return
        
        try:
            # Validate device
            device_info = self.pyaudio_instance.get_device_info_by_index(self.device_id)
            logger.info(f"Starting capture from device: {device_info['name']}")
            
            # Open audio stream
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_id,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_stream_callback
            )
            
            # Start the stream
            self.stream.start_stream()
            self.is_capturing = True
            
            # Start processing thread
            self.capture_thread = threading.Thread(target=self._process_audio_queue)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            logger.info("Audio capture started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            self.is_capturing = False
            raise
    
    def stop_capture(self) -> None:
        """Stop audio capture."""
        if not self.is_capturing:
            logger.warning("Audio capture is not running")
            return
        
        try:
            self.is_capturing = False
            
            # Stop and close stream
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            # Wait for processing thread to finish
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
            
            # Clear audio queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            logger.info("Audio capture stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping audio capture: {e}")
    
    def _audio_stream_callback(self, in_data, frame_count, time_info, status):
        """
        Callback function for PyAudio stream.
        
        Args:
            in_data: Input audio data
            frame_count: Number of frames
            time_info: Timing information
            status: Stream status
        """
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Add to queue for processing
        timestamp = time.time()
        self.audio_queue.put((audio_data, timestamp))
        
        return (None, pyaudio.paContinue)
    
    def _process_audio_queue(self) -> None:
        """Process audio data from the queue and create segments."""
        buffer = []
        buffer_start_time = None
        
        while self.is_capturing:
            try:
                # Get audio data from queue with timeout
                audio_data, timestamp = self.audio_queue.get(timeout=0.1)
                
                # Initialize buffer start time
                if buffer_start_time is None:
                    buffer_start_time = timestamp
                
                # Add to buffer
                buffer.extend(audio_data)
                
                # Check if buffer duration reached
                buffer_duration = len(buffer) / self.sample_rate
                if buffer_duration >= self.buffer_duration:
                    # Create audio segment
                    segment_data = np.array(buffer, dtype=np.int16)
                    segment = AudioSegment(
                        data=segment_data,
                        timestamp=buffer_start_time,
                        duration=buffer_duration,
                        sample_rate=self.sample_rate,
                        segment_id=f"segment_{int(buffer_start_time * 1000)}"
                    )
                    
                    # Call callback if set
                    if self.audio_callback:
                        try:
                            self.audio_callback(segment)
                        except Exception as e:
                            logger.error(f"Error in audio callback: {e}")
                    
                    # Reset buffer
                    buffer = []
                    buffer_start_time = None
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audio queue: {e}")
                break
    
    def is_device_available(self, device_id: int) -> bool:
        """
        Check if a specific audio device is available.
        
        Args:
            device_id: Device ID to check
            
        Returns:
            True if device is available, False otherwise
        """
        try:
            device_info = self.pyaudio_instance.get_device_info_by_index(device_id)
            return device_info['maxInputChannels'] > 0
        except Exception:
            return False
    
    def get_current_device_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the currently selected device.
        
        Returns:
            Dictionary with device information or None if no device selected
        """
        if self.device_id is None:
            return None
        
        try:
            return self.pyaudio_instance.get_device_info_by_index(self.device_id)
        except Exception as e:
            logger.error(f"Failed to get current device info: {e}")
            return None
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.stop_capture()
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
        except Exception:
            pass  # Ignore cleanup errors