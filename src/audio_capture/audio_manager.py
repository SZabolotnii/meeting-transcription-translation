"""
Audio Manager for handling multiple audio sources and automatic switching.
Provides a unified interface for managing microphone and system audio capture.
"""

import logging
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from .audio_capture import AudioCapture, AudioDevice, AudioSegment, AudioSourceType
from .microphone_capture import MicrophoneCapture
from .system_audio_capture import SystemAudioCapture

logger = logging.getLogger(__name__)


class AudioSourcePriority(Enum):
    """Priority levels for audio source selection."""
    MICROPHONE_FIRST = "microphone_first"
    SYSTEM_AUDIO_FIRST = "system_audio_first"
    AUTO_DETECT = "auto_detect"


class AudioManager:
    """
    Unified audio manager for handling multiple audio sources.
    
    Provides automatic switching between microphone and system audio,
    device management, and unified capture interface.
    """
    
    def __init__(self, priority: AudioSourcePriority = AudioSourcePriority.AUTO_DETECT):
        """
        Initialize AudioManager.
        
        Args:
            priority: Priority for audio source selection
        """
        self.priority = priority
        self.current_capture: Optional[AudioCapture] = None
        self.microphone_capture: Optional[MicrophoneCapture] = None
        self.system_audio_capture: Optional[SystemAudioCapture] = None
        
        # Audio callback
        self.audio_callback: Optional[Callable[[AudioSegment], None]] = None
        
        # Initialize capture instances
        self._initialize_captures()
        
        logger.info(f"AudioManager initialized with priority: {priority.value}")
    
    def _initialize_captures(self) -> None:
        """Initialize microphone and system audio capture instances."""
        try:
            # Initialize microphone capture
            self.microphone_capture = MicrophoneCapture()
            logger.info("Microphone capture initialized")
        except Exception as e:
            logger.error(f"Failed to initialize microphone capture: {e}")
        
        try:
            # Initialize system audio capture (macOS only)
            import platform
            if platform.system() == 'Darwin':
                self.system_audio_capture = SystemAudioCapture()
                logger.info("System audio capture initialized")
            else:
                logger.info("System audio capture not available on this platform")
        except Exception as e:
            logger.error(f"Failed to initialize system audio capture: {e}")
    
    def get_available_sources(self) -> List[Dict[str, Any]]:
        """
        Get list of available audio sources.
        
        Returns:
            List of dictionaries describing available audio sources
        """
        sources = []
        
        # Add microphone sources
        if self.microphone_capture:
            mic_devices = self.microphone_capture.get_microphone_devices()
            for device in mic_devices:
                sources.append({
                    'type': 'microphone',
                    'device_id': device.id,
                    'name': device.name,
                    'is_default': device.is_default,
                    'available': True
                })
        
        # Add system audio sources
        if self.system_audio_capture:
            sys_devices = self.system_audio_capture.get_system_audio_devices()
            for device in sys_devices:
                sources.append({
                    'type': 'system_audio',
                    'device_id': device.id,
                    'name': device.name,
                    'is_default': False,
                    'available': self.system_audio_capture.is_blackhole_installed()
                })
        
        return sources
    
    def select_best_source(self) -> Optional[AudioCapture]:
        """
        Select the best available audio source based on priority.
        
        Returns:
            AudioCapture instance for the selected source or None
        """
        if self.priority == AudioSourcePriority.MICROPHONE_FIRST:
            return self._select_microphone_first()
        elif self.priority == AudioSourcePriority.SYSTEM_AUDIO_FIRST:
            return self._select_system_audio_first()
        else:  # AUTO_DETECT
            return self._auto_detect_source()
    
    def _select_microphone_first(self) -> Optional[AudioCapture]:
        """Select microphone as primary source."""
        if self.microphone_capture:
            logger.info("Selected microphone as primary audio source")
            return self.microphone_capture
        elif self.system_audio_capture and self.system_audio_capture.is_blackhole_installed():
            logger.info("Microphone not available, falling back to system audio")
            return self.system_audio_capture
        return None
    
    def _select_system_audio_first(self) -> Optional[AudioCapture]:
        """Select system audio as primary source."""
        if self.system_audio_capture and self.system_audio_capture.is_blackhole_installed():
            logger.info("Selected system audio as primary audio source")
            return self.system_audio_capture
        elif self.microphone_capture:
            logger.info("System audio not available, falling back to microphone")
            return self.microphone_capture
        return None
    
    def _auto_detect_source(self) -> Optional[AudioCapture]:
        """Auto-detect the best available source."""
        # Check if system audio is properly configured
        if (self.system_audio_capture and 
            self.system_audio_capture.is_blackhole_installed()):
            logger.info("Auto-detected system audio as best source")
            return self.system_audio_capture
        elif self.microphone_capture:
            logger.info("Auto-detected microphone as best source")
            return self.microphone_capture
        return None
    
    def start_capture(self, source_type: Optional[str] = None, 
                     device_id: Optional[int] = None) -> bool:
        """
        Start audio capture with specified or auto-selected source.
        
        Args:
            source_type: Type of source ("microphone" or "system_audio")
            device_id: Specific device ID to use
            
        Returns:
            True if capture started successfully, False otherwise
        """
        try:
            # Stop current capture if running
            if self.current_capture and self.current_capture.is_capturing:
                self.stop_capture()
            
            # Select capture instance
            if source_type == "microphone" and self.microphone_capture:
                self.current_capture = self.microphone_capture
                if device_id is not None:
                    self.microphone_capture.set_microphone_device(device_id)
            elif source_type == "system_audio" and self.system_audio_capture:
                self.current_capture = self.system_audio_capture
                if device_id is not None:
                    self.system_audio_capture.set_audio_source("system_audio", device_id)
            else:
                # Auto-select best source
                self.current_capture = self.select_best_source()
            
            if not self.current_capture:
                logger.error("No audio capture source available")
                return False
            
            # Set callback if configured
            if self.audio_callback:
                self.current_capture.set_audio_callback(self.audio_callback)
            
            # Start capture
            self.current_capture.start_capture()
            logger.info("Audio capture started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            return False
    
    def stop_capture(self) -> bool:
        """
        Stop audio capture.
        
        Returns:
            True if capture stopped successfully, False otherwise
        """
        try:
            if self.current_capture:
                self.current_capture.stop_capture()
                logger.info("Audio capture stopped successfully")
                return True
            else:
                logger.warning("No active capture to stop")
                return False
                
        except Exception as e:
            logger.error(f"Failed to stop audio capture: {e}")
            return False
    
    def is_capturing(self) -> bool:
        """
        Check if audio capture is currently active.
        
        Returns:
            True if capturing, False otherwise
        """
        return (self.current_capture is not None and 
                self.current_capture.is_capturing)
    
    def set_audio_callback(self, callback: Callable[[AudioSegment], None]) -> None:
        """
        Set callback function for audio segments.
        
        Args:
            callback: Function to call with AudioSegment objects
        """
        self.audio_callback = callback
        
        # Set callback on current capture if active
        if self.current_capture:
            self.current_capture.set_audio_callback(callback)
    
    def get_current_source_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the currently active audio source.
        
        Returns:
            Dictionary with current source information or None
        """
        if not self.current_capture:
            return None
        
        info = {
            'is_capturing': self.current_capture.is_capturing,
            'device_id': self.current_capture.device_id,
            'source_type': self.current_capture.source_type.value,
            'sample_rate': self.current_capture.sample_rate,
            'channels': self.current_capture.channels
        }
        
        # Add device-specific information
        device_info = self.current_capture.get_current_device_info()
        if device_info:
            info['device_name'] = device_info.get('name', 'Unknown')
            info['max_input_channels'] = device_info.get('maxInputChannels', 0)
        
        return info
    
    def switch_source(self, source_type: str, device_id: Optional[int] = None) -> bool:
        """
        Switch to a different audio source.
        
        Args:
            source_type: Type of source to switch to
            device_id: Specific device ID to use
            
        Returns:
            True if switch successful, False otherwise
        """
        was_capturing = self.is_capturing()
        
        try:
            # Stop current capture
            if was_capturing:
                self.stop_capture()
            
            # Start with new source
            success = self.start_capture(source_type, device_id)
            
            if success:
                logger.info(f"Successfully switched to {source_type}")
            else:
                logger.error(f"Failed to switch to {source_type}")
                
                # Try to restart previous capture if switch failed
                if was_capturing:
                    logger.info("Attempting to restore previous capture")
                    self.start_capture()
            
            return success
            
        except Exception as e:
            logger.error(f"Error switching audio source: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status for audio capture.
        
        Returns:
            Dictionary with system status information
        """
        status = {
            'microphone_available': self.microphone_capture is not None,
            'system_audio_available': (self.system_audio_capture is not None and 
                                     self.system_audio_capture.is_blackhole_installed()),
            'current_source': None,
            'is_capturing': self.is_capturing(),
            'available_sources': len(self.get_available_sources())
        }
        
        # Add current source info
        current_info = self.get_current_source_info()
        if current_info:
            status['current_source'] = current_info
        
        # Add system audio specific status
        if self.system_audio_capture:
            sys_status = self.system_audio_capture.get_system_audio_status()
            status['system_audio_status'] = sys_status
        
        return status
    
    def test_all_sources(self, duration: float = 2.0) -> Dict[str, bool]:
        """
        Test all available audio sources.
        
        Args:
            duration: Test duration for each source
            
        Returns:
            Dictionary with test results for each source
        """
        results = {}
        
        # Test microphone
        if self.microphone_capture:
            try:
                results['microphone'] = self.microphone_capture.test_microphone(duration)
            except Exception as e:
                logger.error(f"Microphone test failed: {e}")
                results['microphone'] = False
        
        # Test system audio
        if self.system_audio_capture:
            try:
                results['system_audio'] = self.system_audio_capture.test_system_audio_capture(duration)
            except Exception as e:
                logger.error(f"System audio test failed: {e}")
                results['system_audio'] = False
        
        return results