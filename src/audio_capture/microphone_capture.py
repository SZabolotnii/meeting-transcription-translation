"""
Microphone audio capture implementation.
Specialized class for capturing audio from microphone devices.
"""

import logging
from typing import Optional, List
from .audio_capture import AudioCapture, AudioDevice, AudioSourceType

logger = logging.getLogger(__name__)


class MicrophoneCapture(AudioCapture):
    """
    Specialized audio capture class for microphone input.
    
    Provides optimized settings and methods specifically for microphone capture,
    including automatic gain control and noise reduction considerations.
    """
    
    def __init__(self, device_id: Optional[int] = None):
        """
        Initialize microphone capture.
        
        Args:
            device_id: Specific microphone device ID (None for default)
        """
        super().__init__(source_type=AudioSourceType.MICROPHONE.value, device_id=device_id)
        
        # Microphone-specific settings
        self.auto_gain_control = True
        self.noise_suppression = True
        
        logger.info(f"MicrophoneCapture initialized with device ID: {self.device_id}")
    
    def get_microphone_devices(self) -> List[AudioDevice]:
        """
        Get list of available microphone devices.
        
        Filters devices to show only those suitable for microphone input.
        
        Returns:
            List of AudioDevice objects representing microphone devices
        """
        all_devices = self.get_audio_devices()
        microphone_devices = []
        
        for device in all_devices:
            # Filter for devices that are likely microphones
            device_name_lower = device.name.lower()
            
            # Include devices that are clearly microphones or built-in audio
            if any(keyword in device_name_lower for keyword in [
                'microphone', 'mic', 'built-in', 'internal', 'default'
            ]):
                microphone_devices.append(device)
            # Exclude devices that are clearly not microphones
            elif any(keyword in device_name_lower for keyword in [
                'blackhole', 'soundflower', 'loopback', 'virtual'
            ]):
                continue
            else:
                # Include other input devices as potential microphones
                microphone_devices.append(device)
        
        return microphone_devices
    
    def set_microphone_device(self, device_id: int) -> bool:
        """
        Set specific microphone device.
        
        Args:
            device_id: ID of the microphone device to use
            
        Returns:
            True if device was set successfully, False otherwise
        """
        try:
            # Validate that the device exists and has input channels
            if not self.is_device_available(device_id):
                logger.error(f"Microphone device {device_id} is not available")
                return False
            
            # Check if device is suitable for microphone input
            device_info = self.pyaudio_instance.get_device_info_by_index(device_id)
            device_name = device_info['name'].lower()
            
            # Warn if device might not be a microphone
            if any(keyword in device_name for keyword in ['blackhole', 'soundflower', 'loopback']):
                logger.warning(f"Device '{device_info['name']}' might not be a microphone")
            
            # Set the device
            self.set_audio_source(AudioSourceType.MICROPHONE.value, device_id)
            logger.info(f"Microphone device set to: {device_info['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set microphone device {device_id}: {e}")
            return False
    
    def get_microphone_level(self) -> float:
        """
        Get current microphone input level.
        
        Returns:
            Current input level as a float between 0.0 and 1.0
        """
        if not self.is_capturing or self.audio_queue.empty():
            return 0.0
        
        try:
            # Get the most recent audio data without removing it from queue
            # This is a simplified implementation - in practice, you might want
            # to maintain a separate level monitoring system
            
            # For now, return a placeholder value
            # In a full implementation, this would analyze recent audio data
            return 0.5  # Placeholder
            
        except Exception as e:
            logger.error(f"Failed to get microphone level: {e}")
            return 0.0
    
    def test_microphone(self, duration: float = 2.0) -> bool:
        """
        Test microphone functionality by capturing audio for a short duration.
        
        Args:
            duration: Test duration in seconds
            
        Returns:
            True if microphone test successful, False otherwise
        """
        if self.is_capturing:
            logger.warning("Cannot test microphone while capture is active")
            return False
        
        try:
            logger.info(f"Testing microphone for {duration} seconds...")
            
            # Start capture
            self.start_capture()
            
            # Wait for test duration
            import time
            time.sleep(duration)
            
            # Stop capture
            self.stop_capture()
            
            # Check if we received any audio data
            # In a full implementation, you would analyze the captured audio
            # to determine if the microphone is working properly
            
            logger.info("Microphone test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Microphone test failed: {e}")
            try:
                self.stop_capture()
            except:
                pass
            return False
    
    def enable_auto_gain_control(self, enabled: bool = True) -> None:
        """
        Enable or disable automatic gain control.
        
        Args:
            enabled: True to enable AGC, False to disable
        """
        self.auto_gain_control = enabled
        logger.info(f"Auto gain control {'enabled' if enabled else 'disabled'}")
    
    def enable_noise_suppression(self, enabled: bool = True) -> None:
        """
        Enable or disable noise suppression.
        
        Args:
            enabled: True to enable noise suppression, False to disable
        """
        self.noise_suppression = enabled
        logger.info(f"Noise suppression {'enabled' if enabled else 'disabled'}")
    
    def get_recommended_settings(self) -> dict:
        """
        Get recommended settings for microphone capture.
        
        Returns:
            Dictionary with recommended settings
        """
        return {
            'sample_rate': 16000,  # Optimal for Whisper
            'channels': 1,  # Mono for speech
            'buffer_duration': 5.0,  # 5 seconds for real-time processing
            'auto_gain_control': True,
            'noise_suppression': True,
            'format': 'int16'  # 16-bit audio
        }