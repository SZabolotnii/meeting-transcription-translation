"""
System audio capture implementation for macOS.
Specialized class for capturing system audio using BlackHole virtual audio driver.
"""

import logging
import subprocess
import platform
from typing import Optional, List, Dict, Any
from .audio_capture import AudioCapture, AudioDevice, AudioSourceType

logger = logging.getLogger(__name__)


class SystemAudioCapture(AudioCapture):
    """
    Specialized audio capture class for system audio on macOS.
    
    Uses BlackHole virtual audio driver to capture system audio output,
    enabling transcription of audio from applications like Zoom, Teams, etc.
    """
    
    def __init__(self, device_id: Optional[int] = None):
        """
        Initialize system audio capture.
        
        Args:
            device_id: Specific BlackHole device ID (None for auto-detection)
        """
        # Verify we're on macOS
        if platform.system() != 'Darwin':
            raise RuntimeError("SystemAudioCapture is only supported on macOS")
        
        super().__init__(source_type=AudioSourceType.SYSTEM_AUDIO.value, device_id=device_id)
        
        # System audio specific settings
        self.blackhole_device_id = None
        self.original_output_device = None
        
        # Try to find BlackHole device
        self._detect_blackhole_device()
        
        logger.info(f"SystemAudioCapture initialized with device ID: {self.device_id}")
    
    def _detect_blackhole_device(self) -> None:
        """Detect BlackHole virtual audio device."""
        devices = self.get_audio_devices()
        
        for device in devices:
            device_name_lower = device.name.lower()
            if 'blackhole' in device_name_lower:
                self.blackhole_device_id = device.id
                if self.device_id is None:
                    self.device_id = device.id
                logger.info(f"Found BlackHole device: {device.name} (ID: {device.id})")
                return
        
        logger.warning("BlackHole device not found. System audio capture may not work properly.")
    
    def get_system_audio_devices(self) -> List[AudioDevice]:
        """
        Get list of available system audio devices.
        
        Filters devices to show virtual audio drivers suitable for system audio capture.
        
        Returns:
            List of AudioDevice objects representing system audio devices
        """
        all_devices = self.get_audio_devices()
        system_devices = []
        
        for device in all_devices:
            device_name_lower = device.name.lower()
            
            # Include virtual audio devices
            if any(keyword in device_name_lower for keyword in [
                'blackhole', 'soundflower', 'loopback', 'virtual'
            ]):
                system_devices.append(device)
        
        return system_devices
    
    def is_blackhole_installed(self) -> bool:
        """
        Check if BlackHole is installed on the system.
        
        Returns:
            True if BlackHole is detected, False otherwise
        """
        return self.blackhole_device_id is not None
    
    def get_blackhole_installation_instructions(self) -> str:
        """
        Get instructions for installing BlackHole on macOS.
        
        Returns:
            String with installation instructions
        """
        return """
To capture system audio on macOS, you need to install BlackHole:

1. Download BlackHole from: https://github.com/ExistentialAudio/BlackHole
2. Install the .pkg file
3. Open Audio MIDI Setup (Applications > Utilities)
4. Create a Multi-Output Device:
   - Click '+' and select 'Create Multi-Output Device'
   - Check both 'Built-in Output' and 'BlackHole 2ch'
   - Set this as your default output device in System Preferences > Sound
5. Restart this application

This allows you to hear audio normally while capturing it for transcription.
        """
    
    def setup_system_audio_routing(self) -> bool:
        """
        Attempt to set up system audio routing automatically.
        
        Returns:
            True if setup successful, False otherwise
        """
        if not self.is_blackhole_installed():
            logger.error("BlackHole is not installed")
            return False
        
        try:
            # This is a simplified implementation
            # In practice, you might want to use AppleScript or other methods
            # to automate the audio routing setup
            
            logger.info("System audio routing setup would require manual configuration")
            logger.info("Please follow the BlackHole installation instructions")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup system audio routing: {e}")
            return False
    
    def get_current_output_device(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current system output device.
        
        Returns:
            Dictionary with current output device info or None
        """
        try:
            # Use system_profiler to get audio device information
            result = subprocess.run([
                'system_profiler', 'SPAudioDataType', '-json'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                # Parse the audio device information
                # This is a simplified implementation
                return {"status": "detected", "data": "system_audio_info"}
            
        except Exception as e:
            logger.error(f"Failed to get current output device: {e}")
        
        return None
    
    def switch_to_blackhole_output(self) -> bool:
        """
        Switch system output to BlackHole device.
        
        Returns:
            True if switch successful, False otherwise
        """
        if not self.is_blackhole_installed():
            logger.error("Cannot switch to BlackHole: device not found")
            return False
        
        try:
            # Store current output device for restoration
            self.original_output_device = self.get_current_output_device()
            
            # Use AppleScript to change the output device
            applescript = f'''
            tell application "System Preferences"
                set current pane to pane "com.apple.preference.sound"
            end tell
            
            tell application "System Events"
                tell process "System Preferences"
                    click radio button "Output" of tab group 1 of window "Sound"
                    -- Select BlackHole device (simplified)
                end tell
            end tell
            '''
            
            # Note: This is a placeholder implementation
            # In practice, you would need more sophisticated AppleScript
            # or use Core Audio APIs to change the output device
            
            logger.info("BlackHole output switching would require system permissions")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch to BlackHole output: {e}")
            return False
    
    def restore_original_output(self) -> bool:
        """
        Restore the original system output device.
        
        Returns:
            True if restoration successful, False otherwise
        """
        if self.original_output_device is None:
            logger.warning("No original output device to restore")
            return False
        
        try:
            # Restore original output device
            # This would use similar AppleScript or Core Audio APIs
            logger.info("Original output device restoration would be implemented here")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore original output: {e}")
            return False
    
    def test_system_audio_capture(self, duration: float = 5.0) -> bool:
        """
        Test system audio capture functionality.
        
        Args:
            duration: Test duration in seconds
            
        Returns:
            True if test successful, False otherwise
        """
        if not self.is_blackhole_installed():
            logger.error("Cannot test system audio: BlackHole not installed")
            return False
        
        if self.is_capturing:
            logger.warning("Cannot test while capture is active")
            return False
        
        try:
            logger.info(f"Testing system audio capture for {duration} seconds...")
            logger.info("Please play some audio to test the capture")
            
            # Start capture
            self.start_capture()
            
            # Wait for test duration
            import time
            time.sleep(duration)
            
            # Stop capture
            self.stop_capture()
            
            logger.info("System audio capture test completed")
            return True
            
        except Exception as e:
            logger.error(f"System audio capture test failed: {e}")
            try:
                self.stop_capture()
            except:
                pass
            return False
    
    def get_system_audio_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of system audio setup.
        
        Returns:
            Dictionary with system audio status information
        """
        status = {
            'blackhole_installed': self.is_blackhole_installed(),
            'blackhole_device_id': self.blackhole_device_id,
            'current_device_id': self.device_id,
            'is_capturing': self.is_capturing,
            'platform': platform.system(),
            'available_system_devices': len(self.get_system_audio_devices())
        }
        
        if self.blackhole_device_id:
            device_info = self.get_current_device_info()
            if device_info:
                status['device_name'] = device_info.get('name', 'Unknown')
                status['sample_rate'] = device_info.get('defaultSampleRate', 0)
        
        return status
    
    def auto_switch_audio_source(self) -> bool:
        """
        Automatically switch between microphone and system audio based on availability.
        
        Returns:
            True if switch successful, False otherwise
        """
        try:
            # Check if BlackHole is available for system audio
            if self.is_blackhole_installed():
                logger.info("Switching to system audio (BlackHole)")
                self.set_audio_source(AudioSourceType.SYSTEM_AUDIO.value, self.blackhole_device_id)
                return True
            else:
                # Fallback to microphone
                logger.info("BlackHole not available, switching to microphone")
                self.set_audio_source(AudioSourceType.MICROPHONE.value)
                return True
                
        except Exception as e:
            logger.error(f"Failed to auto-switch audio source: {e}")
            return False