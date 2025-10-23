"""
Tests for microphone capture module.
Tests microphone-specific functionality and device handling.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from src.audio_capture.microphone_capture import MicrophoneCapture
from src.audio_capture.audio_capture import AudioDevice, AudioSourceType


class TestMicrophoneCapture:
    """Test cases for MicrophoneCapture class."""
    
    @pytest.fixture
    def mock_pyaudio(self):
        """Mock PyAudio for microphone testing."""
        with patch('src.audio_capture.audio_capture.pyaudio.PyAudio') as mock_pa:
            mock_instance = Mock()
            mock_pa.return_value = mock_instance
            
            # Mock microphone devices
            mock_instance.get_device_count.return_value = 5
            mock_instance.get_default_input_device_info.return_value = {
                'index': 0, 'name': 'Built-in Microphone', 'maxInputChannels': 2
            }
            
            devices = [
                {'index': 0, 'name': 'Built-in Microphone', 'maxInputChannels': 2, 'defaultSampleRate': 44100.0},
                {'index': 1, 'name': 'USB Microphone', 'maxInputChannels': 1, 'defaultSampleRate': 48000.0},
                {'index': 2, 'name': 'BlackHole 2ch', 'maxInputChannels': 2, 'defaultSampleRate': 48000.0},
                {'index': 3, 'name': 'Headset Mic', 'maxInputChannels': 1, 'defaultSampleRate': 16000.0},
                {'index': 4, 'name': 'Soundflower', 'maxInputChannels': 2, 'defaultSampleRate': 44100.0}
            ]
            
            def mock_get_device_info(index):
                if 0 <= index < len(devices):
                    return devices[index]
                raise Exception(f"Invalid device index: {index}")
            
            mock_instance.get_device_info_by_index.side_effect = mock_get_device_info
            
            yield mock_instance
    
    @pytest.fixture
    def microphone_capture(self, mock_pyaudio):
        """Create MicrophoneCapture instance for testing."""
        return MicrophoneCapture(device_id=0)
    
    def test_initialization(self, mock_pyaudio):
        """Test MicrophoneCapture initialization."""
        mic_capture = MicrophoneCapture(device_id=1)
        
        assert mic_capture.source_type == AudioSourceType.MICROPHONE
        assert mic_capture.device_id == 1
        assert mic_capture.auto_gain_control is True
        assert mic_capture.noise_suppression is True
    
    def test_initialization_default_device(self, mock_pyaudio):
        """Test initialization with default device selection."""
        mic_capture = MicrophoneCapture()
        
        # Should use default microphone device
        assert mic_capture.device_id == 0
        assert mic_capture.source_type == AudioSourceType.MICROPHONE
    
    def test_get_microphone_devices(self, microphone_capture):
        """Test filtering microphone devices from all audio devices."""
        mic_devices = microphone_capture.get_microphone_devices()
        
        # Should filter out virtual audio devices
        device_names = [device.name for device in mic_devices]
        
        assert 'Built-in Microphone' in device_names
        assert 'USB Microphone' in device_names
        assert 'Headset Mic' in device_names
        # Virtual devices should be excluded
        assert 'BlackHole 2ch' not in device_names
        assert 'Soundflower' not in device_names
    
    def test_get_microphone_devices_empty_list(self, mock_pyaudio):
        """Test behavior when no microphone devices are available."""
        # Mock no input devices
        mock_pyaudio.get_device_count.return_value = 0
        
        mic_capture = MicrophoneCapture()
        devices = mic_capture.get_microphone_devices()
        
        assert len(devices) == 0
    
    def test_set_microphone_device_valid(self, microphone_capture):
        """Test setting a valid microphone device."""
        success = microphone_capture.set_microphone_device(1)
        
        assert success is True
        assert microphone_capture.device_id == 1
        assert microphone_capture.source_type == AudioSourceType.MICROPHONE
    
    def test_set_microphone_device_invalid(self, microphone_capture):
        """Test setting an invalid microphone device."""
        success = microphone_capture.set_microphone_device(99)
        
        assert success is False
        # Device ID should remain unchanged
        assert microphone_capture.device_id == 0
    
    def test_set_microphone_device_virtual_audio(self, microphone_capture):
        """Test setting a virtual audio device as microphone (should warn)."""
        # BlackHole device (index 2)
        success = microphone_capture.set_microphone_device(2)
        
        # Should succeed but log warning
        assert success is True
        assert microphone_capture.device_id == 2
    
    def test_microphone_level_not_capturing(self, microphone_capture):
        """Test getting microphone level when not capturing."""
        level = microphone_capture.get_microphone_level()
        
        # Should return 0.0 when not capturing
        assert level == 0.0
    
    def test_microphone_level_capturing_empty_queue(self, microphone_capture):
        """Test getting microphone level when capturing but queue is empty."""
        microphone_capture.is_capturing = True
        
        level = microphone_capture.get_microphone_level()
        
        # Should return 0.0 when queue is empty
        assert level == 0.0
    
    def test_test_microphone_success(self, microphone_capture, mock_pyaudio):
        """Test successful microphone testing."""
        mock_stream = Mock()
        mock_pyaudio.open.return_value = mock_stream
        
        with patch('time.sleep'):  # Speed up test
            success = microphone_capture.test_microphone(duration=0.1)
        
        assert success is True
        # Should have started and stopped capture
        assert mock_stream.start_stream.called
        assert mock_stream.stop_stream.called
    
    def test_test_microphone_while_capturing(self, microphone_capture):
        """Test microphone testing while already capturing."""
        microphone_capture.is_capturing = True
        
        success = microphone_capture.test_microphone()
        
        assert success is False
    
    def test_test_microphone_failure(self, microphone_capture, mock_pyaudio):
        """Test microphone testing with capture failure."""
        mock_pyaudio.open.side_effect = Exception("Cannot open microphone")
        
        success = microphone_capture.test_microphone(duration=0.1)
        
        assert success is False
    
    def test_enable_auto_gain_control(self, microphone_capture):
        """Test enabling/disabling auto gain control."""
        # Test enabling
        microphone_capture.enable_auto_gain_control(True)
        assert microphone_capture.auto_gain_control is True
        
        # Test disabling
        microphone_capture.enable_auto_gain_control(False)
        assert microphone_capture.auto_gain_control is False
    
    def test_enable_noise_suppression(self, microphone_capture):
        """Test enabling/disabling noise suppression."""
        # Test enabling
        microphone_capture.enable_noise_suppression(True)
        assert microphone_capture.noise_suppression is True
        
        # Test disabling
        microphone_capture.enable_noise_suppression(False)
        assert microphone_capture.noise_suppression is False
    
    def test_get_recommended_settings(self, microphone_capture):
        """Test getting recommended microphone settings."""
        settings = microphone_capture.get_recommended_settings()
        
        expected_settings = {
            'sample_rate': 16000,
            'channels': 1,
            'buffer_duration': 5.0,
            'auto_gain_control': True,
            'noise_suppression': True,
            'format': 'int16'
        }
        
        assert settings == expected_settings


class TestMicrophoneErrorHandling:
    """Test error handling for microphone capture."""
    
    def test_device_enumeration_error(self):
        """Test handling device enumeration errors."""
        with patch('src.audio_capture.audio_capture.pyaudio.PyAudio') as mock_pa:
            mock_instance = Mock()
            mock_pa.return_value = mock_instance
            
            # Mock device enumeration failure
            mock_instance.get_device_count.side_effect = Exception("Device enumeration failed")
            
            mic_capture = MicrophoneCapture()
            devices = mic_capture.get_microphone_devices()
            
            # Should return empty list on error
            assert devices == []
    
    def test_device_info_error_during_filtering(self):
        """Test handling device info errors during filtering."""
        with patch('src.audio_capture.audio_capture.pyaudio.PyAudio') as mock_pa:
            mock_instance = Mock()
            mock_pa.return_value = mock_instance
            
            mock_instance.get_device_count.return_value = 2
            mock_instance.get_default_input_device_info.return_value = {
                'index': 0, 'name': 'Default', 'maxInputChannels': 1
            }
            
            # First device succeeds, second fails
            def mock_get_device_info(index):
                if index == 0:
                    return {'index': 0, 'name': 'Good Device', 'maxInputChannels': 1, 'defaultSampleRate': 44100.0}
                else:
                    raise Exception("Device info error")
            
            mock_instance.get_device_info_by_index.side_effect = mock_get_device_info
            
            mic_capture = MicrophoneCapture()
            devices = mic_capture.get_microphone_devices()
            
            # Should only return the working device
            assert len(devices) == 1
            assert devices[0].name == 'Good Device'
    
    def test_microphone_unavailable_during_test(self):
        """Test microphone test when device becomes unavailable."""
        with patch('src.audio_capture.audio_capture.pyaudio.PyAudio') as mock_pa:
            mock_instance = Mock()
            mock_pa.return_value = mock_instance
            
            mock_instance.get_device_count.return_value = 1
            mock_instance.get_default_input_device_info.return_value = {
                'index': 0, 'name': 'Test Mic', 'maxInputChannels': 1
            }
            mock_instance.get_device_info_by_index.return_value = {
                'index': 0, 'name': 'Test Mic', 'maxInputChannels': 1
            }
            
            # Stream opening fails (device unavailable)
            mock_instance.open.side_effect = Exception("Device unavailable")
            
            mic_capture = MicrophoneCapture()
            success = mic_capture.test_microphone(duration=0.1)
            
            assert success is False
    
    def test_set_device_with_pyaudio_error(self):
        """Test setting device when PyAudio has errors."""
        with patch('src.audio_capture.audio_capture.pyaudio.PyAudio') as mock_pa:
            mock_instance = Mock()
            mock_pa.return_value = mock_instance
            
            mock_instance.get_device_count.return_value = 1
            mock_instance.get_default_input_device_info.return_value = {
                'index': 0, 'name': 'Default', 'maxInputChannels': 1
            }
            
            # Device info fails
            mock_instance.get_device_info_by_index.side_effect = Exception("PyAudio error")
            
            mic_capture = MicrophoneCapture()
            success = mic_capture.set_microphone_device(0)
            
            assert success is False


class TestMicrophoneIntegration:
    """Integration tests for microphone capture."""
    
    @pytest.fixture
    def mock_microphone_environment(self):
        """Setup mock environment with various microphone types."""
        with patch('src.audio_capture.audio_capture.pyaudio.PyAudio') as mock_pa:
            mock_instance = Mock()
            mock_pa.return_value = mock_instance
            
            mock_instance.get_device_count.return_value = 6
            mock_instance.get_default_input_device_info.return_value = {
                'index': 0, 'name': 'Built-in Microphone', 'maxInputChannels': 2
            }
            
            devices = [
                {'index': 0, 'name': 'Built-in Microphone', 'maxInputChannels': 2, 'defaultSampleRate': 44100.0},
                {'index': 1, 'name': 'USB Microphone Pro', 'maxInputChannels': 1, 'defaultSampleRate': 48000.0},
                {'index': 2, 'name': 'Wireless Headset Mic', 'maxInputChannels': 1, 'defaultSampleRate': 16000.0},
                {'index': 3, 'name': 'BlackHole 2ch', 'maxInputChannels': 2, 'defaultSampleRate': 48000.0},
                {'index': 4, 'name': 'Broken Microphone', 'maxInputChannels': 0, 'defaultSampleRate': 0.0},
                {'index': 5, 'name': 'Internal Mic', 'maxInputChannels': 1, 'defaultSampleRate': 44100.0}
            ]
            
            def mock_get_device_info(index):
                if 0 <= index < len(devices):
                    return devices[index]
                raise Exception(f"Invalid device index: {index}")
            
            mock_instance.get_device_info_by_index.side_effect = mock_get_device_info
            
            yield mock_instance
    
    def test_microphone_device_selection_priority(self, mock_microphone_environment):
        """Test microphone device selection prioritizes real microphones."""
        mic_capture = MicrophoneCapture()
        devices = mic_capture.get_microphone_devices()
        
        # Should include real microphones
        device_names = [d.name for d in devices]
        assert 'Built-in Microphone' in device_names
        assert 'USB Microphone Pro' in device_names
        assert 'Wireless Headset Mic' in device_names
        assert 'Internal Mic' in device_names
        
        # Should exclude virtual audio devices
        assert 'BlackHole 2ch' not in device_names
        
        # Should exclude devices with no input channels
        assert 'Broken Microphone' not in device_names
    
    def test_microphone_switching_between_devices(self, mock_microphone_environment):
        """Test switching between different microphone devices."""
        mic_capture = MicrophoneCapture()
        
        # Start with built-in microphone
        assert mic_capture.device_id == 0
        
        # Switch to USB microphone
        success = mic_capture.set_microphone_device(1)
        assert success is True
        assert mic_capture.device_id == 1
        
        # Switch to headset microphone
        success = mic_capture.set_microphone_device(2)
        assert success is True
        assert mic_capture.device_id == 2
        
        # Try to switch to broken device
        success = mic_capture.set_microphone_device(4)
        assert success is False
        # Should remain on previous working device
        assert mic_capture.device_id == 2
    
    def test_microphone_capture_with_different_sample_rates(self, mock_microphone_environment):
        """Test microphone capture adapts to different device sample rates."""
        mic_capture = MicrophoneCapture()
        
        # Test with different devices having different sample rates
        devices_to_test = [0, 1, 2, 5]  # Skip broken device (4)
        
        for device_id in devices_to_test:
            success = mic_capture.set_microphone_device(device_id)
            assert success is True
            
            # Device info should be accessible
            device_info = mic_capture.get_current_device_info()
            assert device_info is not None
            assert device_info['maxInputChannels'] > 0


if __name__ == "__main__":
    pytest.main([__file__])