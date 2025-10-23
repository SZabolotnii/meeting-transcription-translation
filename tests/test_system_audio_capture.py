"""
Tests for system audio capture module.
Tests system audio functionality and BlackHole integration on macOS.
"""

import pytest
import platform
from unittest.mock import Mock, patch, MagicMock
import subprocess

from src.audio_capture.system_audio_capture import SystemAudioCapture
from src.audio_capture.audio_capture import AudioSourceType


class TestSystemAudioCapture:
    """Test cases for SystemAudioCapture class."""
    
    @pytest.fixture
    def mock_macos_environment(self):
        """Mock macOS environment for testing."""
        with patch('platform.system', return_value='Darwin'):
            with patch('src.audio_capture.audio_capture.pyaudio.PyAudio') as mock_pa:
                mock_instance = Mock()
                mock_pa.return_value = mock_instance
                
                mock_instance.get_device_count.return_value = 4
                mock_instance.get_default_input_device_info.return_value = {
                    'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2
                }
                
                devices = [
                    {'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2, 'defaultSampleRate': 44100.0},
                    {'index': 1, 'name': 'BlackHole 2ch', 'maxInputChannels': 2, 'defaultSampleRate': 48000.0},
                    {'index': 2, 'name': 'Soundflower (2ch)', 'maxInputChannels': 2, 'defaultSampleRate': 44100.0},
                    {'index': 3, 'name': 'USB Microphone', 'maxInputChannels': 1, 'defaultSampleRate': 48000.0}
                ]
                
                def mock_get_device_info(index):
                    if 0 <= index < len(devices):
                        return devices[index]
                    raise Exception(f"Invalid device index: {index}")
                
                mock_instance.get_device_info_by_index.side_effect = mock_get_device_info
                
                yield mock_instance
    
    @pytest.fixture
    def system_audio_capture(self, mock_macos_environment):
        """Create SystemAudioCapture instance for testing."""
        return SystemAudioCapture()
    
    def test_initialization_on_macos(self, mock_macos_environment):
        """Test SystemAudioCapture initialization on macOS."""
        sys_capture = SystemAudioCapture()
        
        assert sys_capture.source_type == AudioSourceType.SYSTEM_AUDIO
        assert sys_capture.blackhole_device_id == 1  # BlackHole detected
        assert sys_capture.device_id == 1  # Auto-selected BlackHole
    
    def test_initialization_non_macos(self):
        """Test SystemAudioCapture initialization on non-macOS systems."""
        with patch('platform.system', return_value='Linux'):
            with pytest.raises(RuntimeError, match="only supported on macOS"):
                SystemAudioCapture()
    
    def test_blackhole_device_detection(self, system_audio_capture):
        """Test detection of BlackHole device."""
        assert system_audio_capture.is_blackhole_installed() is True
        assert system_audio_capture.blackhole_device_id == 1
    
    def test_no_blackhole_device(self):
        """Test behavior when BlackHole is not installed."""
        with patch('platform.system', return_value='Darwin'):
            with patch('src.audio_capture.audio_capture.pyaudio.PyAudio') as mock_pa:
                mock_instance = Mock()
                mock_pa.return_value = mock_instance
                
                # No BlackHole device
                mock_instance.get_device_count.return_value = 2
                mock_instance.get_default_input_device_info.return_value = {
                    'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2
                }
                
                devices = [
                    {'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2, 'defaultSampleRate': 44100.0},
                    {'index': 1, 'name': 'USB Microphone', 'maxInputChannels': 1, 'defaultSampleRate': 48000.0}
                ]
                
                def mock_get_device_info(index):
                    if 0 <= index < len(devices):
                        return devices[index]
                    raise Exception(f"Invalid device index: {index}")
                
                mock_instance.get_device_info_by_index.side_effect = mock_get_device_info
                
                sys_capture = SystemAudioCapture()
                
                assert sys_capture.is_blackhole_installed() is False
                assert sys_capture.blackhole_device_id is None
    
    def test_get_system_audio_devices(self, system_audio_capture):
        """Test getting list of system audio devices."""
        devices = system_audio_capture.get_system_audio_devices()
        
        # Should include virtual audio devices
        device_names = [d.name for d in devices]
        assert 'BlackHole 2ch' in device_names
        assert 'Soundflower (2ch)' in device_names
        
        # Should not include regular microphones
        assert 'Built-in Input' not in device_names
        assert 'USB Microphone' not in device_names
    
    def test_get_blackhole_installation_instructions(self, system_audio_capture):
        """Test getting BlackHole installation instructions."""
        instructions = system_audio_capture.get_blackhole_installation_instructions()
        
        assert "BlackHole" in instructions
        assert "github.com/ExistentialAudio/BlackHole" in instructions
        assert "Audio MIDI Setup" in instructions
        assert "Multi-Output Device" in instructions
    
    def test_setup_system_audio_routing(self, system_audio_capture):
        """Test system audio routing setup."""
        success = system_audio_capture.setup_system_audio_routing()
        
        # Should return True (simplified implementation)
        assert success is True
    
    def test_setup_system_audio_routing_no_blackhole(self):
        """Test system audio routing setup without BlackHole."""
        with patch('platform.system', return_value='Darwin'):
            with patch('src.audio_capture.audio_capture.pyaudio.PyAudio') as mock_pa:
                mock_instance = Mock()
                mock_pa.return_value = mock_instance
                
                # No BlackHole
                mock_instance.get_device_count.return_value = 1
                mock_instance.get_default_input_device_info.return_value = {
                    'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2
                }
                mock_instance.get_device_info_by_index.return_value = {
                    'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2
                }
                
                sys_capture = SystemAudioCapture()
                success = sys_capture.setup_system_audio_routing()
                
                assert success is False
    
    def test_get_current_output_device(self, system_audio_capture):
        """Test getting current system output device information."""
        with patch('subprocess.run') as mock_run:
            # Mock successful system_profiler call
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = '{"SPAudioDataType": []}'
            mock_run.return_value = mock_result
            
            device_info = system_audio_capture.get_current_output_device()
            
            assert device_info is not None
            assert "status" in device_info
    
    def test_get_current_output_device_failure(self, system_audio_capture):
        """Test getting current output device when system_profiler fails."""
        with patch('subprocess.run') as mock_run:
            # Mock failed system_profiler call
            mock_run.side_effect = subprocess.TimeoutExpired('system_profiler', 10)
            
            device_info = system_audio_capture.get_current_output_device()
            
            assert device_info is None
    
    def test_switch_to_blackhole_output(self, system_audio_capture):
        """Test switching system output to BlackHole."""
        success = system_audio_capture.switch_to_blackhole_output()
        
        # Should return True (simplified implementation)
        assert success is True
    
    def test_switch_to_blackhole_output_not_installed(self):
        """Test switching to BlackHole when not installed."""
        with patch('platform.system', return_value='Darwin'):
            with patch('src.audio_capture.audio_capture.pyaudio.PyAudio') as mock_pa:
                mock_instance = Mock()
                mock_pa.return_value = mock_instance
                
                # No BlackHole
                mock_instance.get_device_count.return_value = 1
                mock_instance.get_default_input_device_info.return_value = {
                    'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2
                }
                mock_instance.get_device_info_by_index.return_value = {
                    'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2
                }
                
                sys_capture = SystemAudioCapture()
                success = sys_capture.switch_to_blackhole_output()
                
                assert success is False
    
    def test_restore_original_output(self, system_audio_capture):
        """Test restoring original output device."""
        # Set original device
        system_audio_capture.original_output_device = {"name": "Built-in Output"}
        
        success = system_audio_capture.restore_original_output()
        
        # Should return True (simplified implementation)
        assert success is True
    
    def test_restore_original_output_none(self, system_audio_capture):
        """Test restoring original output when none was stored."""
        success = system_audio_capture.restore_original_output()
        
        assert success is False
    
    def test_system_audio_capture_test_success(self, system_audio_capture, mock_macos_environment):
        """Test successful system audio capture test."""
        mock_stream = Mock()
        mock_macos_environment.open.return_value = mock_stream
        
        with patch('time.sleep'):  # Speed up test
            success = system_audio_capture.test_system_audio_capture(duration=0.1)
        
        assert success is True
        assert mock_stream.start_stream.called
        assert mock_stream.stop_stream.called
    
    def test_system_audio_capture_test_no_blackhole(self):
        """Test system audio capture test without BlackHole."""
        with patch('platform.system', return_value='Darwin'):
            with patch('src.audio_capture.audio_capture.pyaudio.PyAudio') as mock_pa:
                mock_instance = Mock()
                mock_pa.return_value = mock_instance
                
                # No BlackHole
                mock_instance.get_device_count.return_value = 1
                mock_instance.get_default_input_device_info.return_value = {
                    'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2
                }
                mock_instance.get_device_info_by_index.return_value = {
                    'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2
                }
                
                sys_capture = SystemAudioCapture()
                success = sys_capture.test_system_audio_capture()
                
                assert success is False
    
    def test_system_audio_capture_test_while_capturing(self, system_audio_capture):
        """Test system audio capture test while already capturing."""
        system_audio_capture.is_capturing = True
        
        success = system_audio_capture.test_system_audio_capture()
        
        assert success is False
    
    def test_get_system_audio_status(self, system_audio_capture):
        """Test getting comprehensive system audio status."""
        status = system_audio_capture.get_system_audio_status()
        
        assert status['blackhole_installed'] is True
        assert status['blackhole_device_id'] == 1
        assert status['current_device_id'] == 1
        assert status['is_capturing'] is False
        assert status['platform'] == 'Darwin'
        assert status['available_system_devices'] == 2  # BlackHole + Soundflower
    
    def test_auto_switch_audio_source_with_blackhole(self, system_audio_capture):
        """Test auto-switching to system audio when BlackHole is available."""
        success = system_audio_capture.auto_switch_audio_source()
        
        assert success is True
        assert system_audio_capture.source_type == AudioSourceType.SYSTEM_AUDIO
        assert system_audio_capture.device_id == 1  # BlackHole device
    
    def test_auto_switch_audio_source_no_blackhole(self):
        """Test auto-switching when BlackHole is not available."""
        with patch('platform.system', return_value='Darwin'):
            with patch('src.audio_capture.audio_capture.pyaudio.PyAudio') as mock_pa:
                mock_instance = Mock()
                mock_pa.return_value = mock_instance
                
                # No BlackHole, but has microphone
                mock_instance.get_device_count.return_value = 1
                mock_instance.get_default_input_device_info.return_value = {
                    'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2
                }
                mock_instance.get_device_info_by_index.return_value = {
                    'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2
                }
                
                sys_capture = SystemAudioCapture()
                success = sys_capture.auto_switch_audio_source()
                
                assert success is True
                assert sys_capture.source_type == AudioSourceType.MICROPHONE


class TestSystemAudioErrorHandling:
    """Test error handling for system audio capture."""
    
    def test_blackhole_detection_with_device_error(self):
        """Test BlackHole detection when device enumeration has errors."""
        with patch('platform.system', return_value='Darwin'):
            with patch('src.audio_capture.audio_capture.pyaudio.PyAudio') as mock_pa:
                mock_instance = Mock()
                mock_pa.return_value = mock_instance
                
                mock_instance.get_device_count.return_value = 2
                mock_instance.get_default_input_device_info.return_value = {
                    'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2
                }
                
                # First device works, second fails
                def mock_get_device_info(index):
                    if index == 0:
                        return {'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2, 'defaultSampleRate': 44100.0}
                    else:
                        raise Exception("Device error")
                
                mock_instance.get_device_info_by_index.side_effect = mock_get_device_info
                
                sys_capture = SystemAudioCapture()
                
                # Should handle error gracefully
                assert sys_capture.blackhole_device_id is None
                assert not sys_capture.is_blackhole_installed()
    
    def test_system_profiler_timeout(self):
        """Test handling system_profiler timeout."""
        with patch('platform.system', return_value='Darwin'):
            with patch('src.audio_capture.audio_capture.pyaudio.PyAudio') as mock_pa:
                mock_instance = Mock()
                mock_pa.return_value = mock_instance
                
                mock_instance.get_device_count.return_value = 1
                mock_instance.get_default_input_device_info.return_value = {
                    'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2
                }
                mock_instance.get_device_info_by_index.return_value = {
                    'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2
                }
                
                sys_capture = SystemAudioCapture()
                
                with patch('subprocess.run') as mock_run:
                    mock_run.side_effect = subprocess.TimeoutExpired('system_profiler', 10)
                    
                    device_info = sys_capture.get_current_output_device()
                    
                    assert device_info is None
    
    def test_system_profiler_invalid_json(self):
        """Test handling invalid JSON from system_profiler."""
        with patch('platform.system', return_value='Darwin'):
            with patch('src.audio_capture.audio_capture.pyaudio.PyAudio') as mock_pa:
                mock_instance = Mock()
                mock_pa.return_value = mock_instance
                
                mock_instance.get_device_count.return_value = 1
                mock_instance.get_default_input_device_info.return_value = {
                    'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2
                }
                mock_instance.get_device_info_by_index.return_value = {
                    'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2
                }
                
                sys_capture = SystemAudioCapture()
                
                with patch('subprocess.run') as mock_run:
                    mock_result = Mock()
                    mock_result.returncode = 0
                    mock_result.stdout = 'invalid json'
                    mock_run.return_value = mock_result
                    
                    device_info = sys_capture.get_current_output_device()
                    
                    assert device_info is None
    
    def test_capture_test_failure(self):
        """Test system audio capture test failure."""
        with patch('platform.system', return_value='Darwin'):
            with patch('src.audio_capture.audio_capture.pyaudio.PyAudio') as mock_pa:
                mock_instance = Mock()
                mock_pa.return_value = mock_instance
                
                mock_instance.get_device_count.return_value = 2
                mock_instance.get_default_input_device_info.return_value = {
                    'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2
                }
                
                devices = [
                    {'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2, 'defaultSampleRate': 44100.0},
                    {'index': 1, 'name': 'BlackHole 2ch', 'maxInputChannels': 2, 'defaultSampleRate': 48000.0}
                ]
                
                def mock_get_device_info(index):
                    if 0 <= index < len(devices):
                        return devices[index]
                    raise Exception(f"Invalid device index: {index}")
                
                mock_instance.get_device_info_by_index.side_effect = mock_get_device_info
                
                # Stream opening fails
                mock_instance.open.side_effect = Exception("Cannot open stream")
                
                sys_capture = SystemAudioCapture()
                success = sys_capture.test_system_audio_capture(duration=0.1)
                
                assert success is False


class TestSystemAudioIntegration:
    """Integration tests for system audio capture."""
    
    @pytest.fixture
    def mock_complete_system(self):
        """Mock complete system with multiple virtual audio devices."""
        with patch('platform.system', return_value='Darwin'):
            with patch('src.audio_capture.audio_capture.pyaudio.PyAudio') as mock_pa:
                mock_instance = Mock()
                mock_pa.return_value = mock_instance
                
                mock_instance.get_device_count.return_value = 6
                mock_instance.get_default_input_device_info.return_value = {
                    'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2
                }
                
                devices = [
                    {'index': 0, 'name': 'Built-in Input', 'maxInputChannels': 2, 'defaultSampleRate': 44100.0},
                    {'index': 1, 'name': 'BlackHole 2ch', 'maxInputChannels': 2, 'defaultSampleRate': 48000.0},
                    {'index': 2, 'name': 'BlackHole 16ch', 'maxInputChannels': 16, 'defaultSampleRate': 48000.0},
                    {'index': 3, 'name': 'Soundflower (2ch)', 'maxInputChannels': 2, 'defaultSampleRate': 44100.0},
                    {'index': 4, 'name': 'Loopback Audio', 'maxInputChannels': 2, 'defaultSampleRate': 48000.0},
                    {'index': 5, 'name': 'USB Microphone', 'maxInputChannels': 1, 'defaultSampleRate': 48000.0}
                ]
                
                def mock_get_device_info(index):
                    if 0 <= index < len(devices):
                        return devices[index]
                    raise Exception(f"Invalid device index: {index}")
                
                mock_instance.get_device_info_by_index.side_effect = mock_get_device_info
                
                yield mock_instance
    
    def test_multiple_virtual_audio_devices(self, mock_complete_system):
        """Test handling multiple virtual audio devices."""
        sys_capture = SystemAudioCapture()
        
        # Should detect first BlackHole device
        assert sys_capture.blackhole_device_id == 1
        assert sys_capture.is_blackhole_installed()
        
        # Should list all virtual audio devices
        devices = sys_capture.get_system_audio_devices()
        device_names = [d.name for d in devices]
        
        assert 'BlackHole 2ch' in device_names
        assert 'BlackHole 16ch' in device_names
        assert 'Soundflower (2ch)' in device_names
        assert 'Loopback Audio' in device_names
        
        # Should not include regular microphones
        assert 'Built-in Input' not in device_names
        assert 'USB Microphone' not in device_names
    
    def test_system_audio_status_comprehensive(self, mock_complete_system):
        """Test comprehensive system audio status reporting."""
        sys_capture = SystemAudioCapture()
        status = sys_capture.get_system_audio_status()
        
        assert status['blackhole_installed'] is True
        assert status['blackhole_device_id'] == 1
        assert status['platform'] == 'Darwin'
        assert status['available_system_devices'] == 4  # All virtual devices
        assert 'device_name' in status
        assert status['device_name'] == 'BlackHole 2ch'


if __name__ == "__main__":
    pytest.main([__file__])