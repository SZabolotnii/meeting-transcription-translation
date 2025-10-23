"""
Tests for audio manager module.
Tests unified audio management and automatic source switching.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import platform

from src.audio_capture.audio_manager import AudioManager, AudioSourcePriority
from src.audio_capture.audio_capture import AudioSourceType


class TestAudioManager:
    """Test cases for AudioManager class."""
    
    @pytest.fixture
    def mock_audio_environment(self):
        """Mock complete audio environment for testing."""
        with patch('src.audio_capture.microphone_capture.MicrophoneCapture') as mock_mic:
            with patch('src.audio_capture.system_audio_capture.SystemAudioCapture') as mock_sys:
                with patch('platform.system', return_value='Darwin'):
                    
                    # Mock microphone capture
                    mock_mic_instance = Mock()
                    mock_mic_instance.is_capturing = False
                    mock_mic_instance.device_id = 0
                    mock_mic_instance.source_type = AudioSourceType.MICROPHONE
                    mock_mic.return_value = mock_mic_instance
                    
                    # Mock system audio capture
                    mock_sys_instance = Mock()
                    mock_sys_instance.is_capturing = False
                    mock_sys_instance.device_id = 1
                    mock_sys_instance.source_type = AudioSourceType.SYSTEM_AUDIO
                    mock_sys_instance.is_blackhole_installed.return_value = True
                    mock_sys.return_value = mock_sys_instance
                    
                    yield {
                        'microphone': mock_mic_instance,
                        'system_audio': mock_sys_instance,
                        'mic_class': mock_mic,
                        'sys_class': mock_sys
                    }
    
    @pytest.fixture
    def audio_manager(self, mock_audio_environment):
        """Create AudioManager instance for testing."""
        return AudioManager(priority=AudioSourcePriority.AUTO_DETECT)
    
    def test_initialization_auto_detect(self, mock_audio_environment):
        """Test AudioManager initialization with auto-detect priority."""
        manager = AudioManager(priority=AudioSourcePriority.AUTO_DETECT)
        
        assert manager.priority == AudioSourcePriority.AUTO_DETECT
        assert manager.microphone_capture is not None
        assert manager.system_audio_capture is not None
        assert manager.current_capture is None
    
    def test_initialization_microphone_first(self, mock_audio_environment):
        """Test AudioManager initialization with microphone priority."""
        manager = AudioManager(priority=AudioSourcePriority.MICROPHONE_FIRST)
        
        assert manager.priority == AudioSourcePriority.MICROPHONE_FIRST
    
    def test_initialization_system_audio_first(self, mock_audio_environment):
        """Test AudioManager initialization with system audio priority."""
        manager = AudioManager(priority=AudioSourcePriority.SYSTEM_AUDIO_FIRST)
        
        assert manager.priority == AudioSourcePriority.SYSTEM_AUDIO_FIRST
    
    def test_initialization_non_macos(self):
        """Test AudioManager initialization on non-macOS systems."""
        with patch('src.audio_capture.microphone_capture.MicrophoneCapture') as mock_mic:
            with patch('platform.system', return_value='Linux'):
                
                mock_mic_instance = Mock()
                mock_mic.return_value = mock_mic_instance
                
                manager = AudioManager()
                
                # Should have microphone but no system audio
                assert manager.microphone_capture is not None
                assert manager.system_audio_capture is None
    
    def test_get_available_sources(self, audio_manager, mock_audio_environment):
        """Test getting list of available audio sources."""
        # Mock device lists
        mic_devices = [
            Mock(id=0, name='Built-in Microphone', is_default=True),
            Mock(id=2, name='USB Microphone', is_default=False)
        ]
        sys_devices = [
            Mock(id=1, name='BlackHole 2ch', is_default=False)
        ]
        
        mock_audio_environment['microphone'].get_microphone_devices.return_value = mic_devices
        mock_audio_environment['system_audio'].get_system_audio_devices.return_value = sys_devices
        
        sources = audio_manager.get_available_sources()
        
        assert len(sources) == 3
        
        # Check microphone sources
        mic_sources = [s for s in sources if s['type'] == 'microphone']
        assert len(mic_sources) == 2
        assert mic_sources[0]['name'] == 'Built-in Microphone'
        assert mic_sources[0]['is_default'] is True
        
        # Check system audio sources
        sys_sources = [s for s in sources if s['type'] == 'system_audio']
        assert len(sys_sources) == 1
        assert sys_sources[0]['name'] == 'BlackHole 2ch'
    
    def test_select_best_source_microphone_first(self, mock_audio_environment):
        """Test selecting best source with microphone priority."""
        manager = AudioManager(priority=AudioSourcePriority.MICROPHONE_FIRST)
        
        best_source = manager.select_best_source()
        
        assert best_source == mock_audio_environment['microphone']
    
    def test_select_best_source_system_audio_first(self, mock_audio_environment):
        """Test selecting best source with system audio priority."""
        manager = AudioManager(priority=AudioSourcePriority.SYSTEM_AUDIO_FIRST)
        
        best_source = manager.select_best_source()
        
        assert best_source == mock_audio_environment['system_audio']
    
    def test_select_best_source_auto_detect(self, mock_audio_environment):
        """Test selecting best source with auto-detect."""
        manager = AudioManager(priority=AudioSourcePriority.AUTO_DETECT)
        
        best_source = manager.select_best_source()
        
        # Should prefer system audio when BlackHole is available
        assert best_source == mock_audio_environment['system_audio']
    
    def test_select_best_source_no_blackhole(self, mock_audio_environment):
        """Test selecting best source when BlackHole is not available."""
        mock_audio_environment['system_audio'].is_blackhole_installed.return_value = False
        
        manager = AudioManager(priority=AudioSourcePriority.AUTO_DETECT)
        best_source = manager.select_best_source()
        
        # Should fallback to microphone
        assert best_source == mock_audio_environment['microphone']
    
    def test_start_capture_with_microphone(self, audio_manager, mock_audio_environment):
        """Test starting capture with microphone source."""
        success = audio_manager.start_capture(source_type="microphone", device_id=0)
        
        assert success is True
        assert audio_manager.current_capture == mock_audio_environment['microphone']
        mock_audio_environment['microphone'].start_capture.assert_called_once()
    
    def test_start_capture_with_system_audio(self, audio_manager, mock_audio_environment):
        """Test starting capture with system audio source."""
        success = audio_manager.start_capture(source_type="system_audio", device_id=1)
        
        assert success is True
        assert audio_manager.current_capture == mock_audio_environment['system_audio']
        mock_audio_environment['system_audio'].start_capture.assert_called_once()
    
    def test_start_capture_auto_select(self, audio_manager, mock_audio_environment):
        """Test starting capture with auto-selection."""
        success = audio_manager.start_capture()
        
        assert success is True
        # Should auto-select system audio (BlackHole available)
        assert audio_manager.current_capture == mock_audio_environment['system_audio']
    
    def test_start_capture_no_source_available(self, mock_audio_environment):
        """Test starting capture when no sources are available."""
        # Mock no available sources
        mock_audio_environment['microphone'] = None
        mock_audio_environment['system_audio'] = None
        
        with patch('src.audio_capture.microphone_capture.MicrophoneCapture', return_value=None):
            with patch('src.audio_capture.system_audio_capture.SystemAudioCapture', return_value=None):
                manager = AudioManager()
                success = manager.start_capture()
                
                assert success is False
    
    def test_start_capture_failure(self, audio_manager, mock_audio_environment):
        """Test starting capture when capture fails."""
        mock_audio_environment['microphone'].start_capture.side_effect = Exception("Capture failed")
        
        success = audio_manager.start_capture(source_type="microphone")
        
        assert success is False
    
    def test_stop_capture(self, audio_manager, mock_audio_environment):
        """Test stopping audio capture."""
        # Start capture first
        audio_manager.start_capture(source_type="microphone")
        
        success = audio_manager.stop_capture()
        
        assert success is True
        mock_audio_environment['microphone'].stop_capture.assert_called_once()
    
    def test_stop_capture_no_active_capture(self, audio_manager):
        """Test stopping capture when no capture is active."""
        success = audio_manager.stop_capture()
        
        assert success is False
    
    def test_stop_capture_failure(self, audio_manager, mock_audio_environment):
        """Test stopping capture when stop fails."""
        audio_manager.current_capture = mock_audio_environment['microphone']
        mock_audio_environment['microphone'].stop_capture.side_effect = Exception("Stop failed")
        
        success = audio_manager.stop_capture()
        
        assert success is False
    
    def test_is_capturing(self, audio_manager, mock_audio_environment):
        """Test checking if capture is active."""
        # Initially not capturing
        assert audio_manager.is_capturing() is False
        
        # Start capture
        audio_manager.start_capture(source_type="microphone")
        mock_audio_environment['microphone'].is_capturing = True
        
        assert audio_manager.is_capturing() is True
    
    def test_set_audio_callback(self, audio_manager, mock_audio_environment):
        """Test setting audio callback."""
        def test_callback(segment):
            pass
        
        audio_manager.set_audio_callback(test_callback)
        
        assert audio_manager.audio_callback == test_callback
    
    def test_set_audio_callback_with_active_capture(self, audio_manager, mock_audio_environment):
        """Test setting callback with active capture."""
        def test_callback(segment):
            pass
        
        # Start capture first
        audio_manager.start_capture(source_type="microphone")
        
        # Set callback
        audio_manager.set_audio_callback(test_callback)
        
        # Should set callback on current capture
        mock_audio_environment['microphone'].set_audio_callback.assert_called_with(test_callback)
    
    def test_get_current_source_info(self, audio_manager, mock_audio_environment):
        """Test getting current source information."""
        # No active capture
        info = audio_manager.get_current_source_info()
        assert info is None
        
        # Start capture
        audio_manager.start_capture(source_type="microphone")
        
        # Mock device info
        mock_audio_environment['microphone'].get_current_device_info.return_value = {
            'name': 'Built-in Microphone',
            'maxInputChannels': 2
        }
        
        info = audio_manager.get_current_source_info()
        
        assert info is not None
        assert info['source_type'] == 'microphone'
        assert info['device_name'] == 'Built-in Microphone'
    
    def test_switch_source(self, audio_manager, mock_audio_environment):
        """Test switching between audio sources."""
        # Start with microphone
        audio_manager.start_capture(source_type="microphone")
        mock_audio_environment['microphone'].is_capturing = True
        
        # Switch to system audio
        success = audio_manager.switch_source("system_audio", device_id=1)
        
        assert success is True
        # Should stop microphone and start system audio
        mock_audio_environment['microphone'].stop_capture.assert_called_once()
        mock_audio_environment['system_audio'].start_capture.assert_called_once()
    
    def test_switch_source_failure_restore(self, audio_manager, mock_audio_environment):
        """Test switching source failure and restoration."""
        # Start with microphone
        audio_manager.start_capture(source_type="microphone")
        mock_audio_environment['microphone'].is_capturing = True
        
        # Mock system audio start failure
        mock_audio_environment['system_audio'].start_capture.side_effect = Exception("Start failed")
        
        success = audio_manager.switch_source("system_audio")
        
        assert success is False
        # Should attempt to restore microphone
        assert mock_audio_environment['microphone'].start_capture.call_count >= 1
    
    def test_get_system_status(self, audio_manager, mock_audio_environment):
        """Test getting comprehensive system status."""
        # Mock system audio status
        mock_audio_environment['system_audio'].get_system_audio_status.return_value = {
            'blackhole_installed': True,
            'platform': 'Darwin'
        }
        
        status = audio_manager.get_system_status()
        
        assert status['microphone_available'] is True
        assert status['system_audio_available'] is True
        assert status['is_capturing'] is False
        assert 'available_sources' in status
        assert 'system_audio_status' in status
    
    def test_test_all_sources(self, audio_manager, mock_audio_environment):
        """Test testing all available audio sources."""
        # Mock test results
        mock_audio_environment['microphone'].test_microphone.return_value = True
        mock_audio_environment['system_audio'].test_system_audio_capture.return_value = True
        
        results = audio_manager.test_all_sources(duration=0.1)
        
        assert results['microphone'] is True
        assert results['system_audio'] is True
        
        mock_audio_environment['microphone'].test_microphone.assert_called_with(0.1)
        mock_audio_environment['system_audio'].test_system_audio_capture.assert_called_with(0.1)
    
    def test_test_all_sources_with_failures(self, audio_manager, mock_audio_environment):
        """Test testing sources when some fail."""
        # Mock test failures
        mock_audio_environment['microphone'].test_microphone.side_effect = Exception("Mic test failed")
        mock_audio_environment['system_audio'].test_system_audio_capture.return_value = False
        
        results = audio_manager.test_all_sources()
        
        assert results['microphone'] is False
        assert results['system_audio'] is False


class TestAudioManagerErrorHandling:
    """Test error handling scenarios for AudioManager."""
    
    def test_initialization_microphone_failure(self):
        """Test AudioManager when microphone initialization fails."""
        with patch('src.audio_capture.microphone_capture.MicrophoneCapture') as mock_mic:
            with patch('platform.system', return_value='Darwin'):
                mock_mic.side_effect = Exception("Microphone init failed")
                
                manager = AudioManager()
                
                # Should handle gracefully
                assert manager.microphone_capture is None
    
    def test_initialization_system_audio_failure(self):
        """Test AudioManager when system audio initialization fails."""
        with patch('src.audio_capture.microphone_capture.MicrophoneCapture') as mock_mic:
            with patch('src.audio_capture.system_audio_capture.SystemAudioCapture') as mock_sys:
                with patch('platform.system', return_value='Darwin'):
                    
                    mock_mic_instance = Mock()
                    mock_mic.return_value = mock_mic_instance
                    
                    mock_sys.side_effect = Exception("System audio init failed")
                    
                    manager = AudioManager()
                    
                    # Should have microphone but no system audio
                    assert manager.microphone_capture is not None
                    assert manager.system_audio_capture is None
    
    def test_source_selection_all_unavailable(self):
        """Test source selection when all sources are unavailable."""
        with patch('src.audio_capture.microphone_capture.MicrophoneCapture', return_value=None):
            with patch('src.audio_capture.system_audio_capture.SystemAudioCapture', return_value=None):
                manager = AudioManager()
                
                best_source = manager.select_best_source()
                
                assert best_source is None
    
    def test_capture_start_with_invalid_source_type(self, mock_audio_environment):
        """Test starting capture with invalid source type."""
        manager = AudioManager()
        
        success = manager.start_capture(source_type="invalid_source")
        
        assert success is False
    
    def test_device_switching_during_capture(self, mock_audio_environment):
        """Test device switching while capture is active."""
        manager = AudioManager()
        
        # Start capture
        manager.start_capture(source_type="microphone")
        mock_audio_environment['microphone'].is_capturing = True
        
        # Mock stop failure
        mock_audio_environment['microphone'].stop_capture.side_effect = Exception("Stop failed")
        
        # Switch should handle error gracefully
        success = manager.switch_source("system_audio")
        
        # Should still attempt the switch despite stop failure
        assert mock_audio_environment['system_audio'].start_capture.called


class TestAudioManagerIntegration:
    """Integration tests for AudioManager."""
    
    @pytest.fixture
    def mock_complete_environment(self):
        """Mock complete audio environment with multiple devices."""
        with patch('src.audio_capture.microphone_capture.MicrophoneCapture') as mock_mic:
            with patch('src.audio_capture.system_audio_capture.SystemAudioCapture') as mock_sys:
                with patch('platform.system', return_value='Darwin'):
                    
                    # Mock multiple microphone devices
                    mic_devices = [
                        Mock(id=0, name='Built-in Microphone', is_default=True),
                        Mock(id=2, name='USB Microphone', is_default=False),
                        Mock(id=3, name='Headset Microphone', is_default=False)
                    ]
                    
                    # Mock multiple system audio devices
                    sys_devices = [
                        Mock(id=1, name='BlackHole 2ch', is_default=False),
                        Mock(id=4, name='Soundflower', is_default=False)
                    ]
                    
                    mock_mic_instance = Mock()
                    mock_mic_instance.get_microphone_devices.return_value = mic_devices
                    mock_mic_instance.is_capturing = False
                    mock_mic.return_value = mock_mic_instance
                    
                    mock_sys_instance = Mock()
                    mock_sys_instance.get_system_audio_devices.return_value = sys_devices
                    mock_sys_instance.is_blackhole_installed.return_value = True
                    mock_sys_instance.is_capturing = False
                    mock_sys.return_value = mock_sys_instance
                    
                    yield {
                        'microphone': mock_mic_instance,
                        'system_audio': mock_sys_instance,
                        'mic_devices': mic_devices,
                        'sys_devices': sys_devices
                    }
    
    def test_comprehensive_source_management(self, mock_complete_environment):
        """Test comprehensive source management with multiple devices."""
        manager = AudioManager(priority=AudioSourcePriority.AUTO_DETECT)
        
        # Get all available sources
        sources = manager.get_available_sources()
        
        # Should have 5 total sources (3 mic + 2 system)
        assert len(sources) == 5
        
        mic_sources = [s for s in sources if s['type'] == 'microphone']
        sys_sources = [s for s in sources if s['type'] == 'system_audio']
        
        assert len(mic_sources) == 3
        assert len(sys_sources) == 2
    
    def test_priority_based_selection_scenarios(self, mock_complete_environment):
        """Test different priority scenarios."""
        # Test microphone first
        manager_mic = AudioManager(priority=AudioSourcePriority.MICROPHONE_FIRST)
        best_mic = manager_mic.select_best_source()
        assert best_mic == mock_complete_environment['microphone']
        
        # Test system audio first
        manager_sys = AudioManager(priority=AudioSourcePriority.SYSTEM_AUDIO_FIRST)
        best_sys = manager_sys.select_best_source()
        assert best_sys == mock_complete_environment['system_audio']
        
        # Test auto-detect (should prefer system audio when available)
        manager_auto = AudioManager(priority=AudioSourcePriority.AUTO_DETECT)
        best_auto = manager_auto.select_best_source()
        assert best_auto == mock_complete_environment['system_audio']
    
    def test_fallback_behavior(self, mock_complete_environment):
        """Test fallback behavior when preferred source is unavailable."""
        # System audio not available
        mock_complete_environment['system_audio'].is_blackhole_installed.return_value = False
        
        manager = AudioManager(priority=AudioSourcePriority.SYSTEM_AUDIO_FIRST)
        best_source = manager.select_best_source()
        
        # Should fallback to microphone
        assert best_source == mock_complete_environment['microphone']


if __name__ == "__main__":
    pytest.main([__file__])