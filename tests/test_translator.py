"""
Tests for translation module.
Tests mock Google Translate API and error handling for network and API failures.
"""

import pytest
import time
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from google.api_core import exceptions as google_exceptions
from google.cloud import translate_v2 as translate

from src.translation.translator import (
    Translator, 
    TranslationResult, 
    TranslationError, 
    TranslationCache
)
from src.config import SUPPORTED_LANGUAGES


class TestTranslationCache:
    """Test cases for TranslationCache class."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def translation_cache(self, temp_cache_dir):
        """Create TranslationCache instance for testing."""
        return TranslationCache(cache_dir=temp_cache_dir)
    
    def test_cache_initialization(self, translation_cache, temp_cache_dir):
        """Test cache initialization creates directory and file."""
        assert translation_cache.cache_dir == Path(temp_cache_dir)
        assert translation_cache.cache_file == Path(temp_cache_dir) / "translation_cache.json"
        assert isinstance(translation_cache.cache, dict)
    
    def test_cache_key_generation(self, translation_cache):
        """Test cache key generation for text and language pairs."""
        key1 = translation_cache._generate_key("hello", "en", "uk")
        key2 = translation_cache._generate_key("hello", "en", "uk")
        key3 = translation_cache._generate_key("hello", "en", "ru")
        
        # Same inputs should generate same key
        assert key1 == key2
        # Different inputs should generate different keys
        assert key1 != key3
        # Keys should be MD5 hashes (32 characters)
        assert len(key1) == 32
    
    def test_cache_set_and_get(self, translation_cache):
        """Test setting and getting cached translations."""
        result = TranslationResult(
            original_text="hello",
            translated_text="привіт",
            source_language="en",
            target_language="uk",
            confidence=0.95,
            timestamp=time.time()
        )
        
        # Set cache
        translation_cache.set(result)
        
        # Get from cache
        cached_result = translation_cache.get("hello", "en", "uk")
        
        assert cached_result is not None
        assert cached_result.original_text == "hello"
        assert cached_result.translated_text == "привіт"
        assert cached_result.source_language == "en"
        assert cached_result.target_language == "uk"
        assert cached_result.cached is True
    
    def test_cache_miss(self, translation_cache):
        """Test cache miss returns None."""
        result = translation_cache.get("nonexistent", "en", "uk")
        assert result is None
    
    def test_cache_persistence(self, temp_cache_dir):
        """Test cache persistence across instances."""
        # Create first cache instance and add data
        cache1 = TranslationCache(cache_dir=temp_cache_dir)
        result = TranslationResult(
            original_text="test",
            translated_text="тест",
            source_language="en",
            target_language="uk",
            timestamp=time.time()
        )
        cache1.set(result)
        
        # Create second cache instance and verify data persists
        cache2 = TranslationCache(cache_dir=temp_cache_dir)
        cached_result = cache2.get("test", "en", "uk")
        
        assert cached_result is not None
        assert cached_result.translated_text == "тест"
    
    def test_cache_clear(self, translation_cache):
        """Test clearing cache."""
        # Add some data
        result = TranslationResult(
            original_text="clear_test",
            translated_text="тест_очищення",
            source_language="en",
            target_language="uk",
            timestamp=time.time()
        )
        translation_cache.set(result)
        
        # Verify data exists
        assert translation_cache.get("clear_test", "en", "uk") is not None
        
        # Clear cache
        translation_cache.clear()
        
        # Verify data is gone
        assert translation_cache.get("clear_test", "en", "uk") is None
        assert len(translation_cache.cache) == 0
    
    def test_cache_file_corruption_handling(self, temp_cache_dir):
        """Test handling of corrupted cache file."""
        cache_file = Path(temp_cache_dir) / "translation_cache.json"
        
        # Create corrupted cache file
        with open(cache_file, 'w') as f:
            f.write("invalid json content")
        
        # Should handle corruption gracefully
        cache = TranslationCache(cache_dir=temp_cache_dir)
        assert isinstance(cache.cache, dict)
        assert len(cache.cache) == 0


class TestTranslator:
    """Test cases for Translator class."""
    
    @pytest.fixture
    def mock_translate_client(self):
        """Mock Google Cloud Translation client."""
        with patch('src.translation.translator.translate.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock successful translation response
            mock_client.translate.return_value = {
                'translatedText': 'привіт світ',
                'detectedSourceLanguage': 'en'
            }
            
            # Mock supported languages
            mock_client.get_languages.return_value = [
                {'language': 'en', 'name': 'English'},
                {'language': 'uk', 'name': 'Ukrainian'},
                {'language': 'ru', 'name': 'Russian'}
            ]
            
            yield mock_client
    
    @pytest.fixture
    def translator(self, mock_translate_client):
        """Create Translator instance for testing."""
        with patch('src.translation.translator.config') as mock_config:
            mock_config.translation.api_key = "test_api_key"
            mock_config.translation.project_id = "test_project"
            mock_config.translation.target_language = "uk"
            mock_config.translation.cache_enabled = True
            mock_config.translation.batch_size = 5
            mock_config.cache_path = "test_cache"
            
            return Translator(api_key="test_api_key", project_id="test_project")
    
    def test_translator_initialization_with_api_key(self, mock_translate_client):
        """Test translator initialization with API key."""
        with patch('src.translation.translator.config') as mock_config:
            mock_config.translation.api_key = "test_key"
            mock_config.translation.target_language = "uk"
            mock_config.translation.cache_enabled = True
            mock_config.cache_path = "test_cache"
            
            translator = Translator(api_key="test_key")
            
            assert translator.api_key == "test_key"
            assert translator.target_language == "uk"
            assert translator.is_available()
    
    def test_translator_initialization_failure(self):
        """Test translator initialization when Google client fails."""
        with patch('src.translation.translator.translate.Client') as mock_client_class:
            mock_client_class.side_effect = Exception("Authentication failed")
            
            with patch('src.translation.translator.config') as mock_config:
                mock_config.translation.api_key = "invalid_key"
                mock_config.translation.target_language = "uk"
                mock_config.translation.cache_enabled = True
                mock_config.cache_path = "test_cache"
                
                translator = Translator()
                assert not translator.is_available()
    
    def test_get_supported_languages_success(self, translator, mock_translate_client):
        """Test getting supported languages from API."""
        languages = translator.get_supported_languages()
        
        assert len(languages) == 3
        assert languages[0]["code"] == "en"
        assert languages[0]["name"] == "English"
        assert languages[1]["code"] == "uk"
        assert languages[1]["name"] == "Ukrainian"
    
    def test_get_supported_languages_fallback(self, translator, mock_translate_client):
        """Test fallback to local languages when API fails."""
        mock_translate_client.get_languages.side_effect = Exception("API error")
        
        languages = translator.get_supported_languages()
        
        # Should fallback to SUPPORTED_LANGUAGES from config
        assert len(languages) > 0
        language_codes = [lang["code"] for lang in languages]
        assert "uk" in language_codes
        assert "en" in language_codes
    
    def test_set_target_language_valid(self, translator):
        """Test setting valid target language."""
        translator.set_target_language("en")
        assert translator.target_language == "en"
    
    def test_set_target_language_invalid(self, translator):
        """Test setting invalid target language raises error."""
        with pytest.raises(ValueError):
            translator.set_target_language("invalid_lang")
    
    def test_translate_text_success(self, translator, mock_translate_client):
        """Test successful text translation."""
        result = translator.translate_text("hello world", "uk")
        
        assert result.original_text == "hello world"
        assert result.translated_text == "привіт світ"
        assert result.source_language == "en"
        assert result.target_language == "uk"
        assert result.processing_time > 0
        assert not result.cached
        
        # Verify API was called
        mock_translate_client.translate.assert_called_once_with(
            "hello world",
            target_language="uk",
            source_language=None
        )
    
    def test_translate_text_empty_input(self, translator):
        """Test translation with empty or whitespace input."""
        result = translator.translate_text("", "uk")
        
        assert result.original_text == ""
        assert result.translated_text == ""
        assert result.source_language == "unknown"
        assert result.target_language == "uk"
    
    def test_translate_text_with_cache_hit(self, translator, mock_translate_client):
        """Test translation with cache hit."""
        # Ensure cache is enabled and working
        assert translator.cache_enabled
        assert translator.cache is not None
        
        # Clear cache to ensure clean state
        translator.clear_cache()
        
        # First translation (cache miss) - specify source language to ensure consistent cache key
        result1 = translator.translate_text("hello_cache_test", "uk", "en")
        assert not result1.cached
        
        # Second translation (cache hit) - use same parameters
        result2 = translator.translate_text("hello_cache_test", "uk", "en")
        assert result2.cached
        assert result2.translated_text == result1.translated_text
        
        # API should only be called once
        assert mock_translate_client.translate.call_count == 1
    
    def test_translate_text_api_unavailable(self, translator):
        """Test translation when API is unavailable."""
        translator.client = None  # Simulate unavailable client
        
        result = translator.translate_text("hello", "uk")
        
        # Should return original text as fallback
        assert result.original_text == "hello"
        assert result.translated_text == "hello"
        assert result.source_language == "unknown"
        assert result.target_language == "uk"
    
    def test_translate_text_api_error_with_retry(self, translator, mock_translate_client):
        """Test translation with API error and retry mechanism."""
        # First two calls fail, third succeeds
        mock_translate_client.translate.side_effect = [
            google_exceptions.GoogleAPIError("Rate limit exceeded"),
            google_exceptions.GoogleAPIError("Temporary error"),
            {
                'translatedText': 'привіт',
                'detectedSourceLanguage': 'en'
            }
        ]
        
        with patch('time.sleep'):  # Speed up test by mocking sleep
            result = translator.translate_text("hello", "uk")
        
        assert result.translated_text == "привіт"
        assert mock_translate_client.translate.call_count == 3
    
    def test_translate_text_api_error_max_retries(self, translator, mock_translate_client):
        """Test translation failure after max retries."""
        # All calls fail
        mock_translate_client.translate.side_effect = google_exceptions.GoogleAPIError("Persistent error")
        
        with patch('time.sleep'):  # Speed up test
            result = translator.translate_text("hello", "uk")
        
        # Should fallback to original text
        assert result.original_text == "hello"
        assert result.translated_text == "hello"
        assert mock_translate_client.translate.call_count == 3  # max_retries
    
    def test_translate_text_unexpected_error(self, translator, mock_translate_client):
        """Test handling of unexpected errors."""
        mock_translate_client.translate.side_effect = ValueError("Unexpected error")
        
        result = translator.translate_text("hello", "uk")
        
        # Should fallback to original text
        assert result.original_text == "hello"
        assert result.translated_text == "hello"
    
    def test_translate_batch_success(self, translator, mock_translate_client):
        """Test successful batch translation."""
        texts = ["hello", "world", "test"]
        
        # Mock different responses for each text
        mock_translate_client.translate.side_effect = [
            {'translatedText': 'привіт', 'detectedSourceLanguage': 'en'},
            {'translatedText': 'світ', 'detectedSourceLanguage': 'en'},
            {'translatedText': 'тест', 'detectedSourceLanguage': 'en'}
        ]
        
        results = translator.translate_batch(texts, "uk")
        
        assert len(results) == 3
        assert results[0].translated_text == "привіт"
        assert results[1].translated_text == "світ"
        assert results[2].translated_text == "тест"
        assert mock_translate_client.translate.call_count == 3
    
    def test_translate_batch_empty_input(self, translator):
        """Test batch translation with empty input."""
        results = translator.translate_batch([])
        assert results == []
    
    def test_translate_batch_with_batch_size_limit(self, translator, mock_translate_client):
        """Test batch translation respects batch size limit."""
        # Create more texts than batch size (5)
        texts = [f"text{i}" for i in range(8)]
        
        mock_translate_client.translate.return_value = {
            'translatedText': 'переклад',
            'detectedSourceLanguage': 'en'
        }
        
        results = translator.translate_batch(texts, "uk")
        
        assert len(results) == 8
        # Should process in batches of 5, so 8 individual calls
        assert mock_translate_client.translate.call_count == 8
    
    @pytest.mark.asyncio
    async def test_translate_text_async(self, translator, mock_translate_client):
        """Test asynchronous text translation."""
        result = await translator.translate_text_async("hello", "uk")
        
        assert result.original_text == "hello"
        assert result.translated_text == "привіт світ"
        assert result.target_language == "uk"
    
    @pytest.mark.asyncio
    async def test_translate_batch_async(self, translator, mock_translate_client):
        """Test asynchronous batch translation."""
        texts = ["hello", "world"]
        
        mock_translate_client.translate.return_value = {
            'translatedText': 'переклад',
            'detectedSourceLanguage': 'en'
        }
        
        results = await translator.translate_batch_async(texts, "uk")
        
        assert len(results) == 2
        assert all(r.translated_text == "переклад" for r in results)
    
    @pytest.mark.asyncio
    async def test_translate_batch_async_with_exceptions(self, translator, mock_translate_client):
        """Test async batch translation handles exceptions gracefully."""
        texts = ["hello", "world"]
        
        # First call succeeds, second fails
        mock_translate_client.translate.side_effect = [
            {'translatedText': 'привіт', 'detectedSourceLanguage': 'en'},
            Exception("Translation failed")
        ]
        
        results = await translator.translate_batch_async(texts, "uk")
        
        assert len(results) == 2
        assert results[0].translated_text == "привіт"
        assert results[1].translated_text == "world"  # Fallback to original
    
    def test_clear_cache(self, translator):
        """Test clearing translation cache."""
        # Add something to cache first
        translator.translate_text("test", "uk")
        
        # Clear cache
        translator.clear_cache()
        
        # Verify cache is cleared
        cache_stats = translator.get_cache_stats()
        assert cache_stats["entries"] == 0
    
    def test_get_cache_stats(self, translator):
        """Test getting cache statistics."""
        stats = translator.get_cache_stats()
        
        assert "enabled" in stats
        assert "entries" in stats
        assert "cache_file" in stats
        assert stats["enabled"] is True
    
    def test_get_cache_stats_disabled(self, mock_translate_client):
        """Test cache stats when cache is disabled."""
        with patch('src.translation.translator.config') as mock_config:
            mock_config.translation.cache_enabled = False
            mock_config.translation.api_key = "test_key"
            mock_config.translation.target_language = "uk"
            
            translator = Translator()
            stats = translator.get_cache_stats()
            
            assert stats["enabled"] is False


class TestTranslatorNetworkErrorHandling:
    """Test network and API error handling scenarios."""
    
    @pytest.fixture
    def translator_with_failing_client(self):
        """Create translator with client that simulates various failures."""
        with patch('src.translation.translator.translate.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            with patch('src.translation.translator.config') as mock_config:
                mock_config.translation.api_key = "test_key"
                mock_config.translation.target_language = "uk"
                mock_config.translation.cache_enabled = True
                mock_config.cache_path = "test_cache"
                
                translator = Translator()
                translator.client = mock_client
                return translator, mock_client
    
    def test_network_timeout_error(self, translator_with_failing_client):
        """Test handling of network timeout errors."""
        translator, mock_client = translator_with_failing_client
        
        mock_client.translate.side_effect = google_exceptions.GoogleAPIError("Request timeout")
        
        with patch('time.sleep'):
            result = translator.translate_text("hello", "uk")
        
        # Should fallback to original text after retries
        assert result.translated_text == "hello"
        assert mock_client.translate.call_count == 3  # max retries
    
    def test_authentication_error(self, translator_with_failing_client):
        """Test handling of authentication errors."""
        translator, mock_client = translator_with_failing_client
        
        mock_client.translate.side_effect = google_exceptions.Forbidden("Invalid API key")
        
        with patch('time.sleep'):
            result = translator.translate_text("hello", "uk")
        
        assert result.translated_text == "hello"
    
    def test_quota_exceeded_error(self, translator_with_failing_client):
        """Test handling of quota exceeded errors."""
        translator, mock_client = translator_with_failing_client
        
        mock_client.translate.side_effect = google_exceptions.TooManyRequests("Quota exceeded")
        
        with patch('time.sleep'):
            result = translator.translate_text("hello", "uk")
        
        assert result.translated_text == "hello"
    
    def test_service_unavailable_error(self, translator_with_failing_client):
        """Test handling of service unavailable errors."""
        translator, mock_client = translator_with_failing_client
        
        mock_client.translate.side_effect = google_exceptions.ServiceUnavailable("Service down")
        
        with patch('time.sleep'):
            result = translator.translate_text("hello", "uk")
        
        assert result.translated_text == "hello"
    
    def test_intermittent_network_failure(self, translator_with_failing_client):
        """Test recovery from intermittent network failures."""
        translator, mock_client = translator_with_failing_client
        
        # First call fails, second succeeds
        mock_client.translate.side_effect = [
            google_exceptions.GoogleAPIError("Network error"),
            {'translatedText': 'привіт', 'detectedSourceLanguage': 'en'}
        ]
        
        with patch('time.sleep'):
            result = translator.translate_text("hello", "uk")
        
        assert result.translated_text == "привіт"
        assert mock_client.translate.call_count == 2
    
    def test_exponential_backoff_timing(self, translator_with_failing_client):
        """Test exponential backoff timing in retry mechanism."""
        translator, mock_client = translator_with_failing_client
        
        mock_client.translate.side_effect = google_exceptions.GoogleAPIError("Persistent error")
        
        with patch('time.sleep') as mock_sleep:
            translator.translate_text("hello", "uk")
        
        # Verify exponential backoff: 1s, 2s, 4s (but capped at max_delay)
        expected_delays = [1.0, 2.0]  # Only first two retries have delays
        actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert actual_delays == expected_delays
    
    def test_max_delay_cap(self, translator_with_failing_client):
        """Test that retry delay is capped at max_delay."""
        translator, mock_client = translator_with_failing_client
        translator.max_delay = 5.0  # Set low max delay for testing
        
        mock_client.translate.side_effect = google_exceptions.GoogleAPIError("Error")
        
        with patch('time.sleep') as mock_sleep:
            translator.translate_text("hello", "uk")
        
        # All delays should be <= max_delay
        for call in mock_sleep.call_args_list:
            delay = call[0][0]
            assert delay <= translator.max_delay


class TestTranslatorIntegration:
    """Integration tests for Translator with various scenarios."""
    
    def test_full_translation_workflow_with_cache(self):
        """Test complete translation workflow with caching."""
        with patch('src.translation.translator.translate.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            mock_client.translate.return_value = {
                'translatedText': 'привіт світ',
                'detectedSourceLanguage': 'en'
            }
            
            with patch('src.translation.translator.config') as mock_config:
                mock_config.translation.api_key = "test_key"
                mock_config.translation.target_language = "uk"
                mock_config.translation.cache_enabled = True
                mock_config.cache_path = "test_cache"
                
                translator = Translator()
                
                # Clear cache to ensure clean state
                translator.clear_cache()
                
                # First translation - should hit API
                result1 = translator.translate_text("hello world integration", "uk", "en")
                assert result1.translated_text == "привіт світ"
                assert not result1.cached
                
                # Second translation - should hit cache
                result2 = translator.translate_text("hello world integration", "uk", "en")
                assert result2.translated_text == "привіт світ"
                assert result2.cached
                
                # API should only be called once
                assert mock_client.translate.call_count == 1
    
    def test_language_switching_scenario(self):
        """Test switching between different target languages."""
        with patch('src.translation.translator.translate.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock different responses for different languages
            def mock_translate(text, target_language, source_language=None):
                translations = {
                    'uk': 'привіт',
                    'ru': 'привет',
                    'de': 'hallo'
                }
                return {
                    'translatedText': translations.get(target_language, text),
                    'detectedSourceLanguage': 'en'
                }
            
            mock_client.translate.side_effect = mock_translate
            
            with patch('src.translation.translator.config') as mock_config:
                mock_config.translation.api_key = "test_key"
                mock_config.translation.target_language = "uk"
                mock_config.translation.cache_enabled = True
                mock_config.cache_path = "test_cache"
                
                translator = Translator()
                
                # Translate to Ukrainian
                result_uk = translator.translate_text("hello", "uk")
                assert result_uk.translated_text == "привіт"
                
                # Translate to Russian
                result_ru = translator.translate_text("hello", "ru")
                assert result_ru.translated_text == "привет"
                
                # Translate to German
                result_de = translator.translate_text("hello", "de")
                assert result_de.translated_text == "hallo"
    
    def test_concurrent_translation_safety(self):
        """Test that translator handles concurrent requests safely."""
        import threading
        
        with patch('src.translation.translator.translate.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            call_count = 0
            def mock_translate(text, target_language, source_language=None):
                nonlocal call_count
                call_count += 1
                time.sleep(0.01)  # Simulate API delay
                return {
                    'translatedText': f'translated_{call_count}',
                    'detectedSourceLanguage': 'en'
                }
            
            mock_client.translate.side_effect = mock_translate
            
            with patch('src.translation.translator.config') as mock_config:
                mock_config.translation.api_key = "test_key"
                mock_config.translation.target_language = "uk"
                mock_config.translation.cache_enabled = True
                mock_config.cache_path = "test_cache"
                
                translator = Translator()
                results = []
                
                def translate_worker(text):
                    result = translator.translate_text(f"text_{text}", "uk")
                    results.append(result)
                
                # Start multiple concurrent translations
                threads = []
                for i in range(5):
                    thread = threading.Thread(target=translate_worker, args=(i,))
                    threads.append(thread)
                    thread.start()
                
                # Wait for all to complete
                for thread in threads:
                    thread.join()
                
                # All translations should complete successfully
                assert len(results) == 5
                assert all(r.translated_text.startswith('translated_') for r in results)


if __name__ == "__main__":
    pytest.main([__file__])