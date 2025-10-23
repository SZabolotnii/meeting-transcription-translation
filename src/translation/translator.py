"""
Translation module using Google Cloud Translation API.
Provides real-time text translation with caching and error handling.
"""

import time
import hashlib
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from google.cloud import translate_v2 as translate
from google.api_core import exceptions as google_exceptions
import logging

from src.config import config, SUPPORTED_LANGUAGES


@dataclass
class TranslationResult:
    """Result of a translation operation"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: Optional[float] = None
    timestamp: float = 0.0
    cached: bool = False
    processing_time: float = 0.0


@dataclass
class TranslationError:
    """Translation error information"""
    original_text: str
    error_message: str
    error_type: str
    timestamp: float
    retry_count: int = 0


class TranslationCache:
    """Local cache for translation results"""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir or config.cache_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "translation_cache.json"
        self.cache: Dict[str, Dict] = {}
        self.load_cache()
        
    def _generate_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key for text and language pair"""
        content = f"{text}:{source_lang}:{target_lang}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, source_lang: str, target_lang: str) -> Optional[TranslationResult]:
        """Get cached translation result"""
        key = self._generate_key(text, source_lang, target_lang)
        if key in self.cache:
            cached_data = self.cache[key]
            return TranslationResult(
                original_text=text,
                translated_text=cached_data["translated_text"],
                source_language=source_lang,
                target_language=target_lang,
                confidence=cached_data.get("confidence"),
                timestamp=time.time(),
                cached=True,
                processing_time=0.0
            )
        return None
    
    def set(self, result: TranslationResult) -> None:
        """Cache translation result"""
        key = self._generate_key(
            result.original_text, 
            result.source_language, 
            result.target_language
        )
        self.cache[key] = {
            "translated_text": result.translated_text,
            "confidence": result.confidence,
            "timestamp": result.timestamp
        }
        self.save_cache()
    
    def load_cache(self) -> None:
        """Load cache from file"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load translation cache: {e}")
            self.cache = {}
    
    def save_cache(self) -> None:
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Failed to save translation cache: {e}")
    
    def clear(self) -> None:
        """Clear all cached translations"""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()


class Translator:
    """
    Google Cloud Translation API client with caching and error handling.
    Provides real-time text translation for meeting transcription system.
    """
    
    def __init__(self, api_key: str = None, project_id: str = None):
        """
        Initialize translator with Google Cloud credentials.
        
        Args:
            api_key: Google Cloud Translation API key
            project_id: Google Cloud project ID
        """
        self.api_key = api_key or config.translation.api_key
        self.project_id = project_id or config.translation.project_id
        self.target_language = config.translation.target_language
        
        # Initialize Google Translate client
        self.client = None
        self._initialize_client()
        
        # Initialize cache
        self.cache_enabled = config.translation.cache_enabled
        self.cache = TranslationCache() if self.cache_enabled else None
        
        # Batch processing
        self.batch_size = config.translation.batch_size
        self.pending_translations = []
        
        # Error handling
        self.max_retries = 3
        self.base_delay = 1.0  # seconds
        self.max_delay = 60.0  # seconds
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logging.info(f"Translator initialized with target language: {self.target_language}")
    
    def _initialize_client(self) -> None:
        """Initialize Google Cloud Translation client"""
        try:
            if self.api_key:
                # Use API key authentication
                # Note: google-cloud-translate v2 doesn't support api_key parameter directly
                # Set as environment variable instead
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
                os.environ['GOOGLE_API_KEY'] = self.api_key
                self.client = translate.Client()
            elif self.project_id:
                # Use project-based authentication
                self.client = translate.Client()
            else:
                # Use default credentials
                self.client = translate.Client()
                
            # Test the connection
            self.client.get_languages()
            logging.info("Google Cloud Translation client initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize Google Cloud Translation client: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if translation service is available"""
        return self.client is not None
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages"""
        if not self.is_available():
            return [{"code": code, "name": name} for code, name in SUPPORTED_LANGUAGES.items()]
        
        try:
            languages = self.client.get_languages()
            return [{"code": lang["language"], "name": lang.get("name", lang["language"])} 
                   for lang in languages]
        except Exception as e:
            logging.error(f"Failed to get supported languages: {e}")
            return [{"code": code, "name": name} for code, name in SUPPORTED_LANGUAGES.items()]
    
    def set_target_language(self, language: str) -> None:
        """Set target language for translations"""
        if language in SUPPORTED_LANGUAGES:
            self.target_language = language
            logging.info(f"Target language set to: {language}")
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    def translate_text(self, text: str, target_language: str = None, source_language: str = None) -> TranslationResult:
        """
        Translate text to target language.
        
        Args:
            text: Text to translate
            target_language: Target language code (defaults to configured target)
            source_language: Source language code (auto-detect if None)
            
        Returns:
            TranslationResult with translation details
        """
        if not text or not text.strip():
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language or "unknown",
                target_language=target_language or self.target_language,
                timestamp=time.time()
            )
        
        target_lang = target_language or self.target_language
        start_time = time.time()
        
        # Check cache first
        if self.cache_enabled and self.cache:
            cached_result = self.cache.get(text, source_language or "auto", target_lang)
            if cached_result:
                return cached_result
        
        # Fallback if translation service is not available
        if not self.is_available():
            logging.warning("Translation service not available, returning original text")
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language or "unknown",
                target_language=target_lang,
                timestamp=time.time(),
                processing_time=time.time() - start_time
            )
        
        # Perform translation with retry logic
        for attempt in range(self.max_retries):
            try:
                result = self.client.translate(
                    text,
                    target_language=target_lang,
                    source_language=source_language
                )
                
                translation_result = TranslationResult(
                    original_text=text,
                    translated_text=result['translatedText'],
                    source_language=result.get('detectedSourceLanguage', source_language or 'unknown'),
                    target_language=target_lang,
                    timestamp=time.time(),
                    processing_time=time.time() - start_time
                )
                
                # Cache the result
                if self.cache_enabled and self.cache:
                    self.cache.set(translation_result)
                
                return translation_result
                
            except google_exceptions.GoogleAPIError as e:
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                logging.warning(f"Translation API error (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                else:
                    # Final fallback - return original text
                    logging.error(f"Translation failed after {self.max_retries} attempts: {e}")
                    return TranslationResult(
                        original_text=text,
                        translated_text=text,
                        source_language=source_language or "unknown",
                        target_language=target_lang,
                        timestamp=time.time(),
                        processing_time=time.time() - start_time
                    )
            
            except Exception as e:
                logging.error(f"Unexpected translation error: {e}")
                return TranslationResult(
                    original_text=text,
                    translated_text=text,
                    source_language=source_language or "unknown",
                    target_language=target_lang,
                    timestamp=time.time(),
                    processing_time=time.time() - start_time
                )
    
    async def translate_text_async(self, text: str, target_language: str = None, source_language: str = None) -> TranslationResult:
        """
        Asynchronous text translation.
        
        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Source language code
            
        Returns:
            TranslationResult with translation details
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.translate_text, 
            text, 
            target_language, 
            source_language
        )
    
    def translate_batch(self, texts: List[str], target_language: str = None, source_language: str = None) -> List[TranslationResult]:
        """
        Translate multiple texts in batch for efficiency.
        
        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language code
            
        Returns:
            List of TranslationResult objects
        """
        if not texts:
            return []
        
        results = []
        target_lang = target_language or self.target_language
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = []
            
            for text in batch:
                result = self.translate_text(text, target_lang, source_language)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    async def translate_batch_async(self, texts: List[str], target_language: str = None, source_language: str = None) -> List[TranslationResult]:
        """
        Asynchronous batch translation.
        
        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language code
            
        Returns:
            List of TranslationResult objects
        """
        if not texts:
            return []
        
        target_lang = target_language or self.target_language
        
        # Create translation tasks
        tasks = [
            self.translate_text_async(text, target_lang, source_language)
            for text in texts
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Batch translation error for text {i}: {result}")
                final_results.append(TranslationResult(
                    original_text=texts[i],
                    translated_text=texts[i],
                    source_language=source_language or "unknown",
                    target_language=target_lang,
                    timestamp=time.time()
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    def clear_cache(self) -> None:
        """Clear translation cache"""
        if self.cache:
            self.cache.clear()
            logging.info("Translation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.cache:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "entries": len(self.cache.cache),
            "cache_file": str(self.cache.cache_file)
        }
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)