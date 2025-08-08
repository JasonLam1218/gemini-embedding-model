from pickle import NONE
import google.generativeai as genai
import numpy as np
from typing import List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger
import os
import time
import signal
from .rate_limiter import gemini_rate_limiter  # ADD THIS IMPORT

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client with enhanced capabilities"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=self.api_key)
        
        # Initialize models
        self.embedding_model = "models/text-embedding-004"
        self.generation_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Configuration
        self.max_content_length = NONE
        self.max_retries = 2  # Reduced from 3
        
        logger.info("✅ Enhanced Gemini client initialized with embedding and generation models")

    def _validate_and_truncate_content(self, text: str) -> str:
        """Validate and truncate content if necessary"""
        if not text or not text.strip():
            raise ValueError("Empty or invalid text content")
        
        text = text.strip()
        
        # Truncate if too long, preserving complete sentences
        # if len(text) > self.max_content_length:
        #     truncated = text[:self.max_content_length]
        #     # Find last complete sentence
        #     last_period = truncated.rfind('.')
        #     if last_period > self.max_content_length * 0.8:
        #         truncated = truncated[:last_period + 1]
        #     logger.warning(f"⚠️ Content truncated from {len(text)} to {len(truncated)} characters")
        #     return truncated
        
        return text

    @gemini_rate_limiter  # ADD THIS DECORATOR
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=5),
        retry=retry_if_exception_type((
            Exception,  # Catch all exceptions for retry
        ))
    )
    def embed_text(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """Generate embedding for a single text using Gemini API"""
        try:
            # Validate and clean content
            clean_text = self._validate_and_truncate_content(text)
            
            result = genai.embed_content(
                model=self.embedding_model,
                content=clean_text,
                task_type=task_type
            )
            
            embedding = result['embedding']
            
            # Validate embedding result
            if not embedding or len(embedding) == 0:
                raise ValueError("Empty embedding returned from API")
            
            logger.debug(f"✅ Generated embedding with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    @gemini_rate_limiter  # ADD THIS DECORATOR
    def generate_content(self, prompt: str, temperature: float = 0.7,
                        max_tokens: int = 1000, timeout: int = 60) -> str:
        """Generate content using Gemini generative model with timeout"""
        
        # Log prompt details BEFORE sending
        logger.info("🔍 GEMINI API REQUEST DETAILS:")
        logger.info(f"📏 Prompt length: {len(prompt)} characters")
        logger.info(f"📊 Estimated tokens: {len(prompt.split()) * 1.3:.0f}")
        logger.info(f"🌡️ Temperature: {temperature}")
        logger.info(f"🔢 Max tokens: {max_tokens}")
        
        # Log first and last parts of prompt for verification
        logger.info("📝 PROMPT PREVIEW (First 500 chars):")
        logger.info(f"'{prompt[:500]}...'")
        logger.info("📝 PROMPT PREVIEW (Last 500 chars):")
        logger.info(f"'...{prompt[-500:]}'")
        
        # Count content sections
        content_sections = prompt.count("===")
        logger.info(f"📁 Content sections detected: {content_sections}")
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Request timed out after {timeout} seconds")
        
        try:
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            # Make API call
            logger.info("🚀 Sending request to Gemini 2.5 Flash...")
            response = self.generation_model.generate_content(
                prompt,
                generation_config=self._get_generation_config(temperature, max_tokens)
            )
            
            signal.alarm(0)  # Cancel alarm
            
            # Log response details
            logger.info("✅ GEMINI API RESPONSE RECEIVED:")
            logger.info(f"📏 Response length: {len(response.text)} characters")
            
            return response.text.strip()
            
        except TimeoutError as e:
            logger.error(f"⏰ Request timed out: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Gemini API request failed: {e}")
            logger.error(f"💡 Failed prompt length was: {len(prompt)} characters")
            raise
        finally:
            signal.alarm(0)  # Ensure alarm is cancelled

    def _get_generation_config(self, temperature: float = 0.7,
                              max_tokens: int = 1000) -> genai.types.GenerationConfig:
        """Get generation configuration"""
        return genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=0.8,
            top_k=40
        )
