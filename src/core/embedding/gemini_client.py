import google.generativeai as genai
import numpy as np
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
import os
import time

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=self.api_key)
        self.model_name = "models/text-embedding-004"
        self.max_content_length = 30000  # Add content length limit
        logger.info("✅ Gemini client initialized")

    def _validate_and_truncate_content(self, text: str) -> str:
        """Validate and truncate content if necessary"""
        if not text or not text.strip():
            raise ValueError("Empty or invalid text content")
        
        text = text.strip()
        
        # Truncate if too long, preserving complete sentences
        if len(text) > self.max_content_length:
            truncated = text[:self.max_content_length]
            # Find last complete sentence
            last_period = truncated.rfind('.')
            if last_period > self.max_content_length * 0.8:  # If we have at least 80% content
                truncated = truncated[:last_period + 1]
            
            logger.warning(f"⚠️ Content truncated from {len(text)} to {len(truncated)} characters")
            return truncated
        
        return text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text using the correct API format"""
        try:
            # Validate and clean content
            clean_text = self._validate_and_truncate_content(text)
            
            result = genai.embed_content(
                model=self.model_name,
                content=clean_text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            
            embedding = result['embedding']
            
            # Validate embedding result
            if not embedding or len(embedding) == 0:
                raise ValueError("Empty embedding returned from API")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with robust error handling"""
        if not texts:
            logger.warning("Empty text list provided")
            return []
        
        embeddings = []
        successful_count = 0
        failed_count = 0
        
        for i, text in enumerate(texts, 1):
            try:
                logger.info(f"🔄 Generating embedding {i}/{len(texts)}")
                
                # Add small delay between requests to avoid rate limiting
                if i > 1:
                    time.sleep(1)
                
                embedding = self.embed_text(text)
                
                # Validate embedding result
                if embedding and isinstance(embedding, list) and len(embedding) > 0:
                    embeddings.append(embedding)
                    successful_count += 1
                    logger.debug(f"✅ Successfully generated embedding {i} ({len(embedding)} dimensions)")
                else:
                    embeddings.append([])  # Empty list instead of None
                    failed_count += 1
                    logger.error(f"❌ Empty embedding returned for text {i}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ Failed to embed text {i}/{len(texts)}: {e}")
                embeddings.append([])  # Empty list instead of None
        
        logger.info(f"📊 Embedding Results: {successful_count} successful, {failed_count} failed out of {len(texts)} total")
        
        # Always return a list of the same length as input
        if len(embeddings) != len(texts):
            logger.warning(f"⚠️ Length mismatch: {len(embeddings)} results for {len(texts)} texts")
            # Pad or truncate to match
            while len(embeddings) < len(texts):
                embeddings.append([])
            embeddings = embeddings[:len(texts)]
        
        return embeddings
