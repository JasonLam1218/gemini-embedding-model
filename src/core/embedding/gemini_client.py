import google.generativeai as genai
import numpy as np
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
import os

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=self.api_key)
        self.model_name = "models/text-embedding-004"
        logger.info("✅ Gemini client initialized")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text using the correct API format"""
        try:
            # Use the SAME format that works in your test_api_connection.py
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            
            return result['embedding']
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        
        for i, text in enumerate(texts, 1):
            try:
                logger.info(f"🔄 Generating embedding {i}/{len(texts)}")
                embedding = self.embed_text(text)
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Failed to embed text {i}: {e}")
                # Continue processing other texts instead of failing completely
                embeddings.append([])
        
        return embeddings
