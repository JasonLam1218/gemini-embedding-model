import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from .gemini_client import GeminiClient
from .rate_limiter import gemini_rate_limiter, batch_rate_limiter
from config.settings import BATCH_SIZE
import time
import json
from datetime import datetime

class EmbeddingGenerator:
    def __init__(self):
        """Initialize embedding generator with Gemini client"""
        self.client = GeminiClient()
        self.batch_size = 5  # Reduced from 10 for safer processing
        self.rate_limiter = gemini_rate_limiter
        logger.info("✅ Embedding generator initialized with enhanced rate limiting")

    def generate_single_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text with enhanced error handling"""
        try:
            if not text or not text.strip():
                logger.warning("⚠️ Empty text provided for embedding generation")
                return None

            embedding = self.client.embed_text(text.strip())
            if embedding and len(embedding) > 0:
                logger.debug(f"✅ Generated embedding with {len(embedding)} dimensions")
                return embedding
            else:
                logger.warning("⚠️ Empty embedding returned")
                return None

        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "429" in error_str:
                logger.error(f"❌ API quota exhausted: {e}")
                raise RuntimeError(f"API quota exhausted: {e}")
            elif "500" in error_str:
                logger.error(f"❌ Server error (500): {e}")
                raise RuntimeError(f"Server error: {e}")
            else:
                logger.error(f"❌ Failed to generate single embedding: {e}")
                return None

    def process_chunks_batch(self, chunks: List[str], batch_size: int = 5) -> List[Dict]:
        """Process chunks in batches with proper delays and error handling"""
        if not chunks:
            logger.warning("No chunks provided for embedding generation")
            return []

        logger.info(f"🧠 Processing {len(chunks)} chunks in batches of {batch_size}")
        results = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(chunks) - 1) // batch_size + 1
            
            logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            batch_results = []
            for j, chunk in enumerate(batch):
                try:
                    logger.info(f"  🧠 Generating embedding {j+1}/{len(batch)} in batch {batch_num}")
                    
                    embedding = self.generate_single_embedding(chunk)
                    if embedding:
                        batch_results.append({
                            'chunk_text': chunk,
                            'embedding': embedding,
                            'success': True,
                            'batch_number': batch_num,
                            'chunk_index_in_batch': j
                        })
                        logger.info(f"  ✅ Success: embedding {j+1}/{len(batch)}")
                    else:
                        logger.warning(f"  ⚠️ Failed: embedding {j+1}/{len(batch)}")
                        batch_results.append({
                            'chunk_text': chunk,
                            'embedding': [],
                            'success': False,
                            'error': 'Empty embedding returned'
                        })
                    
                    # Delay between requests within batch
                    if j < len(batch) - 1:
                        logger.debug("⏳ Inter-request delay: 3 seconds")
                        time.sleep(3)  # 3 second delay between requests
                        
                except Exception as e:
                    logger.error(f"  ❌ Failed batch item {j+1}: {e}")
                    batch_results.append({
                        'chunk_text': chunk,
                        'embedding': [],
                        'success': False,
                        'error': str(e) 
                    })
                    
                    # Check if we should stop due to quota/server errors
                    error_str = str(e).lower()
                    if "quota" in error_str or "429" in error_str or "500" in error_str:
                        logger.error(f"🚨 Stopping batch processing due to: {e}")
                        results.extend(batch_results)
                        return results
            
            results.extend(batch_results)
            
            # Longer delay between batches
            if i + batch_size < len(chunks):
                logger.info("⏳ Inter-batch delay: 8 seconds")
                time.sleep(8)  # 15 second delay between batches
                
        return results

    def check_quota_status(self) -> Dict[str, Any]:
        """Check current API quota status with enhanced error handling"""
        try:
            return {
                'daily_requests': getattr(self.client, 'daily_request_count', 0),
                'total_requests': getattr(self.client, 'request_count', 0),
                'requests_remaining': max(0, 50 - getattr(self.client, 'daily_request_count', 0)),  # Conservative estimate
                'daily_limit': 50,  # Gemini free tier limit
                'can_make_requests': True,
                'last_request_time': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Failed to check quota status: {e}")
            return {
                'daily_requests': 0,
                'total_requests': 0,
                'requests_remaining': 50,
                'daily_limit': 50,
                'can_make_requests': True,
                'last_request_time': datetime.now().isoformat()
            }
