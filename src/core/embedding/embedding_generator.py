import os
import requests
from typing import List, Tuple
from loguru import logger
from .gemini_client import GeminiClient
from .rate_limiter import gemini_rate_limiter
from config.settings import BATCH_SIZE
import time

class EmbeddingGenerator:
    def __init__(self):
        """Initialize embedding generator with Gemini client"""
        self.client = GeminiClient()
        self.batch_size = BATCH_SIZE
        logger.info("✅ Embedding generator initialized")

    def _validate_and_filter_texts(self, texts: List[str]) -> Tuple[List[str], List[int]]:
        """Validate and filter texts, returning valid texts and their original indices"""
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip() and len(text.strip()) > 10:  # Minimum content check
                valid_texts.append(text.strip())
                valid_indices.append(i)
            else:
                logger.warning(f"⚠️ Skipping invalid text at index {i}: too short or empty")
        
        return valid_texts, valid_indices

    @gemini_rate_limiter
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts with validation and error handling"""
        if not texts:
            return []
        
        # Filter out empty or invalid texts
        valid_texts, valid_indices = self._validate_and_filter_texts(texts)
        
        if not valid_texts:
            logger.error("❌ No valid texts in batch")
            return [None] * len(texts)  # Return None placeholders
        
        logger.info(f"📝 Processing {len(valid_texts)}/{len(texts)} valid texts in batch")
        
        try:
            embeddings = self.client.embed_texts(valid_texts)
            
            # Check if embeddings is None or invalid
            if embeddings is None:
                logger.error("❌ Received None from embed_texts")
                return [None] * len(texts)
            
            if not isinstance(embeddings, list):
                logger.error(f"❌ Expected list, got {type(embeddings)}")
                return [None] * len(texts)
            
            # Reconstruct full result list with None for invalid texts
            full_results = [None] * len(texts)
            
            # Safely map valid embeddings back to original positions
            for i, valid_idx in enumerate(valid_indices):
                if i < len(embeddings):
                    embedding = embeddings[i]
                    # Check if embedding is valid (not None and not empty)
                    if embedding is not None and len(embedding) > 0:
                        full_results[valid_idx] = embedding
                    else:
                        logger.warning(f"⚠️ Invalid embedding at index {i}")
            
            return full_results
            
        except Exception as e:
            logger.error(f"❌ Batch embedding failed: {e}")
            raise


    def process_chunks(self, chunks: List[str]) -> List[List[float]]:
        """Process text chunks and generate embeddings with comprehensive error handling"""
        if not chunks:
            logger.warning("No chunks provided for embedding generation")
            return []

        logger.info(f"🧠 Generating embeddings for {len(chunks)} chunks")
        
        all_embeddings = []
        successful_batches = 0
        failed_batches = 0
        
        for i in range(0, len(chunks), self.batch_size):
            batch_num = i // self.batch_size + 1
            batch = chunks[i:i + self.batch_size]
            
            try:
                logger.info(f"🔄 Processing batch {batch_num} ({len(batch)} chunks)")
                
                # Add small delay between batches to avoid rate limiting
                if i > 0:
                    time.sleep(2)
                
                batch_embeddings = self._embed_batch(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Count successful embeddings in this batch
                successful_in_batch = sum(1 for emb in batch_embeddings if emb is not None)
                logger.info(f"✅ Batch {batch_num} completed: {successful_in_batch}/{len(batch)} successful")
                
                successful_batches += 1
                
            except Exception as e:
                failed_batches += 1
                logger.error(f"❌ Failed to embed batch {batch_num}: {e}")
                
                # Add None placeholders for failed batch
                all_embeddings.extend([None] * len(batch))
                
                # Continue processing remaining batches
                continue

        # Final statistics
        total_successful = sum(1 for emb in all_embeddings if emb is not None)
        total_failed = len(all_embeddings) - total_successful
        
        logger.info(f"📊 Final Results:")
        logger.info(f"   ✅ Successful embeddings: {total_successful}")
        logger.info(f"   ❌ Failed embeddings: {total_failed}")
        logger.info(f"   ✅ Successful batches: {successful_batches}")
        logger.info(f"   ❌ Failed batches: {failed_batches}")
        
        if total_successful == 0:
            raise RuntimeError("All embedding generation failed")
        
        if total_failed > 0:
            logger.warning(f"⚠️ {total_failed} embeddings failed and will be excluded from results")
        
        # Filter out None values for return (only valid embeddings)
        valid_embeddings = [emb for emb in all_embeddings if emb is not None]
        
        logger.info(f"🎯 Returning {len(valid_embeddings)} valid embeddings")
        return valid_embeddings

