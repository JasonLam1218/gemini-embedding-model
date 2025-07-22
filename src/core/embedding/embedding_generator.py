import os
import requests
from typing import List
from loguru import logger
from src.core.embedding.gemini_client import GeminiClient
from src.core.embedding.rate_limiter import gemini_rate_limiter
from config.settings import BATCH_SIZE

class EmbeddingGenerator:
    def __init__(self):
        self.client = GeminiClient()
        self.batch_size = BATCH_SIZE

    @gemini_rate_limiter
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.client.embed_texts(texts)

    def process_chunks(self, chunks: List[str]) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i+self.batch_size]
            try:
                batch_embeddings = self._embed_batch(batch)
                embeddings.extend(batch_embeddings)
                logger.info(f"Embedded batch {i//self.batch_size + 1}")
            except Exception as e:
                logger.error(f"Failed to embed batch {i//self.batch_size + 1}: {e}")
        return embeddings 