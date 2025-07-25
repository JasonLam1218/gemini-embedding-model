import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from .gemini_client import GeminiClient
from .rate_limiter import gemini_rate_limiter
from config.settings import BATCH_SIZE
import time
import json
from datetime import datetime


class EmbeddingGenerator:
    def __init__(self):
        """Initialize embedding generator with Gemini client"""
        self.client = GeminiClient()
        self.batch_size = BATCH_SIZE
        logger.info("✅ Embedding generator initialized with similarity support")

    def generate_single_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text"""
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
            logger.error(f"❌ Failed to generate single embedding: {e}")
            return None

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate similarity: {e}")
            return 0.0

    def find_similar_chunks(self, query_embedding: List[float], 
                          embeddings_data: List[Dict], 
                          top_k: int = 5,
                          min_similarity: float = 0.3) -> List[Dict]:
        """Find most similar chunks based on embedding similarity"""
        try:
            similarities = []
            
            for i, chunk_data in enumerate(embeddings_data):
                chunk_embedding = chunk_data.get('embedding', [])
                if not chunk_embedding:
                    continue
                    
                similarity = self.calculate_similarity(query_embedding, chunk_embedding)
                
                if similarity >= min_similarity:
                    similarities.append({
                        'index': i,
                        'similarity': similarity,
                        'chunk_data': chunk_data
                    })
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Return top_k results
            result = []
            for item in similarities[:top_k]:
                result_item = item['chunk_data'].copy()
                result_item['similarity_score'] = item['similarity']
                result.append(result_item)
            
            logger.info(f"🔍 Found {len(result)} similar chunks with min similarity {min_similarity}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to find similar chunks: {e}")
            return []

    def generate_topic_embedding(self, topic: str) -> Optional[List[float]]:
        """Generate embedding for a topic/query"""
        try:
            # Create a more detailed query for better matching
            enhanced_query = f"Educational content about {topic}. Key concepts, principles, and applications in {topic}."
            return self.generate_single_embedding(enhanced_query)
        except Exception as e:
            logger.error(f"❌ Failed to generate topic embedding: {e}")
            return None

    def cluster_similar_chunks(self, embeddings_data: List[Dict], 
                              similarity_threshold: float = 0.7) -> List[List[Dict]]:
        """Cluster chunks based on similarity for question grouping"""
        try:
            clusters = []
            used_indices = set()
            
            for i, chunk_data in enumerate(embeddings_data):
                if i in used_indices:
                    continue
                    
                cluster = [chunk_data]
                used_indices.add(i)
                base_embedding = chunk_data.get('embedding', [])
                
                if not base_embedding:
                    continue
                
                # Find similar chunks for this cluster
                for j, other_chunk in enumerate(embeddings_data):
                    if j in used_indices or j <= i:
                        continue
                        
                    other_embedding = other_chunk.get('embedding', [])
                    if not other_embedding:
                        continue
                        
                    similarity = self.calculate_similarity(base_embedding, other_embedding)
                    
                    if similarity >= similarity_threshold:
                        cluster.append(other_chunk)
                        used_indices.add(j)
                
                if len(cluster) > 1:  # Only keep clusters with multiple items
                    clusters.append(cluster)
            
            logger.info(f"🔗 Created {len(clusters)} content clusters")
            return clusters
            
        except Exception as e:
            logger.error(f"❌ Failed to cluster chunks: {e}")
            return []

    def process_chunks_with_embeddings(self, chunks: List[str]) -> List[Dict]:
        """Process text chunks and generate embeddings with comprehensive error handling"""
        if not chunks:
            logger.warning("No chunks provided for embedding generation")
            return []

        logger.info(f"🧠 Generating embeddings for {len(chunks)} chunks")

        results = []
        successful_count = 0
        failed_count = 0

        for i, chunk in enumerate(chunks):
            try:
                logger.info(f"🔄 Processing chunk {i+1}/{len(chunks)}")
                
                # Add small delay to avoid rate limiting
                if i > 0:
                    time.sleep(1)

                embedding = self.generate_single_embedding(chunk)
                
                if embedding:
                    chunk_data = {
                        'chunk_text': chunk,
                        'chunk_index': i,
                        'chunk_size': len(chunk),
                        'embedding': embedding,
                        'embedding_model': 'text-embedding-004'
                    }
                    results.append(chunk_data)
                    successful_count += 1
                    logger.debug(f"✅ Successfully processed chunk {i+1}")
                else:
                    failed_count += 1
                    logger.warning(f"⚠️ Failed to generate embedding for chunk {i+1}")

            except Exception as e:
                failed_count += 1
                logger.error(f"❌ Failed to process chunk {i+1}: {e}")
                continue

        logger.info(f"📊 Processing Results: {successful_count} successful, {failed_count} failed")
        return results

    def save_embeddings_with_metadata(self, embeddings_data: List[Dict], output_path: str):
        """Save embeddings with comprehensive metadata"""
        try:
            # Add generation metadata
            output_data = {
                'metadata': {
                    'total_embeddings': len(embeddings_data),
                    'embedding_model': 'text-embedding-004',
                    'generation_timestamp': time.time(),
                    'embedding_dimensions': len(embeddings_data[0]['embedding']) if embeddings_data else 0
                },
                'embeddings': embeddings_data
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"✅ Saved {len(embeddings_data)} embeddings to {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save embeddings: {e}")
            raise

    def load_embeddings_data(self, embeddings_path: str) -> List[Dict]:
        """Load embeddings data from file"""
        try:
            with open(embeddings_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both old and new format
            if isinstance(data, dict) and 'embeddings' in data:
                embeddings_data = data['embeddings']
            else:
                embeddings_data = data
                
            logger.info(f"📥 Loaded {len(embeddings_data)} embeddings from {embeddings_path}")
            return embeddings_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {e}")
            return []
    
    def check_quota_status(self) -> Dict[str, Any]:
        """Check current API quota status"""
        try:
            return {
                'daily_requests': getattr(self.client, 'daily_request_count', 0),
                'total_requests': getattr(self.client, 'request_count', 0),
                'requests_remaining': max(0, 1500 - getattr(self.client, 'daily_request_count', 0)),
                'daily_limit': 1500,
                'can_make_requests': getattr(self.rate_limiter, 'can_make_request', lambda: True)(),
                'last_request_time': getattr(self.client, 'last_request_time', datetime.now()).isoformat() if hasattr(self.client, 'last_request_time') else datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Failed to check quota status: {e}")
            return {
                'daily_requests': 0,
                'total_requests': 0,
                'requests_remaining': 1500,
                'daily_limit': 1500,
                'can_make_requests': True,
                'last_request_time': datetime.now().isoformat()
            }
