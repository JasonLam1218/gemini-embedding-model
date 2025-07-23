import json
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger
from pathlib import Path

from ..embedding.embedding_generator import EmbeddingGenerator
from ..storage.vector_store import VectorStore

class RAGPipeline:
    def __init__(self, embeddings_path: str = None):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.embeddings_data = []
        
        # Load embeddings data
        if embeddings_path:
            self.load_embeddings(embeddings_path)
        else:
            default_path = "data/output/processed/embeddings.json"
            if Path(default_path).exists():
                self.load_embeddings(default_path)
        
        logger.info(f"✅ RAG Pipeline initialized with {len(self.embeddings_data)} embeddings")

    def load_embeddings(self, embeddings_path: str):
        """Load embeddings data from file"""
        try:
            self.embeddings_data = self.embedding_generator.load_embeddings_data(embeddings_path)
            logger.info(f"📥 Loaded {len(self.embeddings_data)} embeddings for RAG pipeline")
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {e}")
            self.embeddings_data = []

    def retrieve_relevant_content(self, query: str, top_k: int = 5,
                                min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        """Retrieve relevant content chunks based on query similarity"""
        try:
            if not self.embeddings_data:
                logger.warning("⚠️ No embeddings data available")
                return []

            # Generate query embedding
            query_embedding = self.embedding_generator.generate_topic_embedding(query)
            if not query_embedding:
                logger.error("❌ Failed to generate query embedding")
                return []

            # Find similar chunks
            similar_chunks = self.embedding_generator.find_similar_chunks(
                query_embedding, 
                self.embeddings_data, 
                top_k=top_k, 
                min_similarity=min_similarity
            )

            logger.info(f"🔍 Retrieved {len(similar_chunks)} relevant chunks for query: '{query}'")
            return similar_chunks

        except Exception as e:
            logger.error(f"❌ Failed to retrieve relevant content: {e}")
            return []

    def retrieve_by_topic_categories(self, main_topic: str, 
                                   sub_topics: List[str] = None,
                                   top_k_per_topic: int = 3) -> Dict[str, List[Dict]]:
        """Retrieve content organized by topic categories"""
        try:
            results = {}
            
            # Retrieve for main topic
            main_content = self.retrieve_relevant_content(
                main_topic, top_k=top_k_per_topic * 2
            )
            results[main_topic] = main_content

            # Retrieve for sub-topics if provided
            if sub_topics:
                for sub_topic in sub_topics:
                    combined_query = f"{main_topic} {sub_topic}"
                    sub_content = self.retrieve_relevant_content(
                        combined_query, top_k=top_k_per_topic
                    )
                    results[sub_topic] = sub_content

            logger.info(f"📚 Retrieved content for {len(results)} topic categories")
            return results

        except Exception as e:
            logger.error(f"❌ Failed to retrieve by topic categories: {e}")
            return {}

    def find_content_by_structure_similarity(self, question_structure: Dict) -> List[Dict]:
        """Find content that matches a specific question structure"""
        try:
            main_topic = question_structure.get("main_topic", "")
            sub_parts = question_structure.get("sub_parts", [])

            # Create comprehensive query from structure
            query_parts = [main_topic]
            for sub_part in sub_parts:
                if "focus" in sub_part:
                    query_parts.append(sub_part["focus"])
                elif "topic" in sub_part:
                    query_parts.append(sub_part["topic"])

            combined_query = " ".join(query_parts)
            
            # Retrieve relevant content
            relevant_content = self.retrieve_relevant_content(
                combined_query, 
                top_k=10, 
                min_similarity=0.2
            )

            logger.info(f"🎯 Found {len(relevant_content)} chunks matching structure: {main_topic}")
            return relevant_content

        except Exception as e:
            logger.error(f"❌ Failed to find content by structure similarity: {e}")
            return []

    def retrieve_relevant_chunks_by_similarity(self, query_embedding: np.ndarray,
                                             top_k: int = 5) -> List[Dict]:
        """Retrieve chunks using pre-computed embedding"""
        try:
            if not self.embeddings_data:
                return []

            # Convert numpy array to list if necessary
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            similar_chunks = self.embedding_generator.find_similar_chunks(
                query_embedding, 
                self.embeddings_data, 
                top_k=top_k
            )

            return similar_chunks

        except Exception as e:
            logger.error(f"❌ Failed to retrieve chunks by similarity: {e}")
            return []

    def generate_context_summary(self, relevant_chunks: List[Dict]) -> str:
        """Generate a summary of relevant chunks for context"""
        try:
            if not relevant_chunks:
                return "No relevant content found."

            # Combine top chunks into context
            context_parts = []
            for i, chunk in enumerate(relevant_chunks[:5]):
                chunk_text = chunk.get('chunk_text', '')
                similarity = chunk.get('similarity_score', 0)
                
                # Truncate long chunks
                if len(chunk_text) > 200:
                    chunk_text = chunk_text[:200] + "..."
                
                context_parts.append(f"[Relevance: {similarity:.3f}] {chunk_text}")

            context_summary = "\n\n".join(context_parts)
            logger.info(f"📝 Generated context summary from {len(relevant_chunks)} chunks")
            
            return context_summary

        except Exception as e:
            logger.error(f"❌ Failed to generate context summary: {e}")
            return "Error generating context summary."

    def get_diverse_content_sample(self, query: str, sample_size: int = 8) -> List[Dict]:
        """Get a diverse sample of content using clustering"""
        try:
            # First get a larger set of relevant content
            initial_content = self.retrieve_relevant_content(
                query, top_k=sample_size * 2, min_similarity=0.1
            )

            if len(initial_content) <= sample_size:
                return initial_content

            # Use embedding generator's clustering to get diverse samples
            clusters = self.embedding_generator.cluster_similar_chunks(
                initial_content, similarity_threshold=0.6
            )

            # Select representative content from each cluster
            diverse_sample = []
            for cluster in clusters:
                if not cluster:
                    continue
                    
                # Sort cluster by similarity score and take the best one
                cluster_sorted = sorted(
                    cluster, 
                    key=lambda x: x.get('similarity_score', 0), 
                    reverse=True
                )
                diverse_sample.append(cluster_sorted[0])
                
                if len(diverse_sample) >= sample_size:
                    break

            # Fill remaining spots with highest similarity content if needed
            while len(diverse_sample) < sample_size and len(diverse_sample) < len(initial_content):
                for item in initial_content:
                    if item not in diverse_sample:
                        diverse_sample.append(item)
                        break

            logger.info(f"🎲 Selected {len(diverse_sample)} diverse content samples")
            return diverse_sample[:sample_size]

        except Exception as e:
            logger.error(f"❌ Failed to get diverse content sample: {e}")
            return self.retrieve_relevant_content(query, top_k=sample_size)

    def analyze_content_coverage(self, query: str) -> Dict[str, Any]:
        """Analyze how well the available content covers a query topic"""
        try:
            relevant_content = self.retrieve_relevant_content(
                query, top_k=20, min_similarity=0.1
            )

            if not relevant_content:
                return {
                    "coverage_score": 0.0,
                    "total_chunks": 0,
                    "relevant_chunks": 0,
                    "avg_similarity": 0.0,
                    "coverage_analysis": "No relevant content found"
                }

            # Calculate coverage metrics
            similarities = [chunk.get('similarity_score', 0) for chunk in relevant_content]
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            
            # Count high-quality matches
            high_quality_count = sum(1 for sim in similarities if sim > 0.5)
            
            coverage_score = min(1.0, (avg_similarity + (high_quality_count / 10)) / 2)

            coverage_analysis = self._generate_coverage_analysis(
                coverage_score, len(relevant_content), avg_similarity
            )

            return {
                "coverage_score": coverage_score,
                "total_chunks": len(self.embeddings_data),
                "relevant_chunks": len(relevant_content),
                "high_quality_matches": high_quality_count,
                "avg_similarity": avg_similarity,
                "max_similarity": max(similarities) if similarities else 0,
                "coverage_analysis": coverage_analysis
            }

        except Exception as e:
            logger.error(f"❌ Failed to analyze content coverage: {e}")
            return {"coverage_score": 0.0, "error": str(e)}

    def _generate_coverage_analysis(self, coverage_score: float, 
                                  relevant_count: int, avg_similarity: float) -> str:
        """Generate textual analysis of content coverage"""
        if coverage_score >= 0.7:
            return f"Excellent coverage - {relevant_count} relevant chunks with high similarity (avg: {avg_similarity:.3f})"
        elif coverage_score >= 0.5:
            return f"Good coverage - {relevant_count} relevant chunks with moderate similarity (avg: {avg_similarity:.3f})"
        elif coverage_score >= 0.3:
            return f"Fair coverage - {relevant_count} relevant chunks with lower similarity (avg: {avg_similarity:.3f})"
        else:
            return f"Poor coverage - {relevant_count} relevant chunks with low similarity (avg: {avg_similarity:.3f})"

    def export_retrieval_results(self, query: str, output_path: str):
        """Export retrieval results for analysis"""
        try:
            results = {
                "query": query,
                "timestamp": logger.opt(record=True).info("Export timestamp"),
                "relevant_content": self.retrieve_relevant_content(query, top_k=10),
                "coverage_analysis": self.analyze_content_coverage(query),
                "diverse_sample": self.get_diverse_content_sample(query, sample_size=5)
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"📤 Exported retrieval results to: {output_path}")

        except Exception as e:
            logger.error(f"❌ Failed to export retrieval results: {e}")
