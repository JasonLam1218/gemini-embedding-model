import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from loguru import logger
from .supabase_client import SupabaseClient

@dataclass
class Document:
    title: str
    content: str
    source_file: str
    paper_set: str
    paper_number: str
    metadata: Optional[Dict] = None

    def to_dict(self):
        return asdict(self)

@dataclass
class TextChunk:
    document_id: int
    chunk_text: str
    chunk_index: int
    chunk_size: int
    overlap_size: int = 0
    metadata: Optional[Dict] = None

    def to_dict(self):
        return asdict(self)

@dataclass
class Embedding:
    chunk_id: int
    embedding: np.ndarray
    model_name: str = 'text-embedding-004'

    def to_dict(self):
        return {
            'chunk_id': self.chunk_id,
            'embedding': self.embedding.tolist() if hasattr(self.embedding, 'tolist') else self.embedding,
            'model_name': self.model_name
        }

class VectorStore:
    def __init__(self, table_name: str = "embeddings"):
        self.client = SupabaseClient()
        self.table = table_name
    
    # === DOCUMENT OPERATIONS ===
    
    def insert_document(self, document: Document) -> int:
        """Insert a document and return its ID"""
        try:
            data = {
                'title': document.title,
                'content': document.content,
                'source_file': document.source_file,
                'paper_set': document.paper_set,
                'paper_number': document.paper_number,
                'metadata': document.metadata or {}
            }
            
            response = self.client.client.table('documents').insert(data).execute()
            document_id = response.data[0]['id']
            
            logger.info(f"✅ Inserted document: {document.title} (ID: {document_id})")
            return document_id
            
        except Exception as e:
            logger.error(f"❌ Failed to insert document: {e}")
            raise
    
    def get_document(self, document_id: int) -> Optional[Dict]:
        """Retrieve a document by ID"""
        try:
            response = self.client.client.table('documents').select('*').eq('id', document_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    # === TEXT CHUNK OPERATIONS ===
    
    def insert_text_chunks(self, chunks: List[TextChunk]) -> List[int]:
        """Insert multiple text chunks and return their IDs"""
        try:
            data = []
            for chunk in chunks:
                data.append({
                    'document_id': chunk.document_id,
                    'chunk_text': chunk.chunk_text,
                    'chunk_index': chunk.chunk_index,
                    'chunk_size': chunk.chunk_size,
                    'overlap_size': chunk.overlap_size,
                    'metadata': chunk.metadata or {}
                })
            
            response = self.client.client.table('text_chunks').insert(data).execute()
            chunk_ids = [item['id'] for item in response.data]
            
            logger.info(f"✅ Inserted {len(chunk_ids)} text chunks")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"❌ Failed to insert text chunks: {e}")
            raise
    
    # === EMBEDDING OPERATIONS ===
    
    def insert_embeddings(self, embeddings: List[Embedding]) -> List[int]:
        """Insert multiple embeddings and return their IDs"""
        try:
            data = []
            for emb in embeddings:
                # Convert numpy array to list for JSON serialization
                embedding_list = emb.embedding.tolist() if hasattr(emb.embedding, 'tolist') else emb.embedding
                
                data.append({
                    'chunk_id': emb.chunk_id,
                    'embedding': embedding_list,
                    'model_name': emb.model_name
                })
            
            response = self.client.client.table('embeddings').insert(data).execute()
            embedding_ids = [item['id'] for item in response.data]
            
            logger.info(f"✅ Inserted {len(embedding_ids)} embeddings")
            return embedding_ids
            
        except Exception as e:
            logger.error(f"❌ Failed to insert embeddings: {e}")
            raise
    
    def similarity_search(self, query_embedding: np.ndarray, limit: int = 10, 
                         similarity_threshold: float = 0.8) -> List[Dict]:
        """Perform similarity search using cosine similarity"""
        try:
            # Convert numpy array to list
            query_vector = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
            
            # Use Supabase RPC function for similarity search
            response = self.client.client.rpc(
                'match_documents', 
                {
                    'query_embedding': query_vector,
                    'match_threshold': similarity_threshold,
                    'match_count': limit
                }
            ).execute()
            
            logger.info(f"✅ Found {len(response.data)} similar documents")
            return response.data
            
        except Exception as e:
            logger.error(f"❌ Similarity search failed: {e}")
            raise
    
    # === EXAM OPERATIONS ===
    
    def save_generated_exam(self, exam_data: Dict) -> int:
        """Save generated exam to database"""
        try:
            response = self.client.client.table('generated_exams').insert(exam_data).execute()
            exam_id = response.data[0]['id']
            
            logger.info(f"✅ Saved exam: {exam_data.get('title', 'Untitled')} (ID: {exam_id})")
            return exam_id
            
        except Exception as e:
            logger.error(f"❌ Failed to save exam: {e}")
            raise
    
    # === UTILITY METHODS ===
    
    def get_document_count(self) -> int:
        """Get total number of documents"""
        try:
            response = self.client.client.table('documents').select('id', count='exact').execute()
            return response.count or 0
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def get_embeddings_count(self) -> int:
        """Get total number of embeddings"""
        try:
            response = self.client.client.table('embeddings').select('id', count='exact').execute()
            return response.count or 0
        except Exception as e:
            logger.error(f"Failed to get embeddings count: {e}")
            return 0
    
    def clear_all_data(self):
        """WARNING: Delete all data (for testing only)"""
        try:
            self.client.client.table('embeddings').delete().neq('id', 0).execute()
            self.client.client.table('text_chunks').delete().neq('id', 0).execute()
            self.client.client.table('documents').delete().neq('id', 0).execute()
            self.client.client.table('generated_exams').delete().neq('id', 0).execute()
            logger.warning("⚠️ All data cleared from database")
        except Exception as e:
            logger.error(f"Failed to clear data: {e}")
    
    # === LEGACY METHODS (for compatibility with your original code) ===
    
    def add_embedding(self, embedding: List[float], metadata: Dict[str, Any]):
        """Legacy method - adds a single embedding"""
        record = {"embedding": embedding, **metadata}
        try:
            response = self.client.client.table(self.table).insert(record).execute()
            logger.info(f"✅ Added embedding via legacy method")
            return response.data[0]['id'] if response.data else None
        except Exception as e:
            logger.error(f"❌ Failed to add embedding: {e}")
            raise

    def query_by_metadata(self, match_dict: Dict[str, Any]):
        """Legacy method - queries vectors by metadata"""
        try:
            query = self.client.client.table(self.table).select('*')
            for key, value in match_dict.items():
                query = query.eq(key, value)
            response = query.execute()
            return response.data
        except Exception as e:
            logger.error(f"❌ Failed to query by metadata: {e}")
            raise
