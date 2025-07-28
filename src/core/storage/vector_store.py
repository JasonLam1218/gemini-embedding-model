"""
Vector store implementation with Supabase integration for storing and retrieving
document chunks, embeddings, and generated exams.
Enhanced with duplicate checking functionality.
"""

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
    """Vector store for managing documents, chunks, embeddings, and similarity search"""
    
    def __init__(self):
        self.client = SupabaseClient()
        logger.info("✅ VectorStore initialized with Supabase connection")

    # === NEW: DUPLICATE CHECKING METHODS ===
    
    def document_exists_by_source_file(self, source_file: str) -> Optional[Dict]:
        """Check if document with same source_file exists"""
        try:
            response = self.client.client.table('documents')\
                .select('*')\
                .eq('source_file', source_file)\
                .execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Failed to check document existence: {e}")
            return None

    def get_chunks_by_source_file(self, source_file: str) -> List[Dict]:
        """Get all chunks for a document by source file"""
        doc = self.document_exists_by_source_file(source_file)
        if doc:
            return self.get_chunks_by_document(doc['id'])
        return []

    def embedding_exists_for_chunk(self, chunk_id: int) -> bool:
        """Check if embedding exists for a chunk"""
        try:
            response = self.client.client.table('embeddings')\
                .select('id')\
                .eq('chunk_id', chunk_id)\
                .execute()
            return len(response.data) > 0
        except Exception as e:
            logger.error(f"Failed to check embedding existence: {e}")
            return False

    def get_chunks_without_embeddings(self) -> List[Dict]:
        """Get all chunks that don't have embeddings yet"""
        try:
            # Query for chunks that don't have corresponding embeddings
            response = self.client.client.rpc('get_chunks_without_embeddings').execute()
            return response.data if response.data else []
        except Exception as e:
            # Fallback method using Python logic if RPC doesn't exist
            logger.warning(f"RPC method not available, using fallback: {e}")
            return self._get_chunks_without_embeddings_fallback()

    def _get_chunks_without_embeddings_fallback(self) -> List[Dict]:
        """Fallback method to get chunks without embeddings"""
        try:
            # Get all chunks
            all_chunks = self.client.client.table('text_chunks').select('*').execute().data
            
            # Get all chunk IDs that have embeddings
            embedded_chunks = self.client.client.table('embeddings').select('chunk_id').execute().data
            embedded_chunk_ids = {item['chunk_id'] for item in embedded_chunks}
            
            # Filter chunks that don't have embeddings
            chunks_without_embeddings = [
                chunk for chunk in all_chunks 
                if chunk['id'] not in embedded_chunk_ids
            ]
            
            return chunks_without_embeddings
        except Exception as e:
            logger.error(f"Fallback method failed: {e}")
            return []

    # === EXISTING DOCUMENT OPERATIONS ===

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

    def get_documents_by_set(self, paper_set: str) -> List[Dict]:
        """Get all documents from a specific paper set"""
        try:
            response = self.client.client.table('documents').select('*').eq('paper_set', paper_set).execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to get documents for set {paper_set}: {e}")
            return []

    def get_all_documents(self) -> List[Dict]:
        """Get all documents"""
        try:
            response = self.client.client.table('documents').select('*').execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            return []

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

    def get_chunks_by_document(self, document_id: int) -> List[Dict]:
        """Get all chunks for a specific document"""
        try:
            response = self.client.client.table('text_chunks').select('*').eq('document_id', document_id).order('chunk_index').execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to get chunks for document {document_id}: {e}")
            return []

    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict]:
        """Get a specific chunk by ID"""
        try:
            response = self.client.client.table('text_chunks').select('*').eq('id', chunk_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None

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

    def get_embedding_by_chunk_id(self, chunk_id: int) -> Optional[Dict]:
        """Get embedding for a specific chunk"""
        try:
            response = self.client.client.table('embeddings').select('*').eq('chunk_id', chunk_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Failed to get embedding for chunk {chunk_id}: {e}")
            return None

    def similarity_search(self, query_embedding: np.ndarray, limit: int = 10,
                         similarity_threshold: float = 0.3) -> List[Dict]:
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
            return []

    def similarity_search_with_metadata(self, query_embedding: np.ndarray,
                                      paper_set: Optional[str] = None,
                                      limit: int = 10,
                                      similarity_threshold: float = 0.3) -> List[Dict]:
        """Perform similarity search with optional filtering"""
        try:
            results = self.similarity_search(query_embedding, limit * 2, similarity_threshold)
            
            # Filter by paper_set if specified
            if paper_set:
                filtered_results = []
                for result in results:
                    document = self.get_document(result['document_id'])
                    if document and document.get('paper_set') == paper_set:
                        result['document_metadata'] = document
                        filtered_results.append(result)
                results = filtered_results[:limit]
            else:
                # Add document metadata to all results
                for result in results:
                    document = self.get_document(result['document_id'])
                    result['document_metadata'] = document

            return results[:limit]
        except Exception as e:
            logger.error(f"❌ Similarity search with metadata failed: {e}")
            return []

    # === EXAM OPERATIONS ===

    def save_generated_exam(self, exam_data: Dict) -> int:
        """Save generated exam to database with enhanced validation"""
        try:
            # Extract metadata with fallbacks for different exam formats
            exam_metadata = exam_data.get('exam_metadata', {})
            generation_stats = exam_data.get('generation_stats', {})
            
            # Ensure required fields exist with sensible defaults
            title = exam_metadata.get('title') or exam_data.get('title', 'Untitled Exam')
            topic = exam_metadata.get('topic') or exam_data.get('topic', '')
            difficulty = exam_metadata.get('difficulty', 'standard')
            total_marks = exam_metadata.get('total_marks', 0) or exam_data.get('total_marks', 100)
            total_questions = generation_stats.get('questions_generated', 0) or len(exam_data.get('questions', {}))
            
            # Validate essential data
            if total_questions == 0:
                logger.warning("⚠️ Exam has no questions, but saving anyway")
            
            data = {
                'title': title,
                'exam_json': exam_data,  # Complete exam structure as JSON
                'topic': topic,
                'difficulty': difficulty,
                'total_marks': int(total_marks),
                'total_questions': int(total_questions)
            }
            
            response = self.client.client.table('generated_exams').insert(data).execute()
            exam_id = response.data[0]['id']
            
            logger.info(f"✅ Saved exam: {title[:50]}... (ID: {exam_id})")
            logger.info(f"📊 Exam details: {total_questions} questions, {total_marks} marks, topic: {topic}")
            
            return exam_id
            
        except Exception as e:
            logger.error(f"❌ Failed to save exam: {e}")
            logger.error(f"📋 Exam data keys: {list(exam_data.keys()) if exam_data else 'No data'}")
            raise

    def verify_exam_saved(self, exam_id: int) -> bool:
        """Verify that an exam was successfully saved"""
        try:
            exam = self.get_exam_by_id(exam_id)
            if exam:
                logger.info(f"✅ Verified exam saved: ID {exam_id}, Title: {exam['title']}")
                return True
            else:
                logger.error(f"❌ Exam verification failed: ID {exam_id} not found")
                return False
        except Exception as e:
            logger.error(f"❌ Exam verification error: {e}")
            return False



    def get_generated_exams(self, limit: int = 20) -> List[Dict]:
        """Get recent generated exams"""
        try:
            response = self.client.client.table('generated_exams')\
                .select('*')\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to get generated exams: {e}")
            return []

    def get_exam_by_id(self, exam_id: int) -> Optional[Dict]:
        """Get a specific exam by ID"""
        try:
            response = self.client.client.table('generated_exams')\
                .select('*')\
                .eq('id', exam_id)\
                .execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Failed to get exam {exam_id}: {e}")
            return None

    def get_exams_by_topic(self, topic: str) -> List[Dict]:
        """Get exams by topic"""
        try:
            response = self.client.client.table('generated_exams')\
                .select('*')\
                .ilike('topic', f'%{topic}%')\
                .order('created_at', desc=True)\
                .execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to get exams for topic {topic}: {e}")
            return []

    def get_exams_count(self) -> int:
        """Get total number of generated exams"""
        try:
            response = self.client.client.table('generated_exams').select('id', count='exact').execute()
            return response.count or 0
        except Exception as e:
            logger.error(f"Failed to get exams count: {e}")
            return 0


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

    def get_chunks_count(self) -> int:
        """Get total number of text chunks"""
        try:
            response = self.client.client.table('text_chunks').select('id', count='exact').execute()
            return response.count or 0
        except Exception as e:
            logger.error(f"Failed to get chunks count: {e}")
            return 0

    def get_exams_count(self) -> int:
        """Get total number of generated exams"""
        try:
            response = self.client.client.table('generated_exams').select('id', count='exact').execute()
            return response.count or 0
        except Exception as e:
            logger.error(f"Failed to get exams count: {e}")
            return 0

    def get_database_stats(self) -> Dict[str, int]:
        """Get comprehensive database statistics"""
        return {
            'documents': self.get_document_count(),
            'text_chunks': self.get_chunks_count(),
            'embeddings': self.get_embeddings_count(),
            'generated_exams': self.get_exams_count()  # Includes exam count
        }


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

    # === BATCH OPERATIONS ===

    def batch_insert_document_with_chunks_and_embeddings(self,
                                                        document: Document,
                                                        chunks_text: List[str],
                                                        embeddings_data: List[np.ndarray]) -> Dict[str, Any]:
        """Insert document, chunks, and embeddings in a single transaction"""
        try:
            # Insert document
            doc_id = self.insert_document(document)

            # Create chunks
            chunks = []
            for i, chunk_text in enumerate(chunks_text):
                chunks.append(TextChunk(
                    document_id=doc_id,
                    chunk_text=chunk_text,
                    chunk_index=i,
                    chunk_size=len(chunk_text),
                    overlap_size=0
                ))

            # Insert chunks
            chunk_ids = self.insert_text_chunks(chunks)

            # Create embeddings
            embeddings = []
            for chunk_id, embedding in zip(chunk_ids, embeddings_data):
                embeddings.append(Embedding(
                    chunk_id=chunk_id,
                    embedding=embedding
                ))

            # Insert embeddings
            embedding_ids = self.insert_embeddings(embeddings)

            result = {
                'document_id': doc_id,
                'chunk_ids': chunk_ids,
                'embedding_ids': embedding_ids,
                'success': True
            }

            logger.info(f"✅ Batch insert completed: doc_id={doc_id}, chunks={len(chunk_ids)}, embeddings={len(embedding_ids)}")
            return result
        except Exception as e:
            logger.error(f"❌ Batch insert failed: {e}")
            raise

    # === ADVANCED SEARCH OPERATIONS ===

    def search_by_content(self, search_text: str, limit: int = 10) -> List[Dict]:
        """Search documents by content using text search"""
        try:
            response = self.client.client.table('documents')\
                .select('*')\
                .ilike('content', f'%{search_text}%')\
                .limit(limit)\
                .execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to search by content: {e}")
            return []

    def search_chunks_by_text(self, search_text: str, limit: int = 20) -> List[Dict]:
        """Search text chunks by content"""
        try:
            response = self.client.client.table('text_chunks')\
                .select('*')\
                .ilike('chunk_text', f'%{search_text}%')\
                .limit(limit)\
                .execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to search chunks: {e}")
            return []

    def get_recent_documents(self, limit: int = 10) -> List[Dict]:
        """Get most recently added documents"""
        try:
            response = self.client.client.table('documents')\
                .select('*')\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to get recent documents: {e}")
            return []

    # === VALIDATION AND HEALTH CHECK ===

    def validate_database_schema(self) -> Dict[str, bool]:
        """Validate that all required tables exist"""
        required_tables = ['documents', 'text_chunks', 'embeddings', 'generated_exams']
        results = {}
        
        for table in required_tables:
            try:
                response = self.client.client.table(table).select('*').limit(1).execute()
                results[table] = True
                logger.info(f"✅ Table '{table}' exists and is accessible")
            except Exception as e:
                results[table] = False
                logger.error(f"❌ Table '{table}' validation failed: {e}")
        
        return results

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the vector store"""
        try:
            stats = self.get_database_stats()
            schema_valid = self.validate_database_schema()
            
            return {
                'status': 'healthy' if all(schema_valid.values()) else 'unhealthy',
                'database_stats': stats,
                'schema_validation': schema_valid,
                'total_records': sum(stats.values()),
                'client_healthy': self.client.health_check() if hasattr(self.client, 'health_check') else True
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    # === MAINTENANCE OPERATIONS ===

    def optimize_database(self):
        """Perform database optimization tasks"""
        try:
            # Remove orphaned chunks (chunks without documents)
            orphaned_response = self.client.client.rpc('delete_orphaned_chunks').execute()
            
            # Remove orphaned embeddings (embeddings without chunks)
            orphaned_emb_response = self.client.client.rpc('delete_orphaned_embeddings').execute()
            
            logger.info("✅ Database optimization completed")
            return {
                'orphaned_chunks_removed': orphaned_response.data if orphaned_response.data else 0,
                'orphaned_embeddings_removed': orphaned_emb_response.data if orphaned_emb_response.data else 0
            }
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return {'error': str(e)}

    def backup_data(self, backup_path: str = None) -> Dict[str, Any]:
        """Create a backup of all data"""
        try:
            import json
            from datetime import datetime
            
            if not backup_path:
                backup_path = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # Get all data
            documents = self.get_all_documents()
            all_chunks = []
            all_embeddings = []
            all_exams = self.get_generated_exams(limit=1000)

            # Get chunks and embeddings for all documents
            for doc in documents:
                chunks = self.get_chunks_by_document(doc['id'])
                all_chunks.extend(chunks)
                
                for chunk in chunks:
                    embedding = self.get_embedding_by_chunk_id(chunk['id'])
                    if embedding:
                        all_embeddings.append(embedding)

            backup_data = {
                'backup_timestamp': datetime.now().isoformat(),
                'documents': documents,
                'text_chunks': all_chunks,
                'embeddings': all_embeddings,
                'generated_exams': all_exams,
                'stats': self.get_database_stats()
            }

            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"✅ Backup created: {backup_path}")
            return {
                'success': True,
                'backup_path': backup_path,
                'records_backed_up': sum(backup_data['stats'].values())
            }
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return {'success': False, 'error': str(e)}
