import os
from typing import Optional, List, Dict, Any
from supabase import create_client, Client
from loguru import logger

class SupabaseClient:
    def __init__(self):
        """Initialize Supabase client with configuration from environment"""
        self.url = os.getenv('SUPABASE_URL')
        self.key = os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_ANON_KEY')
        
        if not self.url or not self.key:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in environment")
        
        self.client: Client = create_client(self.url, self.key)
        logger.info("✅ Supabase client initialized")
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            response = self.client.table('documents').select('id').limit(1).execute()
            logger.info("✅ Supabase connection successful")
            return True
        except Exception as e:
            logger.error(f"❌ Supabase connection failed: {e}")
            return False
    
    def execute_query(self, query: str) -> Any:
        """Execute raw SQL query"""
        try:
            response = self.client.postgrest.rpc('sql', {'query': query}).execute()
            return response
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    # Legacy methods for compatibility
    def upsert_vector(self, table_name: str, record: Dict[str, Any]):
        """Legacy method - upsert a vector record"""
        try:
            response = self.client.table(table_name).upsert(record).execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to upsert vector: {e}")
            raise
    
    def query_vectors(self, table_name: str, match_dict: Dict[str, Any]):
        """Legacy method - query vectors"""
        try:
            query = self.client.table(table_name).select('*')
            for key, value in match_dict.items():
                query = query.eq(key, value)
            response = query.execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to query vectors: {e}")
            raise
