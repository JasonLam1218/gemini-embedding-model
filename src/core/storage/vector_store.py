from src.core.storage.supabase_client import SupabaseClient
from loguru import logger
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, table_name: str = "embeddings"):
        self.client = SupabaseClient()
        self.table = table_name

    def add_embedding(self, embedding: List[float], metadata: Dict[str, Any]):
        record = {"embedding": embedding, **metadata}
        self.client.upsert_vector(self.table, record)

    def query_by_metadata(self, match_dict: Dict[str, Any]):
        return self.client.query_vectors(self.table, match_dict)
