import supabase
from config.settings import SUPABASE_URL, SUPABASE_SERVICE_KEY
from loguru import logger

class SupabaseClient:
    def __init__(self):
        self.client = supabase.create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    def upsert_vector(self, table: str, record: dict):
        try:
            response = self.client.table(table).upsert(record).execute()
            logger.info(f"Upserted record into {table}: {response}")
            return response
        except Exception as e:
            logger.error(f"Supabase upsert error: {e}")
            raise

    def query_vectors(self, table: str, match_dict: dict):
        try:
            response = self.client.table(table).select("*").match(match_dict).execute()
            logger.info(f"Queried {table} with {match_dict}: {response}")
            return response.data
        except Exception as e:
            logger.error(f"Supabase query error: {e}")
            raise
