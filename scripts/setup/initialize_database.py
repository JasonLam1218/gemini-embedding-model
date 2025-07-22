import supabase
from config.settings import SUPABASE_URL, SUPABASE_SERVICE_KEY

client = supabase.create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def initialize_embeddings_table():
    # This is a placeholder; actual table creation may require SQL or Supabase dashboard
    print("Ensure your Supabase project has an 'embeddings' table with columns: embedding (vector), metadata (json), chunk (text), etc.")

if __name__ == "__main__":
    initialize_embeddings_table()
