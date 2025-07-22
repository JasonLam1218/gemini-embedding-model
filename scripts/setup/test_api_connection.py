import requests
import supabase
from config.settings import SUPABASE_URL, SUPABASE_SERVICE_KEY, GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL

# Test Supabase connection
client = supabase.create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
try:
    response = client.table("embeddings").select("*").limit(1).execute()
    print("Supabase connection successful.")
except Exception as e:
    print(f"Supabase connection failed: {e}")

# Test Gemini API connection
url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBEDDING_MODEL}:embedContent?key={GEMINI_API_KEY}"
headers = {"Content-Type": "application/json"}
data = {"requests": [{"content": "test", "model": GEMINI_EMBEDDING_MODEL}]}
try:
    response = requests.post(url, json=data, headers=headers, timeout=30)
    response.raise_for_status()
    print("Gemini API connection successful.")
except Exception as e:
    print(f"Gemini API connection failed: {e}")
