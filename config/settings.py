import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
GEMINI_GENERATION_MODEL = os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.0-flash-exp")

# Supabase Configuration  
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Processing Configuration
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", 1500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 10))
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", 60))
RATE_LIMIT_TPM = int(os.getenv("RATE_LIMIT_TPM", 1000000))

# Generation Settings
MAX_QUESTIONS_PER_BATCH = int(os.getenv("MAX_QUESTIONS_PER_BATCH", 5))
DEFAULT_DIFFICULTY = os.getenv("DEFAULT_DIFFICULTY", "intermediate")
CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", 3))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.8))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "data/output/logs/pipeline.log")
ENABLE_DEBUG_MODE = os.getenv("ENABLE_DEBUG_MODE", "false").lower() == "true"

# Database Configuration
VECTOR_DIMENSIONS = int(os.getenv("VECTOR_DIMENSIONS", 768))

# Validate required settings
required_vars = [
    'GEMINI_API_KEY',
    'SUPABASE_URL', 
    'SUPABASE_SERVICE_KEY'
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {missing_vars}")

# Validation checks
if GEMINI_API_KEY and not GEMINI_API_KEY.startswith('AIza'):
    raise ValueError("GEMINI_API_KEY appears to be invalid (should start with 'AIza')")

if SUPABASE_URL and not SUPABASE_URL.startswith('https://'):
    raise ValueError("SUPABASE_URL should start with 'https://'")
