import os
from pathlib import Path

# API Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
RATE_LIMIT_RPM = int(os.getenv('RATE_LIMIT_RPM', '15'))  # Requests per minute

# Processing Configuration
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10'))
MAX_CHUNK_SIZE = int(os.getenv('MAX_CHUNK_SIZE', '1500'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSIONS = 768
MIN_SIMILARITY_THRESHOLD = float(os.getenv('MIN_SIMILARITY_THRESHOLD', '0.3'))

# Exam Generation Configuration
DEFAULT_NUM_QUESTIONS = 4
DEFAULT_MARKS_PER_QUESTION = 25
QUESTION_TYPES = ["conceptual", "application", "analysis", "evaluation"]

# File Paths - UPDATED FOR MARKDOWN WORKFLOW
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# UPDATED: Point to converted markdown files instead of input
INPUT_DIR = DATA_DIR / "output" / "converted_markdown"  # Changed from input/kelvin_papers
MARKDOWN_INPUT_DIR = DATA_DIR / "output" / "converted_markdown"  # New setting
ORIGINAL_INPUT_DIR = DATA_DIR / "input"  # Keep for reference

OUTPUT_DIR = DATA_DIR / "output"
PROCESSED_DIR = OUTPUT_DIR / "processed"
EXAMS_DIR = OUTPUT_DIR / "generated_exams"

# Ensure directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
EXAMS_DIR.mkdir(parents=True, exist_ok=True)

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"

# Database Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY')
