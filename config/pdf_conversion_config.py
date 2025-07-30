"""
Configuration settings for PDF to Markdown conversion using MarkerPDF
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base directories
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output" / "converted_markdown"
REPORTS_DIR = BASE_DIR / "data" / "output" / "conversion_reports" / "markerpdf"
LOGS_DIR = BASE_DIR / "data" / "output" / "logs" / "conversion"

# MarkerPDF API settings
MARKERPDF_CONFIG = {
    "api_key": os.getenv('MARKERPDF_API_KEY'),
    "base_url": "https://api.markerpdf.com/v1",
    "timeout": 300,
    "max_retries": 3,
    "retry_delay": 5,  # seconds
}

# Conversion settings
CONVERSION_CONFIG = {
    "batch_size": 5,
    "max_file_size_mb": 50,
    "supported_formats": ['.pdf'],
    "output_format": "markdown",
    "preserve_images": True,
    "extract_tables": True,
    "preserve_formatting": True,
}

# Output settings
OUTPUT_CONFIG = {
    "create_conversion_reports": True,
    "preserve_directory_structure": True,
    "sanitize_filenames": True,
    "create_index_file": True,
    "backup_original": False,
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_handler": True,
    "console_handler": True,
}

# File naming patterns
NAMING_PATTERNS = {
    "sanitize_chars": r'[^\w\-_.]',
    "replacement_char": '_',
    "max_filename_length": 100,
    "timestamp_format": "%Y%m%d_%H%M%S",
}

# Directory mappings for organized output
DIRECTORY_MAPPINGS = {
    "kelvin_papers": {
        "input": INPUT_DIR / "kelvin_papers",
        "output": OUTPUT_DIR / "kelvin_papers",
    },
    "lectures": {
        "input": INPUT_DIR / "lectures", 
        "output": OUTPUT_DIR / "lectures",
    }
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary"""
    return {
        "markerpdf": MARKERPDF_CONFIG,
        "conversion": CONVERSION_CONFIG,
        "output": OUTPUT_CONFIG,
        "logging": LOGGING_CONFIG,
        "naming": NAMING_PATTERNS,
        "directories": DIRECTORY_MAPPINGS,
        "paths": {
            "base": BASE_DIR,
            "input": INPUT_DIR,
            "output": OUTPUT_DIR,
            "reports": REPORTS_DIR,
            "logs": LOGS_DIR,
        }
    }
