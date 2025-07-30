"""
PDF conversion module for gemini-embedding-model

This module provides functionality to convert PDF files to Markdown format
using the MarkerPDF service, with support for batch processing and organized output.
"""

from .pdf_converter import PDFConverter
from .markerpdf_client import MarkerPDFClient
from .conversion_utils import (
    sanitize_filename,
    get_pdf_files,
    create_conversion_report,
    validate_pdf_file,
    ensure_output_directory
)

__version__ = "1.0.0"
__author__ = "Jason Lam"

__all__ = [
    'PDFConverter',
    'MarkerPDFClient',
    'sanitize_filename',
    'get_pdf_files',
    'create_conversion_report',
    'validate_pdf_file',
    'ensure_output_directory'
]

# Module level configuration
DEFAULT_CONFIG = {
    "batch_size": 5,
    "timeout": 300,
    "max_retries": 3,
}
