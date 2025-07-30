"""
Utility functions for PDF conversion operations
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """
    Sanitize filename for markdown output
    
    Args:
        filename: Original filename
        max_length: Maximum length for the filename
        
    Returns:
        Sanitized filename with .md extension
    """
    # Remove file extension and get base name
    name = Path(filename).stem
    
    # Replace spaces and special characters with underscores
    sanitized = re.sub(r'[^\w\-_.]', '_', name)
    
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Truncate if too long
    if len(sanitized) > max_length - 3:  # -3 for .md extension
        sanitized = sanitized[:max_length - 3]
    
    return f"{sanitized}.md"


def get_pdf_files(directory: str, recursive: bool = True) -> List[str]:
    """
    Get all PDF files from directory
    
    Args:
        directory: Directory path to search
        recursive: Whether to search subdirectories
        
    Returns:
        List of PDF file paths
    """
    pdf_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return pdf_files
    
    try:
        if recursive:
            pattern = "**/*.pdf"
        else:
            pattern = "*.pdf"
            
        pdf_files = [str(file) for file in directory_path.glob(pattern)]
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
    except Exception as e:
        logger.error(f"Error searching for PDF files in {directory}: {e}")
        
    return sorted(pdf_files)


def validate_pdf_file(file_path: str, max_size_mb: int = 50) -> Tuple[bool, str]:
    """
    Validate PDF file for conversion
    
    Args:
        file_path: Path to PDF file
        max_size_mb: Maximum file size in MB
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"
    
    # Check if it's a file
    if not file_path.is_file():
        return False, f"Path is not a file: {file_path}"
    
    # Check file extension
    if file_path.suffix.lower() != '.pdf':
        return False, f"File is not a PDF: {file_path}"
    
    # Check file size
    try:
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File too large ({file_size_mb:.1f}MB > {max_size_mb}MB): {file_path}"
    except Exception as e:
        return False, f"Error checking file size: {e}"
    
    # Check if file is readable
    try:
        with open(file_path, 'rb') as f:
            # Read first few bytes to check if it's a valid PDF
            header = f.read(8)
            if not header.startswith(b'%PDF-'):
                return False, f"Invalid PDF header: {file_path}"
    except Exception as e:
        return False, f"Error reading file: {e}"
    
    return True, ""


def ensure_output_directory(output_path: str) -> bool:
    """
    Ensure output directory exists
    
    Args:
        output_path: Output directory path
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Error creating output directory: {e}")
        return False


def create_conversion_report(
    conversion_results: List[Dict[str, Any]], 
    output_dir: str,
    session_id: str = None
) -> str:
    """
    Create a detailed conversion report
    
    Args:
        conversion_results: List of conversion result dictionaries
        output_dir: Output directory for the report
        session_id: Optional session ID for the report
        
    Returns:
        Path to the created report file
    """
    if session_id is None:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report_path = Path(output_dir) / f"conversion_report_{session_id}.json"
    
    # Ensure output directory exists
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create summary statistics
    total_files = len(conversion_results)
    successful = len([r for r in conversion_results if r.get('success', False)])
    failed = total_files - successful
    
    report = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_files": total_files,
            "successful_conversions": successful,
            "failed_conversions": failed,
            "success_rate": (successful / total_files * 100) if total_files > 0 else 0
        },
        "results": conversion_results
    }
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversion report created: {report_path}")
        return str(report_path)
        
    except Exception as e:
        logger.error(f"Error creating conversion report: {e}")
        return ""


def create_markdown_index(
    markdown_files: List[str], 
    output_dir: str,
    title: str = "Converted Documents"
) -> str:
    """
    Create an index markdown file listing all converted documents
    
    Args:
        markdown_files: List of markdown file paths
        output_dir: Output directory for the index
        title: Title for the index page
        
    Returns:
        Path to the created index file
    """
    index_path = Path(output_dir) / "README.md"
    
    try:
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total documents: {len(markdown_files)}\n\n")
            f.write("## Documents\n\n")
            
            for md_file in sorted(markdown_files):
                file_path = Path(md_file)
                relative_path = file_path.relative_to(output_dir)
                file_name = file_path.stem.replace('_', ' ').title()
                f.write(f"- [{file_name}]({relative_path})\n")
        
        logger.info(f"Index file created: {index_path}")
        return str(index_path)
        
    except Exception as e:
        logger.error(f"Error creating index file: {e}")
        return ""


def get_relative_path(file_path: str, base_path: str) -> str:
    """
    Get relative path from base path
    
    Args:
        file_path: Full file path
        base_path: Base directory path
        
    Returns:
        Relative path string
    """
    try:
        return str(Path(file_path).relative_to(Path(base_path)))
    except ValueError:
        return str(Path(file_path).name)


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate MD5 hash of file for duplicate detection
    
    Args:
        file_path: Path to file
        
    Returns:
        MD5 hash string
    """
    import hashlib
    
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5()
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
        return file_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return ""
