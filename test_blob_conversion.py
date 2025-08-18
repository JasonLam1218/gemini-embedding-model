#!/usr/bin/env python3
"""
Test script for the Vercel Blob to Markdown conversion feature,
prioritizing Azure Document AI.
"""

import sys
import os
import shutil
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import re # Import regex for enhanced verification

# Add project root to Python path if not already added
project_root_dir = Path(__file__).parent.parent
if str(project_root_dir) not in sys.path:
    sys.path.insert(0, str(project_root_dir))

# Initialize logging system
from src.core.utils.logging_config import initialize_logging, log_pipeline_start, log_pipeline_end
from loguru import logger

# Initialize logging for the test script
logs_dir = initialize_logging()
logger.info("🚀 Starting test_blob_conversion.py with comprehensive file logging")
logger.info(f"📁 All test logs will be saved to: {logs_dir}")

# Import the main conversion function from direct_convert.py
# This import needs to correctly resolve direct_convert.py relative to the project root
try:
    from scripts.direct_convert import convert_pdfs_from_vercel_blobs
except ImportError as e:
    logger.error(f"❌ Failed to import convert_pdfs_from_vercel_blobs. Make sure scripts/direct_convert.py exists and is accessible. Error: {e}")
    sys.exit(1)

# Define paths
OUTPUT_DIR = Path("data/output")
CONVERTED_MARKDOWN_DIR = OUTPUT_DIR / "converted_markdown"
TEST_LOGS_DIR = OUTPUT_DIR / "logs" # Assuming logs are stored here based on logging_config

def setup_test_environment():
    """Sets up a clean environment for the test."""
    logger.info("Setting up test environment. Clearing data/output/converted_markdown if it exists.")
    if CONVERTED_MARKDOWN_DIR.exists():
        try:
            shutil.rmtree(CONVERTED_MARKDOWN_DIR)
            logger.info(f"Cleared existing directory: {CONVERTED_MARKDOWN_DIR}")
        except Exception as e:
            logger.error(f"Failed to clear directory {CONVERTED_MARKDOWN_DIR}: {e}")
            raise
    CONVERTED_MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Test environment setup complete.")

def verify_conversion_results(conversion_report: Dict[str, Any], expected_files: List[Dict]) -> bool:
    """Verifies the output of the PDF conversion."""
    logger.info("\n--- Verification ---")

    overall_test_success = True

    # 1. Check if output directory exists
    if CONVERTED_MARKDOWN_DIR.exists():
        logger.info(f"✅ Output directory exists: {CONVERTED_MARKDOWN_DIR}")
    else:
        logger.error(f"❌ Output directory does not exist: {CONVERTED_MARKDOWN_DIR}")
        overall_test_success = False
        return overall_test_success # Early exit if dir not found

    # 2. Check conversion report statistics
    report_stats = conversion_report.get('statistics', {})
    logger.info(f"Conversion Report Stats: {report_stats}")

    if report_stats.get('successful_conversions', 0) != len(expected_files):
        logger.error(f"❌ Reported successful conversions ({report_stats.get('successful_conversions', 0)}) does not match expected ({len(expected_files)}).")
        overall_test_success = False
    else:
        logger.info(f"✅ Reported successful conversions: {report_stats.get('successful_conversions', 0)}")
    
    if report_stats.get('failed_conversions', 0) > 0:
        logger.error(f"❌ Conversion report indicates {report_stats.get('failed_conversions', 0)} failed conversions.")
        overall_test_success = False

    # 3. Verify each expected file
    for file_info in expected_files:
        original_filename_stem = Path(file_info['original_filename']).stem.replace(' ', '_').replace('-', '_')
        expected_md_filename = f"{original_filename_stem}.md"
        expected_md_path = CONVERTED_MARKDOWN_DIR / file_info['category'] / expected_md_filename

        logger.info(f"\nChecking file: {expected_md_path.relative_to(OUTPUT_DIR)}")

        if not expected_md_path.exists():
            logger.error(f"❌ Markdown file not found: {expected_md_path}")
            overall_test_success = False
            continue

        try:
            content = expected_md_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"❌ Failed to read content from {expected_md_path}: {e}")
            overall_test_success = False
            continue

        if not content.strip():
            logger.error(f"❌ Markdown file {expected_md_path.name} is empty.")
            overall_test_success = False
            continue
        
        # Enhanced verification of conversion method
        method_line_match = re.search(r'^\*\*Method:\*\* (.+)$', content, re.MULTILINE)
        if method_line_match:
            actual_method = method_line_match.group(1).strip()
            # In this test, we expect Azure Document AI to be used if configured
            expected_method_in_file = "azure_document_ai" 
            
            if actual_method == expected_method_in_file:
                logger.info(f"✅ Verified {expected_md_path.name} was converted using '{actual_method}'.")
            else:
                logger.error(f"❌ Test failed: Markdown file {expected_md_path.name} was expected to be converted by '{expected_method_in_file}', but found '{actual_method}'.")
                overall_test_success = False
        else:
            logger.error(f"❌ Test failed: Could not find 'Method:' line in the Markdown header for {expected_md_path.name}. Content preview: {content[:500]}...")
            overall_test_success = False

        # Additional content verification (optional, adapt to your specific test needs)
        if "Intended Learning Outcomes" in content and "kelvin_papers" in str(expected_md_path):
             logger.error(f"❌ Test failed: Markdown file {expected_md_path.name} (in kelvin_papers) contains lecture content ('Intended Learning Outcomes'). Content preview: {content[:500]}")
             overall_test_success = False
        elif "explain" in content.lower() and "calculate" in content.lower() and "program" in content.lower() and "lectures" in str(expected_md_path):
             # This is a very rough heuristic, adapt as needed for your actual lecture content.
             # This just ensures we don't accidentally get exam-like content in a lecture file
             # if the blob URL was wrong again.
             logger.debug(f"ℹ️ {expected_md_path.name} (lectures) contains common educational terms.")

    if overall_test_success:
        logger.info("\n🎉 All Vercel Blob conversion verification tests PASSED!")
    else:
        logger.error("\n💔 Some Vercel Blob conversion verification tests FAILED. Please review the logs above.")
    
    return overall_test_success

async def run_blob_conversion_test():
    """Runs the full test for Vercel Blob PDF conversion."""
    log_pipeline_start("run_blob_conversion_test", {"description": "Testing Vercel Blob to Markdown conversion."})
    start_time = time.time()
    overall_test_status = False # Initialize to False, set to True only if all checks pass

    # --- CORRECTED VERCEL BLOB FILES CONFIGURATION ---
    # Ensure these URLs point to the correct content type and match the filename/category.
    # The URLs below are examples from your logs. Replace with your actual blob URLs.
    vercel_blob_files = [
        {
            'url': 'https://88avsgpdqmsyih7d.public.blob.vercel-storage.com/1755497259449-3_Introduction_to_Data_Science.pdf',
            'category': 'lectures',  # Corrected category to 'lectures'
            'original_filename': '3_Introduction_to_Data_Science.pdf' # Corrected filename
        },
        {
            'url': 'https://88avsgpdqmsyih7d.public.blob.vercel-storage.com/1755497259641-4_Statistical_Analysis_for_Data_Analytics.pdf',
            'category': 'lectures',
            'original_filename': '4_Statistical_Analysis_for_Data_Analytics.pdf' # Corrected filename
        }
        # Add more files here if needed, e.g., a real exam paper if you have its blob URL:
        # {
        #     'url': 'YOUR_ACTUAL_EXAM_PAPER_BLOB_URL.pdf',
        #     'category': 'kelvin_papers',
        #     'original_filename': 'example_exam_paper_set1.pdf'
        # }
    ]

    logger.info(f"Attempting to test with {len(vercel_blob_files)} Vercel Blob files.")

    try:
        setup_test_environment()
        logger.info("Starting Vercel Blob conversion test, prioritizing Azure Document AI...")
        
        # Perform the conversion
        conversion_report = await convert_pdfs_from_vercel_blobs(vercel_blob_files)
        
        duration = time.time() - start_time
        logger.info(f"Blob conversion test completed in {duration:.2f} seconds.")

        # Verify the results
        overall_test_status = verify_conversion_results(conversion_report, vercel_blob_files)

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Test failed due to an unexpected error: {e}")
        overall_test_status = False
    finally:
        log_pipeline_end("run_blob_conversion_test", success=overall_test_status, duration=duration)
        if not overall_test_status:
            logger.error("Please check the logs and 'data/output/converted_markdown' directory for more details.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_blob_conversion_test())