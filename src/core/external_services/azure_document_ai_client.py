import os
from typing import Dict, Any, List, Optional
from loguru import logger
from pathlib import Path
import sys
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Adjust import path based on actual settings.py location relative to this file
# Assuming settings.py is in project_root/config/settings.py
# and this file is in project_root/src/core/external_services/
project_root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root_dir))
from config.settings import AZURE_DOCUMENT_AI_ENDPOINT, AZURE_DOCUMENT_AI_KEY, AZURE_DOCUMENT_AI_MODEL


class AzureDocumentAIClient:
    """Client for interacting with Azure Document Intelligence (Document AI) service."""

    def __init__(self):
        if not AZURE_DOCUMENT_AI_ENDPOINT or not AZURE_DOCUMENT_AI_KEY:
            logger.error("❌ Azure Document AI endpoint or key not configured in settings.")
            raise ValueError("Azure Document AI credentials are required.")
        
        self.endpoint = AZURE_DOCUMENT_AI_ENDPOINT
        self.key = AZURE_DOCUMENT_AI_KEY
        self.model_id = AZURE_DOCUMENT_AI_MODEL
        
        try:
            self.client = DocumentIntelligenceClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.key)
            )
            logger.info(f"✅ Azure Document AI client initialized with model '{self.model_id}'.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure Document AI client: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3), # Retry up to 3 times
        wait=wait_exponential(multiplier=1, min=4, max=10), # Exponential backoff between retries
        retry=retry_if_exception_type(HttpResponseError) # Only retry on HTTP errors
    )
    def analyze_pdf_content(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Analyzes PDF content using Azure Document AI and returns extracted data.

        Args:
            pdf_bytes (bytes): The raw bytes content of the PDF file.

        Returns:
            Dict[str, Any]: A dictionary containing extracted text, tables, and other data.
                            Returns an empty dict on failure.
        """
        logger.info(f"🚀 Sending PDF content ({len(pdf_bytes)} bytes) to Azure Document AI for analysis.")
        try:
            # Use 'prebuilt-layout' model for general document processing including text, tables, and structure.
            # For pure OCR on images/scanned docs, 'prebuilt-read' is an option.
            poller = self.client.begin_analyze_document(self.model_id, pdf_bytes)
            
            # Wait for the analysis to complete
            result = poller.result()
            
            extracted_text = ""
            for page in result.pages:
                if page.lines:
                    extracted_text += "\n".join([line.content for line in page.lines]) + "\n"
            
            extracted_tables = []
            if result.tables:
                for table in result.tables:
                    rows = []
                    # Create a 2D array representation of the table
                    # Ensure max_col_index and max_row_index are at least 0
                    max_col_index = max([cell.column_index for cell in table.cells] + [0])
                    max_row_index = max([cell.row_index for cell in table.cells] + [0])
                    
                    table_array = [['' for _ in range(max_col_index + 1)] for _ in range(max_row_index + 1)]
                    
                    for cell in table.cells:
                        if 0 <= cell.row_index <= max_row_index and 0 <= cell.column_index <= max_col_index:
                            table_array[cell.row_index][cell.column_index] = cell.content
                    extracted_tables.append(table_array)

            logger.info(f"✅ Azure Document AI analysis complete. Extracted {len(extracted_text)} characters and {len(extracted_tables)} tables.")
            
            return {
                "full_text": extracted_text,
                "tables": extracted_tables,
                "paragraphs": [p.content for p in result.paragraphs] if result.paragraphs else [],
                "document_model_result": result # Store full result for debugging if needed
            }
        except HttpResponseError as e:
            logger.error(f"❌ Azure Document AI API error (Status: {e.status_code}): {e.message}")
            if e.status_code == 429:
                logger.warning("Azure Document AI rate limit hit. Retrying...")
            raise # Re-raise for tenacity to handle
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred during Azure Document AI analysis: {e}")
            return {} # Return empty on unexpected errors