import os
from typing import Optional, List, Dict, Any
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient, AnalyzeResult
from loguru import logger
from config.settings import AZURE_DOCUMENT_AI_ENDPOINT, AZURE_DOCUMENT_AI_KEY, AZURE_DOCUMENT_AI_MODEL

class AzureDocumentAIClient:
    """Client for interacting with Azure Document AI service."""

    def __init__(self):
        self.endpoint = AZURE_DOCUMENT_AI_ENDPOINT
        self.key = AZURE_DOCUMENT_AI_KEY
        self.model_id = AZURE_DOCUMENT_AI_MODEL # 'prebuilt-read' or 'prebuilt-layout'

        if not self.endpoint or not self.key:
            raise ValueError("Azure Document AI endpoint or key is not configured.")

        self.document_analysis_client = DocumentAnalysisClient(
            endpoint=self.endpoint, credential=AzureKeyCredential(self.key)
        )
        logger.info(f"✅ Azure Document AI client initialized with model '{self.model_id}'.")

    def analyze_pdf_content(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Analyzes PDF content using Azure Document AI.
        
        Extracts full text and tables, and can be extended for other entities.
        """
        try:
            logger.info(f"🚀 Sending PDF content ({len(pdf_bytes)} bytes) to Azure Document AI for analysis.")
            
            # Use begin_analyze_document for general document analysis
            # The 'content' property of the result provides the reading-order text.
            poller = self.document_analysis_client.begin_analyze_document(
                self.model_id, pdf_bytes
            )
            result: AnalyzeResult = poller.result()

            full_text = ""
            extracted_tables = [] # Changed name to avoid conflict with method parameter

            # Prioritize extracting full text from paragraphs for better structural integrity
            if result.paragraphs:
                # Sort paragraphs by their bounding regions and page number to maintain reading order
                sorted_paragraphs = sorted(result.paragraphs, key=lambda p: (p.bounding_regions[0].page_number, p.bounding_regions[0].polygon[0].y))
                for paragraph in sorted_paragraphs:
                    full_text += paragraph.content + "\n\n" # Add newlines for paragraph separation
            elif result.content: # Fallback to raw content if no paragraphs are found
                full_text = result.content

            # Extract tables already converted to Markdown by Azure
            # Extract tables
            if result.tables:
                for i, table in enumerate(result.tables):
                    # Azure Document AI's table object often contains a 'as_markdown()' method or similar
                    # Or, the client already gives markdown formatted tables
                    # Assuming the 'content' field of the table object in the list is the markdown string
                    # based on the `direct_convert.py` changes.
                    if hasattr(table, 'as_markdown') and callable(table.as_markdown):
                        # If the table object itself has a method to get markdown
                        extracted_tables.append(table.as_markdown())
                    elif isinstance(table.content, str) and table.content.strip():
                        # If table content is already a markdown string
                        extracted_tables.append(table.content)
                    else:
                        logger.warning(f"Could not extract markdown from table {i}. Raw table object: {table}")
                        # Fallback for old table extraction if needed. For now, this old logic will be removed
                        # in favor of direct markdown output from Azure.
                        # If the direct markdown from Azure is not available, you would need
                        # to re-implement _table_to_markdown logic here or a similar helper.

            logger.info(f"✅ Azure Document AI analysis complete. Extracted {len(full_text)} characters and {len(extracted_tables)} tables.")
            
            return {
                "full_text": full_text,
                "tables": extracted_tables,
                "paragraphs": [p.content for p in result.paragraphs] if result.paragraphs else [],
                "page_count": len(result.pages) if result.pages else 0
            }

        except Exception as e:
            logger.error(f"❌ Error analyzing PDF with Azure Document AI: {e}")
            raise