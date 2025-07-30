"""
MarkerPDF API client for PDF to Markdown conversion
"""

import requests
import logging
import time
import json
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class MarkerPDFClient:
    """
    Client for MarkerPDF conversion service
    
    This client handles API communication with MarkerPDF service,
    including authentication, file upload, conversion requests,
    and result retrieval.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: str = "https://api.markerpdf.com/v1",
        timeout: int = 300,
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        """
        Initialize MarkerPDF client
        
        Args:
            api_key: MarkerPDF API key
            base_url: Base URL for MarkerPDF API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key or os.getenv('MARKERPDF_API_KEY')
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if not self.api_key:
            logger.warning("No MarkerPDF API key provided. Some operations may fail.")
        
        # Setup session with default headers
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'User-Agent': 'gemini-embedding-model/1.0.0'
            })
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test connection to MarkerPDF API
        
        Returns:
            Tuple of (success, message)
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=10
            )
            
            if response.status_code == 200:
                return True, "Connection successful"
            else:
                return False, f"API returned status {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return False, f"Connection failed: {e}"
    
    # Replace the convert_pdf_to_markdown method in markerpdf_client.py
    def convert_pdf_to_markdown(
        self, 
        pdf_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Convert PDF to markdown using MistralAI API
        """
        if not self.api_key:
            return False, "", {"error": "No API key provided"}
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return False, "", {"error": f"File not found: {pdf_path}"}
        
        try:
            # For MistralAI, we'll use a different approach
            # First, let's use a local conversion method as MistralAI doesn't directly convert PDFs
            import pymupdf4llm
            
            logger.info(f"Converting {pdf_path.name} using local pymupdf4llm")
            
            # Convert using local library
            markdown_content = pymupdf4llm.to_markdown(str(pdf_path))
            
            metadata = {
                "processing_method": "pymupdf4llm",
                "file_size": pdf_path.stat().st_size,
                "content_length": len(markdown_content),
                "success": True,
                "processing_time": time.time()
            }
            
            logger.info(f"Successfully converted {pdf_path.name}")
            return True, markdown_content, metadata
            
        except Exception as e:
            logger.error(f"Error converting {pdf_path.name}: {e}")
            return False, "", {"error": str(e), "success": False}

    
    def _upload_and_convert(
        self, 
        pdf_path: Path, 
        options: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Upload PDF file and perform conversion
        
        Args:
            pdf_path: Path to PDF file
            options: Conversion options
            
        Returns:
            Tuple of (success, content, metadata)
        """
        try:
            # Step 1: Upload file
            upload_success, file_id, upload_metadata = self._upload_file(pdf_path)
            if not upload_success:
                return False, "", upload_metadata
            
            # Step 2: Request conversion
            conversion_success, job_id, conversion_metadata = self._request_conversion(
                file_id, options
            )
            if not conversion_success:
                return False, "", conversion_metadata
            
            # Step 3: Poll for completion and get result
            result_success, content, result_metadata = self._get_conversion_result(job_id)
            
            # Combine metadata
            combined_metadata = {
                **upload_metadata,
                **conversion_metadata,
                **result_metadata
            }
            
            return result_success, content, combined_metadata
            
        except Exception as e:
            logger.error(f"Error in upload and convert process: {e}")
            return False, "", {"error": str(e)}
    
    def _upload_file(self, pdf_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
        """Upload PDF file to MarkerPDF"""
        try:
            with open(pdf_path, 'rb') as f:
                files = {'file': (pdf_path.name, f, 'application/pdf')}
                
                response = self.session.post(
                    f"{self.base_url}/upload",
                    files=files,
                    timeout=self.timeout
                )
            
            if response.status_code == 200:
                result = response.json()
                file_id = result.get('file_id')
                
                metadata = {
                    "file_id": file_id,
                    "file_size": pdf_path.stat().st_size,
                    "upload_time": time.time()
                }
                
                return True, file_id, metadata
            else:
                return False, "", {"error": f"Upload failed with status {response.status_code}"}
                
        except Exception as e:
            return False, "", {"error": f"Upload error: {e}"}
    
    def _request_conversion(
        self, 
        file_id: str, 
        options: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Request conversion of uploaded file"""
        try:
            payload = {
                "file_id": file_id,
                **options
            }
            
            response = self.session.post(
                f"{self.base_url}/convert",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                job_id = result.get('job_id')
                
                metadata = {
                    "job_id": job_id,
                    "conversion_start_time": time.time(),
                    "options": options
                }
                
                return True, job_id, metadata
            else:
                return False, "", {"error": f"Conversion request failed with status {response.status_code}"}
                
        except Exception as e:
            return False, "", {"error": f"Conversion request error: {e}"}
    
    def _get_conversion_result(
        self, 
        job_id: str,
        max_wait_time: int = 300
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Poll for conversion completion and retrieve result"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                response = self.session.get(
                    f"{self.base_url}/status/{job_id}",
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    status = result.get('status')
                    
                    if status == 'completed':
                        # Get the converted content
                        content_response = self.session.get(
                            f"{self.base_url}/result/{job_id}",
                            timeout=self.timeout
                        )
                        
                        if content_response.status_code == 200:
                            content = content_response.text
                            
                            metadata = {
                                "completion_time": time.time(),
                                "processing_duration": time.time() - start_time,
                                "status": "completed",
                                "content_length": len(content)
                            }
                            
                            return True, content, metadata
                        else:
                            return False, "", {"error": "Failed to retrieve conversion result"}
                    
                    elif status == 'failed':
                        error_msg = result.get('error', 'Conversion failed')
                        return False, "", {"error": error_msg, "status": "failed"}
                    
                    elif status in ['pending', 'processing']:
                        # Continue polling
                        time.sleep(5)
                        continue
                    
                    else:
                        return False, "", {"error": f"Unknown status: {status}"}
                
                else:
                    return False, "", {"error": f"Status check failed with status {response.status_code}"}
                    
            except Exception as e:
                logger.error(f"Error checking conversion status: {e}")
                time.sleep(5)
        
        return False, "", {"error": "Conversion timed out"}
    
    def get_account_info(self) -> Tuple[bool, Dict[str, Any]]:
        """Get account information and usage statistics"""
        if not self.api_key:
            return False, {"error": "No API key provided"}
        
        try:
            response = self.session.get(
                f"{self.base_url}/account",
                timeout=30
            )
            
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {"error": f"Failed to get account info: {response.status_code}"}
                
        except Exception as e:
            return False, {"error": f"Account info request failed: {e}"}
    
    def __del__(self):
        """Cleanup session on object destruction"""
        if hasattr(self, 'session'):
            self.session.close()
