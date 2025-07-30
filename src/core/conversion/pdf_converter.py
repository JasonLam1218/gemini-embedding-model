"""
PDF to Markdown converter using MarkerPDF service

This module provides the main PDFConverter class that orchestrates
the conversion of PDF files to Markdown format, with support for
batch processing, organized output, and comprehensive reporting.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import concurrent.futures
from threading import Lock

from .markerpdf_client import MarkerPDFClient
from .conversion_utils import (
    sanitize_filename,
    get_pdf_files,
    validate_pdf_file,
    ensure_output_directory,
    create_conversion_report,
    create_markdown_index,
    get_relative_path
)

logger = logging.getLogger(__name__)


class PDFConverter:
    """
    Main PDF conversion class that handles the conversion of PDF files
    to Markdown format using the MarkerPDF service.
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize PDF converter
        
        Args:
            input_dir: Input directory containing PDF files
            output_dir: Output directory for converted markdown files
            config: Optional configuration dictionary
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config or {}
        
        # Initialize MarkerPDF client
        markerpdf_config = self.config.get('markerpdf', {})
        self.client = MarkerPDFClient(
            api_key=markerpdf_config.get('api_key'),
            base_url=markerpdf_config.get('base_url', 'https://api.markerpdf.com/v1'),
            timeout=markerpdf_config.get('timeout', 300),
            max_retries=markerpdf_config.get('max_retries', 3),
            retry_delay=markerpdf_config.get('retry_delay', 5)
        )
        
        # Conversion settings
        conversion_config = self.config.get('conversion', {})
        self.batch_size = conversion_config.get('batch_size', 5)
        self.max_file_size_mb = conversion_config.get('max_file_size_mb', 50)
        self.supported_formats = conversion_config.get('supported_formats', ['.pdf'])
        
        # Output settings
        output_config = self.config.get('output', {})
        self.create_reports = output_config.get('create_conversion_reports', True)
        self.preserve_structure = output_config.get('preserve_directory_structure', True)
        self.sanitize_names = output_config.get('sanitize_filenames', True)
        self.create_index = output_config.get('create_index_file', True)
        
        # Thread safety
        self._lock = Lock()
        self._conversion_results = []
        
        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"PDFConverter initialized - Session: {self.session_id}")
    
    def convert_all_pdfs(self) -> Dict[str, Any]:
        """
        Convert all PDFs in the input directory to markdown
        
        Returns:
            Dictionary containing conversion results and statistics
        """
        logger.info(f"Starting conversion of all PDFs in {self.input_dir}")
        
        # Find all PDF files
        pdf_files = get_pdf_files(str(self.input_dir), recursive=True)
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.input_dir}")
            return {
                "session_id": self.session_id,
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "results": []
            }
        
        logger.info(f"Found {len(pdf_files)} PDF files to convert")
        
        # Process files in batches
        all_results = []
        for i in range(0, len(pdf_files), self.batch_size):
            batch = pdf_files[i:i + self.batch_size]
            logger.info(f"Processing batch {i // self.batch_size + 1} ({len(batch)} files)")
            
            batch_results = self._convert_batch(batch)
            all_results.extend(batch_results)
        
        # Compile final results
        successful = len([r for r in all_results if r['success']])
        failed = len(all_results) - successful
        
        results_summary = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "total_files": len(pdf_files),
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / len(pdf_files) * 100) if pdf_files else 0,
            "results": all_results
        }
        
        # Create reports and index
        if self.create_reports:
            self._generate_reports(results_summary)
        
        if self.create_index:
            self._create_index_files(all_results)
        
        logger.info(f"Conversion completed - {successful}/{len(pdf_files)} files successful")
        return results_summary
    
    def convert_directory(
        self, 
        input_subdir: str, 
        output_subdir: str
    ) -> Dict[str, Any]:
        """
        Convert PDFs from a specific subdirectory
        
        Args:
            input_subdir: Input subdirectory path
            output_subdir: Output subdirectory path
            
        Returns:
            Dictionary containing conversion results
        """
        logger.info(f"Converting PDFs from {input_subdir} to {output_subdir}")
        
        # Temporarily override directories
        original_input = self.input_dir
        original_output = self.output_dir
        
        self.input_dir = Path(input_subdir)
        self.output_dir = Path(output_subdir)
        
        try:
            results = self.convert_all_pdfs()
            return results
        finally:
            # Restore original directories
            self.input_dir = original_input
            self.output_dir = original_output
    
    def convert_single_pdf(
        self, 
        pdf_path: str, 
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert a single PDF to markdown
        
        Args:
            pdf_path: Path to PDF file
            output_path: Optional custom output path
            
        Returns:
            Dictionary containing conversion result
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Converting single PDF: {pdf_path.name}")
        
        # Validate file
        is_valid, error_msg = validate_pdf_file(str(pdf_path), self.max_file_size_mb)
        if not is_valid:
            result = {
                "input_file": str(pdf_path),
                "output_file": "",
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
            logger.error(f"Validation failed for {pdf_path}: {error_msg}")
            return result
        
        # Determine output path
        if output_path is None:
            if self.sanitize_names:
                output_filename = sanitize_filename(pdf_path.name)
            else:
                output_filename = pdf_path.stem + ".md"
            
            output_path = self.output_dir / output_filename
        
        # Ensure output directory exists
        ensure_output_directory(str(output_path))
        
        # Perform conversion
        result = self._convert_file(str(pdf_path), str(output_path))
        
        return result
    
    def _convert_batch(self, pdf_files: List[str]) -> List[Dict[str, Any]]:
        """
        Convert a batch of PDF files using parallel processing
        
        Args:
            pdf_files: List of PDF file paths
            
        Returns:
            List of conversion result dictionaries
        """
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all conversion tasks
            future_to_file = {}
            for pdf_file in pdf_files:
                output_file = self._get_output_path(pdf_file)
                future = executor.submit(self._convert_file, pdf_file, output_file)
                future_to_file[future] = pdf_file
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                result = future.result()
                results.append(result)
                
                # Log progress
                pdf_file = future_to_file[future]
                if result['success']:
                    logger.info(f"✓ Converted: {Path(pdf_file).name}")
                else:
                    logger.error(f"✗ Failed: {Path(pdf_file).name} - {result.get('error', 'Unknown error')}")
        
        return results
    
    def _convert_file(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Convert a single file with error handling and metadata collection
        
        Args:
            input_path: Input PDF file path
            output_path: Output markdown file path
            
        Returns:
            Dictionary containing conversion result and metadata
        """
        start_time = datetime.now()
        input_file = Path(input_path)
        
        # Initialize result dictionary
        result = {
            "input_file": str(input_path),
            "output_file": str(output_path),
            "success": False,
            "error": "",
            "timestamp": start_time.isoformat(),
            "processing_time": 0,
            "file_size": 0,
            "output_size": 0
        }
        
        try:
            # Get file size
            result["file_size"] = input_file.stat().st_size
            
            # Validate file
            is_valid, error_msg = validate_pdf_file(input_path, self.max_file_size_mb)
            if not is_valid:
                result["error"] = error_msg
                return result
            
            # Ensure output directory exists
            ensure_output_directory(output_path)
            
            # Convert using MarkerPDF
            success, markdown_content, metadata = self.client.convert_pdf_to_markdown(
                input_path,
                options=self._get_conversion_options()
            )
            
            if success and markdown_content:
                # Write markdown content to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
                # Update result
                result.update({
                    "success": True,
                    "output_size": len(markdown_content),
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "metadata": metadata
                })
                
            else:
                result["error"] = metadata.get('error', 'Conversion failed')
                
        except Exception as e:
            result["error"] = str(e)
            logger.exception(f"Unexpected error converting {input_path}")
        
        finally:
            result["processing_time"] = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _get_output_path(self, input_path: str) -> str:
        """
        Generate output path for converted markdown file
        
        Args:
            input_path: Input PDF file path
            
        Returns:
            Output markdown file path
        """
        input_file = Path(input_path)
        
        # Determine relative path structure
        if self.preserve_structure:
            try:
                relative_path = input_file.relative_to(self.input_dir)
                output_subdir = self.output_dir / relative_path.parent
            except ValueError:
                # File is not under input_dir
                output_subdir = self.output_dir
        else:
            output_subdir = self.output_dir
        
        # Generate output filename
        if self.sanitize_names:
            output_filename = sanitize_filename(input_file.name)
        else:
            output_filename = input_file.stem + ".md"
        
        return str(output_subdir / output_filename)
    
    def _get_conversion_options(self) -> Dict[str, Any]:
        """Get conversion options for MarkerPDF"""
        conversion_config = self.config.get('conversion', {})
        return {
            "output_format": conversion_config.get('output_format', 'markdown'),
            "preserve_images": conversion_config.get('preserve_images', True),
            "extract_tables": conversion_config.get('extract_tables', True),
            "preserve_formatting": conversion_config.get('preserve_formatting', True)
        }
    
    def _generate_reports(self, results_summary: Dict[str, Any]) -> None:
        """Generate conversion reports"""
        try:
            reports_dir = self.config.get('paths', {}).get('reports', 'data/output/conversion_reports')
            
            # Create detailed JSON report
            report_path = create_conversion_report(
                results_summary['results'],
                reports_dir,
                self.session_id
            )
            
            if report_path:
                logger.info(f"Detailed report created: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
    
    def _create_index_files(self, results: List[Dict[str, Any]]) -> None:
        """Create index files for converted documents"""
        try:
            # Get successful conversions
            successful_files = [
                r['output_file'] for r in results 
                if r['success'] and r['output_file']
            ]
            
            if successful_files:
                index_path = create_markdown_index(
                    successful_files,
                    str(self.output_dir),
                    "Converted PDF Documents"
                )
                
                if index_path:
                    logger.info(f"Index file created: {index_path}")
            
        except Exception as e:
            logger.error(f"Error creating index files: {e}")
    
    def get_conversion_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current conversion session"""
        return {
            "session_id": self.session_id,
            "input_directory": str(self.input_dir),
            "output_directory": str(self.output_dir),
            "configuration": self.config,
            "client_status": self.client.test_connection()
        }
