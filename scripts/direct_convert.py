#!/usr/bin/env python3
"""
Enhanced PDF to Markdown conversion with multiple approaches and image extraction.
Handles complex PDFs, extracts images, and provides fallback conversion methods.
This version is adapted to fetch PDF files from Vercel Blob Storage URLs
and integrates Azure Document AI for superior OCR.
"""

import pymupdf4llm
import fitz  # PyMuPDF
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import hashlib
import base64
from datetime import datetime
import requests # NEW: For downloading files from URLs
import tempfile # NEW: For creating temporary files
import sys

# Import Azure Document AI client
from src.core.external_services.azure_document_ai_client import AzureDocumentAIClient
# Import Azure settings
from config.settings import AZURE_DOCUMENT_AI_ENDPOINT, AZURE_DOCUMENT_AI_KEY

# Additional imports for alternative conversion methods
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("⚠️ pdfplumber not available - some conversion methods disabled")

try:
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    logger.warning("⚠️ pdfminer not available - some conversion methods disabled")

class EnhancedPDFConverter:
    """Enhanced PDF to Markdown converter with multiple approaches and image extraction"""
    
    def __init__(self):
        self.base_output = Path("data/output/converted_markdown")
        self.image_output = self.base_output / "images"
        
        # Create directories
        self.base_output.mkdir(parents=True, exist_ok=True)
        self.image_output.mkdir(parents=True, exist_ok=True)
        
        # Conversion statistics
        self.conversion_stats = {
            "total_files": 0,
            "successful_conversions": 0,
            "failed_conversions": 0,
            "images_extracted": 0,
            "empty_conversions": 0,
            "fallback_used": 0
        }

        self.azure_client: Optional[AzureDocumentAIClient] = None
        if AZURE_DOCUMENT_AI_ENDPOINT and AZURE_DOCUMENT_AI_KEY:
            try:
                self.azure_client = AzureDocumentAIClient()
                logger.info("Azure Document AI client initialized successfully.")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize Azure Document AI client: {e}. Azure OCR will not be used.")
        else:
            logger.info("Azure Document AI credentials not provided. Azure OCR will not be used.")
        
        logger.info("✅ Enhanced PDF Converter initialized")

    def convert_blobs_to_markdown(self, blob_files: List[Dict]) -> Dict[str, Any]:
        """Convert PDFs from Vercel Blob URLs to Markdown.
        
        Args:
            blob_files (List[Dict]): A list of dictionaries, where each dict
                                    contains 'url', 'category', and 'original_filename'.
                                    Example: [{'url': '...', 'category': 'lectures', 'original_filename': 'lecture1.pdf'}]
        """
        logger.info("🚀 Starting Enhanced PDF to Markdown Conversion from Vercel Blobs")
        logger.info("=" * 60)

        # Reset stats for a new conversion run
        self.conversion_stats = {
            "total_files": 0, "successful_conversions": 0, "failed_conversions": 0,
            "images_extracted": 0, "empty_conversions": 0, "fallback_used": 0
        }

        # Group blob files by category to maintain output directory structure
        categories_data: Dict[str, List[Dict]] = {}
        for blob_file_info in blob_files:
            category = blob_file_info.get('category', 'unknown').lower() # 'lectures', 'kelvin_papers'
            if category not in categories_data:
                categories_data[category] = []
            categories_data[category].append(blob_file_info)
        
        if not categories_data:
            logger.warning("⚠️ No valid blob file information provided for conversion.")
            return self._generate_final_report() # Return empty report

        for category_name, files_in_category in categories_data.items():
            logger.info(f"\n📚 Processing category from blobs: {category_name.replace('_', ' ').title()}")
            
            output_category_dir = self.base_output / category_name
            output_category_dir.mkdir(parents=True, exist_ok=True)

            category_image_dir = self.image_output / category_name
            category_image_dir.mkdir(parents=True, exist_ok=True)

            self.conversion_stats["total_files"] += len(files_in_category)

            for file_info in files_in_category:
                pdf_url = file_info['url']
                original_filename = file_info.get('original_filename', Path(pdf_url).name)
                
                success = self._download_and_convert_single_pdf_blob(
                    pdf_url, original_filename, output_category_dir, category_image_dir, category_name
                )
                if success:
                    self.conversion_stats["successful_conversions"] += 1
                else:
                    self.conversion_stats["failed_conversions"] += 1

        return self._generate_final_report()

    def _download_and_convert_single_pdf_blob(self, pdf_url: str, original_filename: str,
                                              output_dir: Path, image_dir: Path, category: str) -> bool:
        """Downloads a PDF from a given URL and then converts it to markdown."""
        logger.info(f" ⬇️ Downloading: {original_filename} from {pdf_url}...")
        
        temp_pdf_file_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                response = requests.get(pdf_url, stream=True)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                temp_pdf_file_path = Path(tmp_file.name)
                logger.info(f"  ✅ Downloaded to temporary file: {temp_pdf_file_path.name}")
            
            success = self._convert_pdf_from_path(
                temp_pdf_file_path, original_filename, output_dir, image_dir, category
            )
            return success
        except requests.exceptions.RequestException as e:
            logger.error(f"  ❌ Failed to download {original_filename} from {pdf_url}: {e}")
            return False
        except Exception as e:
            logger.error(f"  ❌ Error during conversion of downloaded file {original_filename}: {e}")
            return False
        finally:
            if temp_pdf_file_path and temp_pdf_file_path.exists():
                os.unlink(temp_pdf_file_path)
                logger.debug(f"  🗑️ Cleaned up temporary file: {temp_pdf_file_path.name}")

    def _convert_pdf_from_path(self, pdf_file_path: Path, original_filename: str, output_dir: Path, 
                               image_dir: Path, category: str) -> bool:
        """Internal method to convert a single PDF from a local path (could be temp or actual input)"""
        logger.info(f" 📄 Converting: {original_filename} (from {pdf_file_path.name})...")
        
        clean_name = Path(original_filename).stem.replace(' ', '_').replace('-', '_')
        output_file = output_dir / f"{clean_name}.md"
        
        conversion_methods = []
        # Prioritize Azure Document AI if initialized
        if self.azure_client:
            conversion_methods = [("azure_document_ai", lambda p, i, n: self._convert_with_azure_document_ai(p, i, n))]
        else:
            # Add existing methods as fallbacks
            conversion_methods.extend([
                ("pymupdf4llm", lambda p, i, n: self._convert_with_pymupdf4llm(p, i, n)),
                ("fitz_enhanced", lambda p, i, n: self._convert_with_fitz_enhanced(p, i, n)),
                ("pdfplumber", lambda p, i, n: self._convert_with_pdfplumber(p, i, n)),
                ("pdfminer", lambda p, i, n: self._convert_with_pdfminer(p, i, n))
            ])
        
        images_extracted = 0
        best_content = ""
        best_method = ""
        
        for method_name, method_func in conversion_methods:
            try:
                content, method_images = method_func(pdf_file_path, image_dir, clean_name)
                
                if self._is_good_conversion(content):
                    best_content = content
                    best_method = method_name
                    images_extracted += method_images
                    logger.info(f"  ✅ Success with {method_name}")
                    break
                else:
                    logger.warning(f"  ⚠️ Poor quality with {method_name} for {original_filename}")
                    
            except Exception as e:
                logger.warning(f"  ❌ {method_name} failed for {original_filename}: {e}")
                continue
        
        if best_content and len(best_content.strip()) > 100:
            enhanced_content = self._enhance_markdown_content(
                best_content, Path(original_filename), best_method, images_extracted
            )
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(enhanced_content)
                
            self.conversion_stats["images_extracted"] += images_extracted
            logger.info(f"  ✅ Saved: {output_file.name} ({best_method}, {images_extracted} images)")
            return True
        else:
            placeholder_content = self._create_error_placeholder(Path(original_filename), "All conversion methods failed")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(placeholder_content)
                
            self.conversion_stats["empty_conversions"] += 1
            logger.error(f"  ❌ Failed: {original_filename} - all methods failed or produced empty content.")
            return False

    def convert_all_pdfs_from_local(self) -> Dict[str, Any]:
        """Convert all PDFs from local data/input directory with enhanced processing and image extraction"""
        logger.info("🚀 Starting Enhanced PDF to Markdown Conversion from LOCAL files")
        logger.info("=" * 60)

        self.conversion_stats = {
            "total_files": 0, "successful_conversions": 0, "failed_conversions": 0,
            "images_extracted": 0, "empty_conversions": 0, "fallback_used": 0
        }

        categories = {
            "Kelvin Papers": {
                "input": Path("data/input") / "kelvin_papers",
                "output": self.base_output / "kelvin_papers"
            },
            "Lectures": {
                "input": Path("data/input") / "lectures",
                "output": self.base_output / "lectures"
            }
        }

        for category_name, paths in categories.items():
            logger.info(f"\n📚 Processing {category_name}...")
            self._process_category(category_name, paths)

        return self._generate_final_report()

    def _process_category(self, category_name: str, paths: Dict[str, Path]):
        """Process a category of PDF files from local input directory"""
        logger.info(f"Input: {paths['input']}")
        logger.info(f"Output: {paths['output']}")
        
        paths['output'].mkdir(parents=True, exist_ok=True)
        category_image_dir = self.image_output / category_name.lower().replace(" ", "_")
        category_image_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(paths['input'].glob("*.pdf"))
        self.conversion_stats["total_files"] += len(pdf_files)
        
        if not pdf_files:
            logger.warning(f"⚠️ No PDF files found in {paths['input']}")
            return
            
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            success = self._convert_pdf_from_path(
                pdf_file, pdf_file.name, paths['output'], category_image_dir, category_name
            )
            if success:
                self.conversion_stats["successful_conversions"] += 1
            else:
                self.conversion_stats["failed_conversions"] += 1

    # NEW: Method for Azure Document AI conversion
    def _convert_with_azure_document_ai(self, pdf_file: Path, image_dir: Path, 
                                        clean_name: str) -> Tuple[str, int]:
        """Convert using Azure Document AI for advanced OCR and layout extraction."""
        if not self.azure_client:
            raise RuntimeError("Azure Document AI client not initialized.")
            
        pdf_bytes = pdf_file.read_bytes()
        
        azure_result = self.azure_client.analyze_pdf_content(pdf_bytes)
        
        if not azure_result or not azure_result.get("full_text"):
            raise ValueError("Azure Document AI returned empty or invalid result.")
            
        extracted_content = azure_result["full_text"]
        
        if azure_result.get("tables"):
            table_markdown = []
            for i, table_array in enumerate(azure_result["tables"]):
                # Ensure table_array is not empty or malformed before passing to _table_to_markdown
                if table_array and any(table_array):
                    table_markdown.append(f"\n\n**Extracted Table {i+1}:**\n")
                    table_markdown.append(self._table_to_markdown(table_array))
            extracted_content += "\n".join(table_markdown)

        # Extract images using fitz (Azure Document AI processes images for OCR, but doesn't extract them as files)
        images_extracted = self._extract_images_with_fitz(pdf_file, image_dir, clean_name)
        
        return extracted_content, images_extracted

    def _convert_with_pymupdf4llm(self, pdf_file: Path, image_dir: Path, 
                                 clean_name: str) -> Tuple[str, int]:
        """Convert using pymupdf4llm with image extraction"""
        images_extracted = self._extract_images_with_fitz(pdf_file, image_dir, clean_name)
        content = pymupdf4llm.to_markdown(str(pdf_file))
        return content, images_extracted

    def _convert_with_fitz_enhanced(self, pdf_file: Path, image_dir: Path, 
                                   clean_name: str) -> Tuple[str, int]:
        """Enhanced conversion using PyMuPDF (fitz) with better formatting"""
        doc = fitz.open(pdf_file)
        content_parts = []
        images_extracted = 0
        try:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_dict = page.get_text("dict")
                page_content = self._process_fitz_text_dict(text_dict)
                if page_content.strip():
                    content_parts.append(f"\n\n---\n**Page {page_num + 1}**\n\n{page_content}")
                page_images = self._extract_page_images_fitz(page, image_dir, clean_name, page_num)
                images_extracted += page_images
        finally:
            doc.close()
        return "\n".join(content_parts), images_extracted

    def _convert_with_pdfplumber(self, pdf_file: Path, image_dir: Path, 
                                clean_name: str) -> Tuple[str, int]:
        """Convert using pdfplumber for better table handling"""
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber not available")
        import pdfplumber
        content_parts = []
        images_extracted = 0
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                tables = page.extract_tables()
                page_content = f"\n\n---\n**Page {page_num + 1}**\n\n"
                if text:
                    page_content += text + "\n\n"
                for table_num, table in enumerate(tables):
                    if table:
                        page_content += f"\n**Table {table_num + 1}:**\n\n"
                        # Ensure table is in correct format before passing to _table_to_markdown
                        # pdfplumber.extract_tables() returns a list of lists of strings, which is fine
                        page_content += self._table_to_markdown(table) + "\n\n"
                content_parts.append(page_content)
        images_extracted = self._extract_images_with_fitz(pdf_file, image_dir, clean_name)
        return "\n".join(content_parts), images_extracted

    def _convert_with_pdfminer(self, pdf_file: Path, image_dir: Path, 
                              clean_name: str) -> Tuple[str, int]:
        """Convert using pdfminer for text extraction"""
        if not PDFMINER_AVAILABLE:
            raise ImportError("pdfminer not available")
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams
        laparams = LAParams(char_margin=2.0, line_margin=0.5, word_margin=0.1, boxes_flow=0.5, detect_vertical=True)
        content = extract_text(str(pdf_file), laparams=laparams)
        images_extracted = self._extract_images_with_fitz(pdf_file, image_dir, clean_name)
        return content, images_extracted

    def _extract_images_with_fitz(self, pdf_file: Path, image_dir: Path, 
                                 clean_name: str) -> int:
        """Extract all images from PDF using fitz"""
        doc = fitz.open(pdf_file)
        images_extracted = 0
        try:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                images_extracted += self._extract_page_images_fitz(
                    page, image_dir, clean_name, page_num
                )
        finally:
            doc.close()
        return images_extracted

    def _extract_page_images_fitz(self, page, image_dir: Path, 
                                 clean_name: str, page_num: int) -> int:
        """Extract images from a single page"""
        image_list = page.get_images()
        images_saved = 0
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                if pix.width < 50 or pix.height < 50:
                    pix = None
                    continue
                
                img_filename = f"{clean_name}_page{page_num + 1}_img{img_index + 1}"
                
                if pix.n < 5:  # GRAY or RGB
                    img_path = image_dir / f"{img_filename}.png"
                    pix.save(str(img_path))
                    images_saved += 1
                    logger.debug(f"    🖼️ Saved image: {img_path.name}")
                else:  # CMYK
                    pix1 = fitz.Pixmap(fitz.csRGB, pix)
                    img_path = image_dir / f"{img_filename}.png"
                    pix1.save(str(img_path))
                    pix1 = None
                    images_saved += 1
                    logger.debug(f"    🖼️ Saved image (CMYK): {img_path.name}")
                    
                pix = None
                
            except Exception as e:
                logger.warning(f"    ⚠️ Failed to extract image {img_index}: {e}")
                
        return images_saved

    def _process_fitz_text_dict(self, text_dict: Dict) -> str:
        """Process fitz text dictionary to create better formatted markdown"""
        content_parts = []
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:  # Text block
                block_text = []
                for line in block["lines"]:
                    line_text = []
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            # Basic formatting based on font properties
                            if span["flags"] & 2**4:  # Bold
                                text = f"**{text}**"
                            if span["flags"] & 2**1:  # Italic
                                text = f"*{text}*"
                            line_text.append(text)
                    
                    if line_text:
                        block_text.append(" ".join(line_text))
                
                if block_text:
                    content_parts.append("\n".join(block_text))
        
        return "\n\n".join(content_parts)

    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """Convert table data to markdown format"""
        if not table:
            return ""
            
        markdown_lines = []
        
        # Header row
        if table:
            header = [cell or "" for cell in table[0]]
            markdown_lines.append("| " + " | ".join(header) + " |")
            markdown_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
            
            # Data rows
            for row in table[1:]:
                row_cells = [cell or "" for cell in row]
                # Pad row if shorter than header
                while len(row_cells) < len(header):
                    row_cells.append("")
                markdown_lines.append("| " + " | ".join(row_cells) + " |")
        
        return "\n".join(markdown_lines)

    def _is_good_conversion(self, content: str) -> bool:
        """Check if conversion result is of good quality"""
        if not content or len(content.strip()) < 100:
            return False
            
        word_count = len(content.split())
        if word_count < 50:
            return False
            
        for char in "._-|":
            if content.count(char) > len(content) * 0.1 and not ("| ---" in content and "|" in content): # Allow tables
                return False
                
        return True

    def _enhance_markdown_content(self, content: str, pdf_file: Path, 
                                 method: str, images_count: int) -> str:
        """Enhance markdown content with metadata and image references"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        header = f"""# {pdf_file.stem.replace('_', ' ').title()}

**Source:** {pdf_file.name}  
**Converted:** {timestamp}  
**Method:** {method}  
**Images Extracted:** {images_count}

---

"""
        if images_count > 0:
            clean_name = pdf_file.stem.replace(' ', '_').replace('-', '_')
            image_section = f"\n\n## Extracted Images\n\n"
            # This image path is for local reference, frontend needs to serve these images
            image_section += f"Images for this document are located in data/output/converted_markdown/images/{Path(pdf_file).parent.name}/ and are named like {clean_name}_pageX_imgY.png\n\n"
            # You could dynamically link if you know where the images will be hosted
            # For example: image_section += f"![Image {i+1}](<vercel-blob-image-url>/{clean_name}_page*_img{i+1}.png)\n\n"

            content += image_section
        
        return header + content

    def _create_error_placeholder(self, pdf_file: Path, error: str) -> str:
        """Create error placeholder content"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""# {pdf_file.stem.replace('_', ' ').title()} - Conversion Failed

**Source:** {pdf_file.name}  
**Conversion Attempted:** {timestamp}  
**Status:** FAILED  
**Error:** {error}

---

## Conversion Status

❌ This PDF could not be converted successfully using any of the available methods:
- Azure Document AI (if configured)
- pymupdf4llm
- PyMuPDF (fitz) enhanced
- pdfplumber
- pdfminer

## Possible Issues

- Scanned document requiring OCR (Azure Document AI is best for this)
- Complex formatting or layout
- Corrupted or encrypted PDF
- Unsupported PDF features

## Manual Review Required

This file requires manual review and possible alternative processing methods.
"""

    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final conversion report"""
        stats = self.conversion_stats
        
        logger.info(f"\n🎯 FINAL CONVERSION RESULTS")
        logger.info("=" * 60)
        logger.info(f"✅ Total converted: {stats['successful_conversions']}/{stats['total_files']} files")
        logger.info(f"🖼️ Images extracted: {stats['images_extracted']}")
        logger.info(f"📁 Output location: {self.base_output}")
        logger.info(f"🖼️ Images location: {self.image_output}")
        
        if stats['total_files'] > 0:
            success_rate = (stats['successful_conversions'] / stats['total_files']) * 100
            logger.info(f"📈 Success rate: {success_rate:.1f}%")
        
        self._create_conversion_index()
        
        return {
            "success": True,
            "statistics": stats,
            "output_directory": str(self.base_output),
            "images_directory": str(self.image_output)
        }

    def _create_conversion_index(self):
        """Create comprehensive index of converted files"""
        readme_path = self.base_output / "README.md"
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("# Enhanced PDF to Markdown Conversion Results\n\n")
            f.write(f"**Conversion Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            stats = self.conversion_stats
            f.write("## Conversion Statistics\n\n")
            f.write(f"- **Total Files:** {stats['total_files']}\n")
            f.write(f"- **Successful:** {stats['successful_conversions']}\n")
            f.write(f"- **Failed:** {stats['failed_conversions']}\n")
            f.write(f"- **Empty Results:** {stats['empty_conversions']}\n")
            f.write(f"- **Images Extracted:** {stats['images_extracted']}\n\n")
            for category in ["kelvin_papers", "lectures"]: # Hardcoded categories, ensure consistency
                category_dir = self.base_output / category
                if category_dir.exists():
                    f.write(f"## {category.replace('_', ' ').title()}\n\n")
                    md_files = sorted(category_dir.glob("*.md"))
                    for md_file in md_files:
                        title = md_file.stem.replace('_', ' ').title()
                        f.write(f"- [{title}]({category}/{md_file.name})\n")
                    f.write("\n")
            if self.image_output.exists() and any(self.image_output.iterdir()):
                f.write("## Extracted Images\n\n")
                f.write(f"Images are stored in the `images/` directory, organized by source category.\n\n")
        logger.info(f"📋 Conversion index created: {readme_path}")

# Export the functions for external use
def convert_all_pdfs_enhanced_from_local():
    """Main function to run enhanced PDF conversion from local files."""
    converter = EnhancedPDFConverter()
    return converter.convert_all_pdfs_from_local()

def convert_pdfs_from_vercel_blobs(blob_files: List[Dict]):
    """Main function to run enhanced PDF conversion from Vercel Blob URLs."""
    converter = EnhancedPDFConverter()
    return converter.convert_blobs_to_markdown(blob_files)

if __name__ == "__main__":
    # Example usage for local conversion (for development/testing)
    print("This script is designed to be imported and called programmatically.")
    print("For local PDF conversion: call convert_all_pdfs_enhanced_from_local()")
    print("For Vercel Blob conversion: call convert_pdfs_from_vercel_blobs(blob_file_list)")
    
    # For local testing of blob conversion with dummy URLs (requires a local HTTP server)
    # import http.server
    # import socketserver
    # import threading
    
    # PORT = 8000
    # DIRECTORY = "data/input/lectures" # Or "data/input/kelvin_papers"
    
    # class Handler(http.server.SimpleHTTPRequestHandler):
    #     def __init__(self, *args, **kwargs):
    #         super().__init__(*args, directory=DIRECTORY, **kwargs)
            
    # logger.info(f"Serving files from {DIRECTORY} on port {PORT}")
    # with socketserver.TCPServer(("", PORT), Handler) as httpd:
    #     server_thread = threading.Thread(target=httpd.serve_forever)
    #     server_thread.daemon = True # Allow main program to exit even if server is running
    #     server_thread.start()
        
    #     # Example dummy blob files pointing to the local server
    #     dummy_blob_files = [
    #         {'url': f'http://localhost:{PORT}/1 Introduction to AI.pdf', 'category': 'lectures', 'original_filename': '1 Introduction to AI.pdf'},
    #         # {'url': f'http://localhost:{PORT}/exam_paper_set1.pdf', 'category': 'kelvin_papers', 'original_filename': 'exam_paper_set1.pdf'},
    #     ]
    #     print("\nRunning dummy blob conversion (requires local PDF server):")
    #     convert_pdfs_from_vercel_blobs(dummy_blob_files)
        
    #     httpd.shutdown() # Shutdown the dummy server

    print("\nRunning local PDF conversion (using data/input):")
    convert_all_pdfs_enhanced_from_local()