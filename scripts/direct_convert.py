#!/usr/bin/env python3
"""
Enhanced PDF to Markdown conversion with multiple approaches and image extraction.
Handles complex PDFs, extracts images, and provides fallback conversion methods.
This version is adapted to fetch PDF files from Vercel Blob Storage URLs
and integrates Azure Document AI for superior OCR.
"""

import sys
from pathlib import Path

# IMPORTANT: Load environment variables at the very beginning of the script
from dotenv import load_dotenv
load_dotenv()

# Add project root to Python's sys.path to resolve internal module imports
# This ensures that imports like 'src.core.external_services.azure_document_ai_client' work correctly
project_root_dir = Path(__file__).parent.parent
if str(project_root_dir) not in sys.path:
    sys.path.insert(0, str(project_root_dir))

import pymupdf4llm
import fitz  # PyMuPDF
import os
import shutil
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import hashlib
import base64
from datetime import datetime
import requests # For downloading files from URLs
import tempfile # For creating temporary files

# Import Vercel Blob SDK's list function
from vercel_blob import put, list as list_blobs

# Import Azure Document AI client
from src.core.external_services.azure_document_ai_client import AzureDocumentAIClient
# Import Azure settings
from config.settings import AZURE_DOCUMENT_AI_ENDPOINT, AZURE_DOCUMENT_AI_KEY, BLOB_READ_WRITE_TOKEN, VERCEL_BLOB_BASE_URL # Import new Vercel settings


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
        # This is where the check for Azure credentials happens.
        # If AZURE_DOCUMENT_AI_ENDPOINT and AZURE_DOCUMENT_AI_KEY are set,
        # self.azure_client will be initialized, and thus Azure Document AI will be prioritized.
        if AZURE_DOCUMENT_AI_ENDPOINT and AZURE_DOCUMENT_AI_KEY:
            try:
                self.azure_client = AzureDocumentAIClient()
                logger.info("Azure Document AI client initialized successfully.")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize Azure Document AI client: {e}. Azure OCR will not be used.")
        else:
            logger.info("Azure Document AI credentials not provided. Azure OCR will not be used.")
        
        logger.info("✅ Enhanced PDF Converter initialized")

    async def fetch_all_pdfs_from_vercel_blob(self) -> List[Dict]:
        """Fetches all PDF files from Vercel Blob storage. (NEW METHOD)"""
        logger.info("🚀 Attempting to list all PDF files from Vercel Blob storage...")
        
        # This check is essential: Ensure VERCEL_BLOB_READ_WRITE_TOKEN is set
        if not BLOB_READ_WRITE_TOKEN:
            logger.error("❌ VERCEL_BLOB_READ_WRITE_TOKEN is not set. Cannot list files from Vercel Blob.")
            return []

        all_blob_files = []
        try:
            # REMOVED THE WHILE LOOP AND CURSOR/LIMIT ARGUMENTS
            # Because the error "list() got an unexpected keyword argument 'cursor'"
            # indicates these arguments are not supported by your installed vercel_blob library.
            # This call will retrieve the first page of results (up to the default limit of the Vercel Blob API).
            list_response = list_blobs() 
            
            if 'blobs' not in list_response:
                raise ValueError("Unexpected response from Vercel Blob list: 'blobs' key missing.")
                
            for blob in list_response['blobs']:
                # Check if the blob is a PDF by its URL or pathname
                if blob['pathname'].lower().endswith('.pdf'):
                    # Infer category based on typical naming conventions or folder structure
                    category = "unknown"
                    if "lectures" in blob['pathname'].lower():
                        category = "lectures"
                    elif "kelvin_papers" in blob['pathname'].lower() or "exam_papers" in blob['pathname'].lower():
                        category = "kelvin_papers" # Or "exam_papers"
                    
                    all_blob_files.append({
                        'url': blob['url'],
                        'category': category,
                        'original_filename': Path(blob['pathname']).name
                    })
            
            # Since pagination arguments are not supported, we assume this is the complete list for now.
            logger.info(f"✅ Found {len(all_blob_files)} PDF files in Vercel Blob storage (from single API call).")
            # If `hasMore` is still returned in `list_response` even without `cursor` argument, it means
            # there are more blobs, but we cannot retrieve them with the current library version.
            if list_response.get('hasMore'):
                logger.warning("⚠️ Note: The installed 'vercel-blob-py' library version does not support pagination. Only the first batch of blobs could be retrieved.")

            return all_blob_files
        except Exception as e:
            logger.error(f"❌ Failed to list files from Vercel Blob storage: {e}")
            return []

    async def convert_blobs_to_markdown(self, blob_files: List[Dict]) -> Dict[str, Any]:
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
                
                success = await self._download_and_convert_single_pdf_blob( # Await this call
                    pdf_url, original_filename, output_category_dir, category_image_dir, category_name
                )
                if success:
                    self.conversion_stats["successful_conversions"] += 1
                else:
                    self.conversion_stats["failed_conversions"] += 1

        return self._generate_final_report()

    async def _download_and_convert_single_pdf_blob(self, pdf_url: str, original_filename: str,
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
            
            # The _convert_pdf_from_path might also need to be awaited if _convert_with_azure_document_ai is awaited
            success = await self._convert_pdf_from_path( 
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

    async def _convert_pdf_from_path(self, pdf_file_path: Path, original_filename: str, output_dir: Path, 
                               image_dir: Path, category: str) -> bool:
        """Internal method to convert a single PDF from a local path (could be temp or actual input)"""
        logger.info(f" 📄 Converting: {original_filename} (from {pdf_file_path.name})...")
        
        clean_name = Path(original_filename).stem.replace(' ', '_').replace('-', '_')
        output_file = output_dir / f"{clean_name}.md"
        

        # Initialize best results with empty content and 0 images
        best_content = ""
        best_method = ""
        total_images_extracted = 0

        # Try Azure Document AI first if configured
        if self.azure_client:
            logger.info(f"  🌟 Attempting conversion with Azure Document AI for {original_filename}.")
            try:
                # Await the async Azure Document AI call
                azure_content, azure_images = await self._convert_with_azure_document_ai(pdf_file_path, image_dir, clean_name)
                if self._is_good_conversion(azure_content):
                    best_content = azure_content
                    best_method = "azure_document_ai"
                    total_images_extracted = azure_images
                    logger.info(f"  ✅ Azure Document AI successful for {original_filename}.")
                else:
                    logger.warning(f"  ⚠️ Azure Document AI produced poor quality or empty content for {original_filename}.")
            except Exception as e:
                logger.warning(f"  ❌ Azure Document AI failed for {original_filename}: {e}")

        # Always try pymupdf4llm, either as primary or as fallback
        logger.info(f"  ⚙️ Attempting conversion with pymupdf4llm for {original_filename}.")
        try:
            # pymupdf4llm is synchronous, so no await here.
            pymupdf_content, pymupdf_images = self._convert_with_pymupdf4llm(pdf_file_path, image_dir, clean_name)
            if self._is_good_conversion(pymupdf_content):
                # If pymupdf_content is better (longer) than current best_content, or if azure_content was poor
                if len(pymupdf_content) > len(best_content):
                    best_content = pymupdf_content
                    best_method = "pymupdf4llm"
                    total_images_extracted = pymupdf_images
                    logger.info(f"  ✅ PyMuPDF4LLM successful and selected as best for {original_filename}.")
                else:
                    logger.info(f"  ✅ PyMuPDF4LLM successful for {original_filename}, but Azure was better or equally good.")
            else:
                logger.warning(f"  ⚠️ PyMuPDF4LLM produced poor quality or empty content for {original_filename}.")
        except Exception as e:
            logger.warning(f"  ❌ PyMuPDF4LLM failed for {original_filename}: {e}")

        # Final decision based on the best content found
        if best_content and self._is_good_conversion(best_content): # Re-check best_content with strict criteria
            enhanced_content = self._enhance_markdown_content(
                best_content, Path(original_filename), best_method, total_images_extracted
            )
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(enhanced_content)
                
            self.conversion_stats["images_extracted"] += total_images_extracted
            if best_method == "pymupdf4llm" and self.azure_client and self.conversion_stats["fallback_used"] == 0: # Only count fallback if Azure was available and wasn't used
                 self.conversion_stats["fallback_used"] += 1
            logger.info(f"  ✅ Saved: {output_file.name} ({best_method}, {total_images_extracted} images)")
            return True
        else:
            # This branch is hit if no method produced good content, or all failed the quality check.
            placeholder_content = self._create_error_placeholder(Path(original_filename),
                                                                f"Conversion failed or produced empty/poor quality content using {best_method if best_method else 'available methods'}.")
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
            # For local conversion, this is sync, so it processes one by one
            # The async _convert_pdf_from_path will be called within this, but the loop itself is sync
            self._process_category(category_name, paths)

        return self._generate_final_report()

    def _process_category(self, category_name: str, paths: Dict[str, Path]):
        """Process a category of PDF files from local input directory (synchronously)"""
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
            # Need to run _convert_pdf_from_path with asyncio.run() or similar,
            # as it's an async function being called from a sync context.
            # This is generally not recommended in a tight loop, but for a one-off
            # local script, it's simpler than re-architecting the whole `_process_category` to be async.
            import asyncio
            success = asyncio.run(self._convert_pdf_from_path(
                pdf_file, pdf_file.name, paths['output'], category_image_dir, category_name
            ))
            if success:
                self.conversion_stats["successful_conversions"] += 1
            else:
                self.conversion_stats["failed_conversions"] += 1

    # NEW: Method for Azure Document AI conversion
    async def _convert_with_azure_document_ai(self, pdf_file: Path, image_dir: Path, 
                                        clean_name: str) -> Tuple[str, int]:
        """Convert using Azure Document AI for advanced OCR and layout extraction."""
        if not self.azure_client:
            raise RuntimeError("Azure Document AI client not initialized.")
            
        pdf_bytes = pdf_file.read_bytes()
        
        # Await the async call to analyze_pdf_content
        azure_result = await self.azure_client.analyze_pdf_content(pdf_bytes)
        
        if not azure_result or not azure_result.get("full_text"):
            # If Azure returns no full text, it's considered a failure for this method.
            raise ValueError("Azure Document AI returned empty or invalid result.")
            
        extracted_content = azure_result["full_text"]
        
        if azure_result.get("tables"):
            table_markdown = []
            # 'tables' in azure_result is a list of Markdown table strings generated by AzureDocumentAIClient
            for i, table_md in enumerate(azure_result["tables"]):
                if table_md.strip(): # Ensure the table markdown is not empty
                    table_markdown.append(f"\n\n**Extracted Table {i+1}:**\n")
                    table_markdown.append(table_md)
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
        # This method is specifically for converting a list of lists (like from pdfplumber)
        # into a Markdown table.
        # The AzureDocumentAIClient now returns tables already formatted as Markdown strings,
        # so this method might not be called directly from Azure conversion if table extraction is complete.
        # It's kept for potential future use or if Azure gives raw table data.
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
        """
        Checks if conversion result is of good quality.
        This is a heuristic. For truly critical scenarios, manual review or more
        advanced content analysis might be needed.
        """
        # A very minimal amount of content is considered "not good"
        if not content or len(content.strip()) < 500: # Increased minimum length for 'good'
            return False
            
        # A very low word count might indicate poor extraction
        word_count = len(content.split())
        if word_count < 200: # Increased minimum word count
            return False
            
        # These checks might filter out some valid but unusual content (e.g., highly symbolic PDFs)
        # For general text, they help identify garbled output.
        # However, they are now more lenient if tables are present.
        problematic_chars = "._-" 
        for char in problematic_chars:
            # If a character like '.' or '_' appears excessively, it might indicate binary data or corrupted text.
            # Exception for tables, where '---' and '|' are expected.
            if content.count(char) > len(content) * 0.1 and not ("| ---" in content and "|" in content):
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
        """Create error placeholder content for failed conversions"""
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
            for category in ["kelvin_papers", "lectures", "unknown"]: # Added "unknown" for robustness
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

# Make this an async function as it will now call async methods
async def convert_pdfs_from_vercel_blobs(blob_files: Optional[List[Dict]] = None):
    """Main function to run enhanced PDF conversion from Vercel Blob URLs.
       If blob_files is None, it will attempt to fetch all PDFs from Vercel Blob.
    """
    converter = EnhancedPDFConverter()
    if blob_files is None:
        blob_files_to_convert = await converter.fetch_all_pdfs_from_vercel_blob()
    else:
        blob_files_to_convert = blob_files

    if not blob_files_to_convert:
        logger.warning("No PDF files found or provided for Vercel Blob conversion.")
        return converter._generate_final_report() # Return an empty report

    return await converter.convert_blobs_to_markdown(blob_files_to_convert)

if __name__ == "__main__":
    import asyncio # Import asyncio for running async functions

    # --- Option 1: Convert ALL PDFs from Vercel Blob Storage ---
    # This will automatically list all PDF files in your Vercel Blob storage
    # and attempt to convert them.
    print("\nAttempting to convert ALL PDFs from Vercel Blob storage:")
    # Pass None to trigger automatic fetching of all PDFs
    asyncio.run(convert_pdfs_from_vercel_blobs(None))

    # --- Option 2: Convert SPECIFIC PDFs from Vercel Blob Storage (as before) ---
    # Uncomment and use this block if you want to convert only specific URLs.
    # Replace these dummy URLs with your actual Vercel Blob URLs.
    # Example blob files pointing to Vercel storage.
    # Ensure the 'category' matches your desired output subdirectory (e.g., 'lectures', 'kelvin_papers').
    # vercel_blob_files = [
    #     {
    #         'url': 'https://88avsgpdqmsyih7d.public.blob.vercel-storage.com/1755497259449-3_Introduction_to_Data_Science.pdf',
    #         'category': 'lectures',
    #         'original_filename': '3_Introduction_to_Data_Science.pdf'
    #     },
    #     {
    #         'url': 'https://88avsgpdqmsyih7d.public.blob.vercel-storage.com/1755497259641-4_Statistical_Analysis_for_Dat-Analytics.pdf',
    #         'category': 'lectures',
    #         'original_filename': '4_Statistical_Analysis_for_Dat-Analytics.pdf'
    #     },
    # ]
    # print("\nAttempting SPECIFIC Vercel Blob PDF conversion:")
    # asyncio.run(convert_pdfs_from_vercel_blobs(vercel_blob_files))

    # --- Option 3: Convert local PDFs (using data/input) ---
    # Uncomment the following line if you also need to convert local PDFs.
    # print("\nRunning LOCAL PDF conversion (using data/input):")
    # convert_all_pdfs_enhanced_from_local()