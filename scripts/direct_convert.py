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

# Import Vercel Blob SDK's put function
from vercel_blob import put, list as list_blobs

# Import Azure Document AI client
from src.core.external_services.azure_document_ai_client import AzureDocumentAIClient
# Correctly import VERCEL_BLOB_READ_WRITE_TOKEN from config.settings
from config.settings import AZURE_DOCUMENT_AI_ENDPOINT, AZURE_DOCUMENT_AI_KEY, BLOB_READ_WRITE_TOKEN, VERCEL_BLOB_BASE_URL


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
        # The VERCEL_BLOB_READ_WRITE_TOKEN is now correctly imported from config.settings
        if not BLOB_READ_WRITE_TOKEN:
            logger.error("❌ VERCEL_BLOB_READ_WRITE_TOKEN is not set. Cannot list files from Vercel Blob.")
            return []

        all_blob_files = []
        try:
            # Removed await, cursor, and limit arguments as per previous debugging steps.
            # This call will retrieve the first page of results (up to the default limit of the Vercel Blob API).
            list_response = list_blobs() 
            
            if 'blobs' not in list_response:
                raise ValueError("Unexpected response from Vercel Blob list: 'blobs' key missing.")
                
            for blob in list_response['blobs']:
                # Check if the blob is a PDF by its URL or pathname
                if blob['pathname'].lower().endswith('.pdf'):
                    # Infer category based on typical naming conventions or folder structure
                    # We still infer category to pass to _download_and_convert_single_pdf_blob,
                    # but it won't affect the final output directory.
                    category = "unknown"
                    if "lectures" in blob['pathname'].lower():
                        category = "lectures"
                    elif "kelvin_papers" in blob['pathname'].lower() or "exam_papers" in blob['pathname'].lower():
                        category = "kelvin_papers" # Or "exam_papers"
                    
                    all_blob_files.append({
                        'url': blob['url'],
                        'category': category, # Keep category in data for _enhance_markdown_content metadata if needed, even if not used for paths
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

        # No need to group by category for output directories anymore
        # The output files will all go directly into self.base_output
        self.conversion_stats["total_files"] += len(blob_files)

        for file_info in blob_files:
            pdf_url = file_info['url']
            original_filename = file_info.get('original_filename', Path(pdf_url).name)
            category = file_info.get('category', 'unknown') # Still pass category for _enhance_markdown_content metadata if relevant
            
            success = await self._download_and_convert_single_pdf_blob( 
                pdf_url, original_filename, 
                self.base_output, # Direct to base output directory
                self.image_output, # Direct to image output directory
                category
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
        output_file = output_dir / f"{clean_name}.md" # Use the provided output_dir directly
        

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
            # Pass original_filename.name to _enhance_markdown_content for consistent naming
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

        categories = { # Still use categories for input organization
            "Kelvin Papers": Path("data/input") / "kelvin_papers",
            "Lectures": Path("data/input") / "lectures"
        }
            
        for category_name, input_dir in categories.items():
            logger.info(f"\n📚 Processing {category_name} from local input: {input_dir}...")
            
            # The output_dir and image_dir will now be the base directories directly
            output_dir_for_category = self.base_output 
            image_dir_for_category = self.image_output
            
            # No need to create sub-directories here, base_output and image_output are already created in __init__
            
            pdf_files = list(input_dir.glob("*.pdf"))
            self.conversion_stats["total_files"] += len(pdf_files)
            
            if not pdf_files:
                logger.warning(f"⚠️ No PDF files found in {input_dir}")
                continue # Use continue to process next category if current is empty
                
            logger.info(f"Found {len(pdf_files)} PDF files")
            
            for pdf_file in pdf_files:
                # Need to run _convert_pdf_from_path with asyncio.run() or similar,
                # as it's an async function being called from a sync context.
                import asyncio
                success = asyncio.run(self._convert_pdf_from_path(
                    pdf_file, pdf_file.name, output_dir_for_category, image_dir_for_category, category_name
                ))
                if success:
                    self.conversion_stats["successful_conversions"] += 1
                else:
                    self.conversion_stats["failed_conversions"] += 1

        return self._generate_final_report()

    async def _convert_with_azure_document_ai(self, pdf_file_path: Path, image_dir: Path, clean_name: str) -> Tuple[str, int]:
        """Placeholder for Azure Document AI conversion."""
        logger.warning(f"  Placeholder: _convert_with_azure_document_ai called for {pdf_file_path.name}")
        # This needs the actual Azure Document AI client logic.
        # For now, return empty content and 0 images as a basic fallback.
        if self.azure_client:
            try:
                # In a real scenario, you'd call:
                # result = await self.azure_client.process_pdf(pdf_file_path)
                # markdown_content = result.get("markdown", "")
                # images_count = self._extract_images_from_azure_result(result, image_dir, clean_name) # Assuming helper
                # For now, a mock:
                markdown_content = f"## Content from Azure Document AI for {clean_name}\n\n[Azure conversion placeholder content]\n"
                # You would add logic here to parse images from Azure's response and save them
                images_count = 0 
                return markdown_content, images_count
            except Exception as e:
                logger.error(f"Error in Azure Document AI conversion placeholder: {e}")
                return "", 0
        return "", 0

    def _convert_with_pymupdf4llm(self, pdf_file_path: Path, image_dir: Path, clean_name: str) -> Tuple[str, int]:
        """Converts PDF to markdown using pymupdf4llm and extracts images using fitz."""
        logger.info(f"  Attempting pymupdf4llm conversion for {pdf_file_path.name}")
        content = ""
        images_extracted = 0
        try:
            content = pymupdf4llm.to_markdown(pdf_file_path)
            
            # Use fitz (PyMuPDF) to extract images separately
            doc = fitz.open(pdf_file_path)
            images_extracted = self._extract_images_with_fitz(doc, image_dir, clean_name)
            doc.close()
            
        except Exception as e:
            logger.error(f"Error during pymupdf4llm conversion for {pdf_file_path.name}: {e}")
            content = ""
            images_extracted = 0
        return content, images_extracted

    def _extract_images_with_fitz(self, doc: fitz.Document, image_dir: Path, file_stem: str) -> int:
        """Extracts images from all pages of a PyMuPDF document."""
        total_images = 0
        logger.debug(f"  Extracting images with Fitz for {file_stem}...")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            total_images += self._extract_page_images_fitz(page, image_dir, file_stem, page_num)
        logger.debug(f"  Total images extracted for {file_stem}: {total_images}")
        return total_images

    def _extract_page_images_fitz(self, page: fitz.Page, image_dir: Path, file_stem: str, page_num: int) -> int:
        """Extracts images from a single page using PyMuPDF (fitz)."""
        images_on_page = 0
        try:
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Generate a unique filename
                image_filename = image_dir / f"{file_stem}_page{page_num+1}_img{img_index+1}.{image_ext}"
                
                with open(image_filename, "wb") as img_file:
                    img_file.write(image_bytes)
                images_on_page += 1
                logger.debug(f"    Saved image: {image_filename.name}")
        except Exception as e:
            logger.warning(f"    Failed to extract images from page {page_num+1} of {file_stem}: {e}")
        return images_on_page

    def _is_good_conversion(self, content: str) -> bool:
        """Determines if the converted content is 'good' (not empty or trivial)."""
        # A simple heuristic: content should have more than 50 characters, excluding whitespace.
        return bool(content.strip()) and len(content.strip()) > 50


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
            # Updated image path to reflect flattened structure
            image_section += f"Images for this document are located in the `{self.image_output.name}/` directory and are named like {clean_name}_pageX_imgY.png\n\n"
            # You could dynamically link if you know where the images will be hosted
            # For example: image_section += f"![Image {i+1}](<vercel-blob-image-url>/{clean_name}_page*_img{i+1}.png)\n\n"

            content += image_section
        
        return header + content

    def _create_error_placeholder(self, pdf_file: Path, error_message: str) -> str:
        """Creates a markdown placeholder for failed conversions."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""# Conversion Failed: {pdf_file.stem.replace('_', ' ').title()}

**Source:** {pdf_file.name}  
**Attempted Conversion Date:** {timestamp}  
**Status:** Failed

---

**Error:** {error_message}

This PDF could not be converted to a meaningful Markdown format.
Please check the original PDF file for issues or try a different conversion tool.
"""

    def _generate_final_report(self) -> Dict[str, Any]:
        """Generates the final report and creates the conversion index."""
        self._create_conversion_index() # Call the method to create README.md
        logger.info("\n📊 Conversion Summary:")
        for key, value in self.conversion_stats.items():
            logger.info(f"- {key.replace('_', ' ').title()}: {value}")
        logger.info("=" * 60)
        logger.info(f"Output available in: {self.base_output.resolve()}")
        logger.info(f"Images available in: {self.image_output.resolve()}")
        return self.conversion_stats


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
            
            # List all .md files directly under self.base_output
            f.write("## Converted Markdown Files\n\n")
            md_files = sorted(self.base_output.glob("*.md"))
            for md_file in md_files:
                # Exclude README.md itself
                if md_file.name.lower() == "readme.md":
                    continue
                title = md_file.stem.replace('_', ' ').title()
                f.write(f"- [{title}]({md_file.name})\n") # Link directly
            f.write("\n")

            if self.image_output.exists() and any(self.image_output.iterdir()):
                f.write("## Extracted Images\n\n")
                # Updated image path description for flattened structure
                f.write(f"Images are stored in the `{self.image_output.name}/` directory.\n\n")
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
    # convert_all_pdfs_enhanced_from_local()(venv) jasonlam@JasondeMacBook-Air gemini-embedding-model % python scripts/direct_convert.py
