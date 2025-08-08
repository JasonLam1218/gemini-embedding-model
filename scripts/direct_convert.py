#!/usr/bin/env python3
"""
Enhanced PDF to Markdown conversion with multiple approaches and image extraction.
Handles complex PDFs, extracts images, and provides fallback conversion methods.
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
        self.base_input = Path("data/input")
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
        
        logger.info("✅ Enhanced PDF Converter initialized")

    def convert_all_pdfs(self) -> Dict[str, Any]:
        """Convert all PDFs with enhanced processing and image extraction"""
        logger.info("🚀 Starting Enhanced PDF to Markdown Conversion")
        logger.info("=" * 60)

        categories = {
            "Kelvin Papers": {
                "input": self.base_input / "kelvin_papers",
                "output": self.base_output / "kelvin_papers"
            },
            "Lectures": {
                "input": self.base_input / "lectures",
                "output": self.base_output / "lectures"
            }
        }

        for category_name, paths in categories.items():
            logger.info(f"\n📚 Processing {category_name}...")
            self._process_category(category_name, paths)

        return self._generate_final_report()

    def _process_category(self, category_name: str, paths: Dict[str, Path]):
        """Process a category of PDF files"""
        logger.info(f"Input: {paths['input']}")
        logger.info(f"Output: {paths['output']}")
        
        # Create output directory
        paths['output'].mkdir(parents=True, exist_ok=True)
        
        # Create category-specific image directory
        category_image_dir = self.image_output / category_name.lower().replace(" ", "_")
        category_image_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(paths['input'].glob("*.pdf"))
        self.conversion_stats["total_files"] += len(pdf_files)
        
        if not pdf_files:
            logger.warning(f"⚠️ No PDF files found in {paths['input']}")
            return
            
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            success = self._convert_single_pdf(
                pdf_file, 
                paths['output'], 
                category_image_dir,
                category_name
            )
            if success:
                self.conversion_stats["successful_conversions"] += 1
            else:
                self.conversion_stats["failed_conversions"] += 1

    def _convert_single_pdf(self, pdf_file: Path, output_dir: Path, 
                           image_dir: Path, category: str) -> bool:
        """Convert a single PDF using multiple approaches"""
        logger.info(f" 📄 Converting: {pdf_file.name}...")
        
        clean_name = pdf_file.stem.replace(' ', '_').replace('-', '_')
        output_file = output_dir / f"{clean_name}.md"
        
        # Try multiple conversion approaches
        conversion_methods = [
            ("pymupdf4llm", self._convert_with_pymupdf4llm),
            ("fitz_enhanced", self._convert_with_fitz_enhanced),
            ("pdfplumber", self._convert_with_pdfplumber),
            ("pdfminer", self._convert_with_pdfminer)
        ]
        
        images_extracted = 0
        best_content = ""
        best_method = ""
        
        for method_name, method_func in conversion_methods:
            try:
                content, method_images = method_func(pdf_file, image_dir, clean_name)
                
                if self._is_good_conversion(content):
                    best_content = content
                    best_method = method_name
                    images_extracted += method_images
                    logger.info(f"  ✅ Success with {method_name}")
                    break
                else:
                    logger.warning(f"  ⚠️ Poor quality with {method_name}")
                    
            except Exception as e:
                logger.warning(f"  ❌ {method_name} failed: {e}")
                continue
        
        # Save the best conversion
        if best_content and len(best_content.strip()) > 100:
            enhanced_content = self._enhance_markdown_content(
                best_content, pdf_file, best_method, images_extracted
            )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(enhanced_content)
                
            self.conversion_stats["images_extracted"] += images_extracted
            logger.info(f"  ✅ Saved: {output_file.name} ({best_method}, {images_extracted} images)")
            return True
        else:
            # Create placeholder with error info
            placeholder_content = self._create_error_placeholder(pdf_file, "All conversion methods failed")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(placeholder_content)
            
            self.conversion_stats["empty_conversions"] += 1
            logger.error(f"  ❌ Failed: {pdf_file.name} - all methods failed")
            return False

    def _convert_with_pymupdf4llm(self, pdf_file: Path, image_dir: Path, 
                                 clean_name: str) -> Tuple[str, int]:
        """Convert using pymupdf4llm with image extraction"""
        # Extract images first using fitz
        images_extracted = self._extract_images_with_fitz(pdf_file, image_dir, clean_name)
        
        # Convert to markdown
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
                
                # Extract text with formatting
                text_dict = page.get_text("dict")
                page_content = self._process_fitz_text_dict(text_dict)
                
                if page_content.strip():
                    content_parts.append(f"\n\n---\n**Page {page_num + 1}**\n\n{page_content}")
                
                # Extract images from this page
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
                # Extract text
                text = page.extract_text()
                
                # Extract tables
                tables = page.extract_tables()
                
                page_content = f"\n\n---\n**Page {page_num + 1}**\n\n"
                
                if text:
                    page_content += text + "\n\n"
                
                # Add tables as markdown
                for table_num, table in enumerate(tables):
                    if table:
                        page_content += f"\n**Table {table_num + 1}:**\n\n"
                        page_content += self._table_to_markdown(table) + "\n\n"
        
                content_parts.append(page_content)
        
        # Extract images using fitz (pdfplumber doesn't handle images well)
        images_extracted = self._extract_images_with_fitz(pdf_file, image_dir, clean_name)
        
        return "\n".join(content_parts), images_extracted

    def _convert_with_pdfminer(self, pdf_file: Path, image_dir: Path, 
                              clean_name: str) -> Tuple[str, int]:
        """Convert using pdfminer for text extraction"""
        if not PDFMINER_AVAILABLE:
            raise ImportError("pdfminer not available")
            
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams
        
        # Configure layout analysis
        laparams = LAParams(
            char_margin=2.0,
            line_margin=0.5,
            word_margin=0.1,
            boxes_flow=0.5,
            detect_vertical=True
        )
        
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
                # Get image data
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                # Skip images that are too small (likely decorative)
                if pix.width < 50 or pix.height < 50:
                    pix = None
                    continue
                
                # Generate filename
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
            
        # Check for reasonable text density
        word_count = len(content.split())
        if word_count < 50:
            return False
            
        # Check for excessive repeated characters (indicates OCR issues)
        for char in "._-|":
            if content.count(char) > len(content) * 0.1:
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
        
        # Add image references if images were extracted
        if images_count > 0:
            clean_name = pdf_file.stem.replace(' ', '_').replace('-', '_')
            image_section = f"\n\n## Extracted Images\n\n"
            for i in range(images_count):
                image_section += f"![Image {i+1}](images/{clean_name}_page*_img{i+1}.png)\n\n"
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
- pymupdf4llm
- PyMuPDF (fitz) enhanced
- pdfplumber
- pdfminer

## Possible Issues

- Scanned document requiring OCR
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
        
        if stats['successful_conversions'] > 0:
            success_rate = (stats['successful_conversions'] / stats['total_files']) * 100
            logger.info(f"📈 Success rate: {success_rate:.1f}%")
        
        # Create conversion report
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
            
            # Statistics
            stats = self.conversion_stats
            f.write("## Conversion Statistics\n\n")
            f.write(f"- **Total Files:** {stats['total_files']}\n")
            f.write(f"- **Successful:** {stats['successful_conversions']}\n")
            f.write(f"- **Failed:** {stats['failed_conversions']}\n")
            f.write(f"- **Empty Results:** {stats['empty_conversions']}\n")
            f.write(f"- **Images Extracted:** {stats['images_extracted']}\n\n")
            
            # List converted files by category
            for category in ["kelvin_papers", "lectures"]:
                category_dir = self.base_output / category
                if category_dir.exists():
                    f.write(f"## {category.replace('_', ' ').title()}\n\n")
                    md_files = sorted(category_dir.glob("*.md"))
                    for md_file in md_files:
                        title = md_file.stem.replace('_', ' ').title()
                        f.write(f"- [{title}]({category}/{md_file.name})\n")
                    f.write("\n")
            
            # Images directory
            if self.image_output.exists() and any(self.image_output.iterdir()):
                f.write("## Extracted Images\n\n")
                f.write(f"Images are stored in the `images/` directory, organized by source category.\n\n")
        
        logger.info(f"📋 Conversion index created: {readme_path}")

def convert_all_pdfs_enhanced():
    """Main function to run enhanced PDF conversion"""
    converter = EnhancedPDFConverter()
    return converter.convert_all_pdfs()

if __name__ == "__main__":
    convert_all_pdfs_enhanced()
