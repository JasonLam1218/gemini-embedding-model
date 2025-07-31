"""
Text loading and preprocessing for markdown files converted from PDFs.
Enhanced to handle both lecture notes and exam papers with content classification.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from loguru import logger

@dataclass
class TextDocument:
    content: str
    metadata: Dict[str, Any]
    source_file: str
    paper_set: str
    paper_number: str
    content_type: str  # New field for content classification

class TextLoader:
    def __init__(self):
        self.documents: List[TextDocument] = []
        
    def load_markdown_file(self, md_file: Path) -> str:
        """Load content from markdown file"""
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"✅ Loaded markdown file: {md_file.name}")
            return content
        except Exception as e:
            logger.error(f"❌ Failed to load {md_file}: {e}")
            return ""
    
    def classify_content_type(self, file_path: Path) -> str:
        """Classify if content is exam paper, model answers, or lecture notes"""
        file_name = file_path.name.lower()
        parent_dir = file_path.parent.name.lower()
        
        if "kelvin_papers" in parent_dir:
            if "ms" in file_name or "model" in file_name:
                return "model_answers"
            elif "exam" in file_name or "paper" in file_name:
                return "exam_questions"
            else:
                return "sample_paper"
        elif "lectures" in parent_dir:
            return "lecture_notes"
        else:
            return "unknown"
    
    def extract_markdown_metadata(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from markdown content"""
        metadata = {
            "file_type": "markdown",
            "source_type": self.classify_content_type(file_path),
            "file_name": file_path.name,
            "parent_directory": file_path.parent.name,
            "content_length": len(content),
            "estimated_pages": len(content) // 2000,  # Rough estimate
        }
        
        # Extract title from first heading if available
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            metadata["extracted_title"] = title_match.group(1).strip()
        
        # Count sections/headings
        headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        metadata["section_count"] = len(headings)
        metadata["headings"] = headings[:5]  # Store first 5 headings
        
        return metadata
    
    def clean_markdown_text(self, text: str) -> str:
        """Clean and normalize markdown text content while preserving structure"""
        # Remove excessive whitespace but preserve paragraph structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Clean up markdown artifacts that might interfere with processing
        # Remove markdown table formatting that's broken
        text = re.sub(r'\|[\s\-\|]*\|', '', text)
        
        # Normalize bullet points
        text = re.sub(r'^\s*[\*\-\+]\s+', '• ', text, flags=re.MULTILINE)
        
        # Clean up excessive spaces
        text = re.sub(r' +', ' ', text)
        
        # Preserve important punctuation for academic content
        # Don't remove mathematical symbols, percentages, etc.
        
        return text.strip()
    
    def process_markdown_directory(self, input_dir: Path) -> List[TextDocument]:
        """Process all markdown files in directory structure"""
        logger.info(f"📂 Processing markdown directory: {input_dir}")
        
        if not input_dir.exists():
            logger.error(f"❌ Directory does not exist: {input_dir}")
            return []
        
        total_files = 0
        processed_files = 0
        
        # Process all .md files recursively
        for md_file in input_dir.rglob("*.md"):
            if md_file.name == "README.md":
                continue  # Skip README files
                
            total_files += 1
            logger.info(f"🔄 Processing: {md_file.relative_to(input_dir)}")
            
            # Load markdown content
            content = self.load_markdown_file(md_file)
            if not content:
                logger.warning(f"⚠️ Empty content in: {md_file.name}")
                continue
            
            # Clean content
            cleaned_content = self.clean_markdown_text(content)
            if len(cleaned_content) < 100:  # Skip very short files
                logger.warning(f"⚠️ Content too short in: {md_file.name}")
                continue
            
            # Extract metadata
            metadata = self.extract_markdown_metadata(content, md_file)
            
            # Determine paper set and number from path
            relative_path = md_file.relative_to(input_dir)
            paper_set = relative_path.parts[0] if len(relative_path.parts) > 1 else "general"
            paper_number = md_file.stem
            
            # Create document
            doc = TextDocument(
                content=cleaned_content,
                metadata=metadata,
                source_file=str(md_file),
                paper_set=paper_set,
                paper_number=paper_number,
                content_type=metadata["source_type"]
            )
            
            self.documents.append(doc)
            processed_files += 1
            
            logger.info(f"✅ Processed: {md_file.name} ({metadata['source_type']}, {len(cleaned_content)} chars)")
        
        logger.info(f"📊 Processing complete: {processed_files}/{total_files} files processed")
        logger.info(f"📋 Content types found: {set(doc.content_type for doc in self.documents)}")
        
        return self.documents
    
    def process_directory(self, input_dir: Path) -> List[TextDocument]:
        """Main processing method - routes to appropriate handler"""
        if not input_dir.exists():
            logger.error(f"❌ Input directory does not exist: {input_dir}")
            return []
        
        # Check if this is a markdown directory
        md_files = list(input_dir.rglob("*.md"))
        txt_files = list(input_dir.rglob("*.txt"))
        
        if md_files and not txt_files:
            logger.info("📄 Detected markdown files - using markdown processing")
            return self.process_markdown_directory(input_dir)
        elif txt_files and not md_files:
            logger.info("📄 Detected text files - using legacy text processing")
            return self.process_text_directory(input_dir)
        elif md_files and txt_files:
            logger.info("📄 Found both markdown and text files - prioritizing markdown")
            return self.process_markdown_directory(input_dir)
        else:
            logger.error("❌ No .md or .txt files found in directory")
            return []
    
    def process_text_directory(self, input_dir: Path) -> List[TextDocument]:
        """Legacy method for processing .txt files"""
        logger.info(f"📂 Processing text directory: {input_dir}")
        
        for set_dir in input_dir.iterdir():
            if not set_dir.is_dir():
                continue
                
            logger.info(f"📁 Processing paper set: {set_dir.name}")
            
            for text_file in set_dir.glob("*_text.txt"):
                metadata_file = text_file.with_name(
                    text_file.name.replace("_text.txt", "_metadata.json")
                )
                
                content = self.load_text_file(text_file)
                metadata = self.load_metadata_file(metadata_file)
                
                if content:
                    cleaned_content = self.clean_text(content)
                    doc = TextDocument(
                        content=cleaned_content,
                        metadata=metadata,
                        source_file=str(text_file),
                        paper_set=set_dir.name,
                        paper_number=text_file.stem.split('_')[1],
                        content_type="legacy_text"
                    )
                    
                    self.documents.append(doc)
                    logger.info(f"✅ Processed document: {text_file.name}")
        
        return self.documents
    
    def load_text_file(self, text_file: Path) -> str:
        """Load content from text file (legacy support)"""
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"✅ Loaded text file: {text_file}")
            return content
        except Exception as e:
            logger.error(f"❌ Failed to load {text_file}: {e}")
            return ""
    
    def load_metadata_file(self, metadata_file: Path) -> Dict[str, Any]:
        """Load metadata from JSON file (legacy support)"""
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            logger.warning(f"⚠️ No metadata file found for {metadata_file}: {e}")
            return {}
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content (legacy method)"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\?\!\:\;\(\)\-\+\=\%\$]', '', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def get_content_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed content"""
        if not self.documents:
            return {"total_documents": 0}
        
        content_types = {}
        paper_sets = {}
        total_chars = 0
        
        for doc in self.documents:
            # Count by content type
            content_type = doc.content_type
            if content_type not in content_types:
                content_types[content_type] = 0
            content_types[content_type] += 1
            
            # Count by paper set
            paper_set = doc.paper_set
            if paper_set not in paper_sets:
                paper_sets[paper_set] = 0
            paper_sets[paper_set] += 1
            
            # Total characters
            total_chars += len(doc.content)
        
        return {
            "total_documents": len(self.documents),
            "content_types": content_types,
            "paper_sets": paper_sets,
            "total_characters": total_chars,
            "average_document_size": total_chars // len(self.documents) if self.documents else 0
        }
    
    def save_processed_documents(self, output_file: Path):
        """Save processed documents for later use"""
        output_data = []
        for doc in self.documents:
            output_data.append({
                'content': doc.content,
                'metadata': doc.metadata,
                'source_file': doc.source_file,
                'paper_set': doc.paper_set,
                'paper_number': doc.paper_number,
                'content_type': doc.content_type
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Log statistics
        stats = self.get_content_statistics()
        logger.info(f"💾 Saved {len(output_data)} processed documents to {output_file}")
        logger.info(f"📊 Statistics: {stats}")
