"""
Text loading and preprocessing for manually input text files.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import re
from loguru import logger

@dataclass
class TextDocument:
    content: str
    metadata: Dict[str, Any]
    source_file: str
    paper_set: str
    paper_number: str

class TextLoader:
    def __init__(self):
        self.documents: List[TextDocument] = []
    
    def load_text_file(self, text_file: Path) -> str:
        """Load content from text file"""
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Loaded text file: {text_file}")
            return content
        except Exception as e:
            logger.error(f"Failed to load {text_file}: {e}")
            return ""
    
    def load_metadata_file(self, metadata_file: Path) -> Dict[str, Any]:
        """Load metadata from JSON file"""
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            logger.warning(f"No metadata file found for {metadata_file}: {e}")
            return {}
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep educational content
        text = re.sub(r'[^\w\s\.\,\?\!\:\;\(\)\-\+\=\%\$]', '', text)
        # Normalize line breaks
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def process_directory(self, input_dir: Path) -> List[TextDocument]:
        """Process all text files in directory structure"""
        logger.info(f"Processing directory: {input_dir}")
        for set_dir in input_dir.iterdir():
            if not set_dir.is_dir():
                continue
            logger.info(f"Processing paper set: {set_dir.name}")
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
                        paper_number=text_file.stem.split('_')[1]
                    )
                    self.documents.append(doc)
                    logger.info(f"Processed document: {text_file.name}")
        logger.info(f"Total documents processed: {len(self.documents)}")
        return self.documents
    
    def save_processed_documents(self, output_file: Path):
        """Save processed documents for later use"""
        output_data = []
        for doc in self.documents:
            output_data.append({
                'content': doc.content,
                'metadata': doc.metadata,
                'source_file': doc.source_file,
                'paper_set': doc.paper_set,
                'paper_number': doc.paper_number
            })
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(output_data)} processed documents to {output_file}") 