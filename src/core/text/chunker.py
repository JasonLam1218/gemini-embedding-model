import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List, Optional
from loguru import logger

class TextChunker:
    def __init__(self, max_chunk_size: int = 1500, overlap: int = 200):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.nlp = None
        
        # Try to load spaCy model with fallback
        self._initialize_nlp()
        
        # Download NLTK data if needed
        self._ensure_nltk_data()
    
    def _initialize_nlp(self):
        """Initialize NLP model with fallback options"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ Using spaCy model for text processing")
        except OSError:
            logger.warning("⚠️ spaCy model 'en_core_web_sm' not found. Installing...")
            try:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("✅ Downloaded and loaded spaCy model")
            except Exception as e:
                logger.warning(f"⚠️ Could not install spaCy model: {e}. Using NLTK fallback.")
                self.nlp = None
    
    def _ensure_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("📥 Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text into smaller segments with overlap"""
        if not text.strip():
            return []
        
        if self.nlp:
            return self._chunk_with_spacy(text)
        else:
            return self._chunk_with_nltk(text)
    
    def _chunk_with_spacy(self, text: str) -> List[str]:
        """Chunk text using spaCy for better sentence boundary detection"""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return self._create_chunks_from_sentences(sentences)
    
    def _chunk_with_nltk(self, text: str) -> List[str]:
        """Fallback chunking using NLTK"""
        sentences = sent_tokenize(text)
        return self._create_chunks_from_sentences(sentences)
    
    def _create_chunks_from_sentences(self, sentences: List[str]) -> List[str]:
        """Create overlapping chunks from sentences"""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed the limit
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.max_chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with current sentence
                if len(sentence) <= self.max_chunk_size:
                    current_chunk = sentence
                else:
                    # Handle very long sentences by splitting them
                    words = sentence.split()
                    word_chunks = self._split_words_into_chunks(words)
                    chunks.extend(word_chunks[:-1])  # Add all but last chunk
                    current_chunk = word_chunks[-1] if word_chunks else ""
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Add overlap if we have multiple chunks
        if len(chunks) > 1:
            chunks = self._add_overlap(chunks)
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _split_words_into_chunks(self, words: List[str]) -> List[str]:
        """Split words into chunks that fit within size limits"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length <= self.max_chunk_size:
                current_chunk.append(word)
                current_length += word_length
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]  # First chunk remains unchanged
        
        for i in range(1, len(chunks)):
            # Get overlap from previous chunk
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # Extract last part of previous chunk for overlap
            prev_words = prev_chunk.split()
            overlap_words = prev_words[-min(len(prev_words)//4, 20):]  # Use last 25% or 20 words
            overlap_text = " ".join(overlap_words)
            
            # Create overlapped chunk
            overlapped_chunk = overlap_text + " " + current_chunk
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
