import spacy
from typing import List

class TextChunker:
    def __init__(self, max_chunk_size=1500, chunk_overlap=200):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.nlp = spacy.load("en_core_web_sm")

    def chunk_text(self, text: str) -> List[str]:
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        chunks = []
        current_chunk = ""
        for sent in sentences:
            if len(current_chunk) + len(sent) + 1 <= self.max_chunk_size:
                current_chunk += (" " if current_chunk else "") + sent
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Overlap logic
                if self.chunk_overlap > 0 and chunks:
                    overlap = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap + " " + sent
                else:
                    current_chunk = sent
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks 