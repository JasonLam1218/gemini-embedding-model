#!/usr/bin/env python3
"""
Content aggregation for optimal single-prompt generation.
FIXED VERSION - Handles actual embeddings data structure properly.
"""

import re
from typing import List, Dict, Any
from loguru import logger

class ContentAggregator:
    """Aggregates content optimally for single-prompt generation"""
    
    def __init__(self):
        self.max_tokens = 800000
        self.token_ratio = 1.3
        logger.info("✅ ContentAggregator initialized")
    
    def aggregate_for_single_prompt(self, embeddings_data: List[Dict],
                                   topic: str, max_tokens: int = 800000) -> str:
        """Combine lecture notes + sample exams optimally for single prompt"""
        
        logger.info(f"📋 Aggregating content for topic: {topic}")
        logger.info(f"📊 Input embeddings data: {len(embeddings_data)} items")
        
        if not embeddings_data:
            logger.error("❌ No embeddings data provided")
            return self._fallback_load_content()
        
        # Debug: Check first item structure
        if embeddings_data:
            sample_keys = list(embeddings_data[0].keys())
            logger.info(f"📋 Sample data keys: {sample_keys}")
        
        # Extract content chunks with improved logic
        content_chunks = self._extract_content_chunks_improved(embeddings_data)
        
        if not content_chunks:
            logger.warning("⚠️ No content chunks extracted, using fallback")
            return self._fallback_load_content()
        
        # Build structured content
        aggregated_content = self._build_comprehensive_content(content_chunks)
        
        # Validate final content
        # if len(aggregated_content) < 1000:
        #     logger.warning(f"⚠️ Aggregated content too short ({len(aggregated_content)} chars), using fallback")
        #     return self._fallback_load_content()
        
        logger.info(f"✅ Aggregated content: {len(aggregated_content)} characters")
        return aggregated_content
    
    def _extract_content_chunks_improved(self, embeddings_data: List[Dict]) -> List[Dict]:
        """Extract content chunks with improved data structure handling"""
        
        content_chunks = []
        
        for i, item in enumerate(embeddings_data):
            # Handle multiple possible content keys
            chunk_text = None
            
            # Try different content key variations
            for key in ['chunk_text', 'content', 'text', 'chunk_content']:
                if key in item and item[key]:
                    chunk_text = item[key]
                    break
            
            # Skip if no content found
            if not chunk_text or len(chunk_text.strip()) < 50:
                logger.debug(f"Skipping item {i}: no substantial content")
                continue
            
            # Extract metadata
            content_type = self._determine_content_type(item, chunk_text)
            source_file = item.get('source_file', f'unknown_{i}')
            
            content_chunks.append({
                'content': chunk_text.strip(),
                'content_type': content_type,
                'source_file': source_file,
                'chunk_index': item.get('chunk_index', i)
            })
        
        logger.info(f"📄 Successfully extracted {len(content_chunks)} content chunks")
        return content_chunks
    
    def _determine_content_type(self, item: Dict, content: str) -> str:
        """Determine content type from item metadata or content analysis"""
        
        # Check explicit content_type
        if 'content_type' in item:
            return item['content_type']
        
        # Check source file for clues
        source_file = item.get('source_file', '').lower()
        if 'ms' in source_file or 'model' in source_file:
            return 'model_answers'
        elif 'exam' in source_file or 'paper' in source_file:
            return 'exam_questions'
        elif 'lecture' in source_file or 'chapter' in source_file:
            return 'lecture_notes'
        
        # Analyze content for clues
        content_lower = content.lower()
        if 'question' in content_lower and ('marks' in content_lower or 'points' in content_lower):
            return 'exam_questions'
        elif 'answer' in content_lower or 'solution' in content_lower:
            return 'model_answers'
        else:
            return 'lecture_notes'
    
    def _build_comprehensive_content(self, content_chunks: List[Dict]) -> str:
        """Build comprehensive content for prompt with optimized limits"""
        
        # Optimized limits for stable API processing
        MAX_EXAM_SECTIONS = 4      # Reduced from 4
        MAX_ANSWER_SECTIONS = 4    # Reduced from 4
        MAX_LECTURE_SECTIONS = 10   # Reduced from 15
        
        # Group by content type
        exam_papers = []
        model_answers = []
        lecture_notes = []
        
        for chunk in content_chunks:
            content_type = chunk['content_type']
            content = chunk['content']
            source = chunk['source_file']
            
            if content_type == 'exam_questions':
                exam_papers.append(f"=== EXAM_PAPER: {source} ===\n{content}\n=== END EXAM_PAPER ===")
            elif content_type == 'model_answers':
                model_answers.append(f"=== MODEL_ANSWERS: {source} ===\n{content}\n=== END MODEL_ANSWERS ===")
            elif content_type == 'lecture_notes':
                lecture_notes.append(f"=== LECTURE: {source} ===\n{content}\n=== END LECTURE ===")
        
        # Combine sections with optimized limits
        sections = []
        
        if exam_papers:
            sections.extend(exam_papers[:MAX_EXAM_SECTIONS])
            logger.info(f"📋 Added {len(exam_papers[:MAX_EXAM_SECTIONS])} exam paper sections")
            
        if model_answers:
            sections.extend(model_answers[:MAX_ANSWER_SECTIONS])
            logger.info(f"📝 Added {len(model_answers[:MAX_ANSWER_SECTIONS])} model answer sections")
            
        if lecture_notes:
            sections.extend(lecture_notes[:MAX_LECTURE_SECTIONS])
            logger.info(f"📚 Added {len(lecture_notes[:MAX_LECTURE_SECTIONS])} lecture sections")
        
        aggregated = "\n\n".join(sections)
        
        # Truncate if too long
        if len(aggregated) > self.max_tokens:
            aggregated = aggregated[:self.max_tokens]
            logger.warning(f"⚠️ Content truncated to {self.max_tokens} characters")
        
        return aggregated
    
    def validate_aggregated_content(self, content: str) -> Dict[str, Any]:
        """Validate aggregated content quality"""
        
        validation = {
            'length_adequate': len(content) > 1000,
            'has_exam_content': 'EXAM_PAPER' in content,
            'has_lecture_content': 'LECTURE' in content,
            'has_model_answers': 'MODEL_ANSWERS' in content,
            'content_sections': content.count('==='),
            'total_characters': len(content)
        }
        
        validation['overall_valid'] = (
            validation['length_adequate'] and 
            validation['has_lecture_content'] and
            validation['content_sections'] >= 3
        )
        
        return validation
    
    def _fallback_load_content(self) -> str:
        """Fallback: Load content directly from markdown files"""
        
        logger.info("🔄 Using fallback content loading from markdown files")
        
        from pathlib import Path
        markdown_dir = Path("data/output/converted_markdown")
        
        if not markdown_dir.exists():
            logger.error("❌ No markdown directory found for fallback")
            return ""
        
        content_sections = []
        total_chars = 0
        
        # Load exam papers first
        exam_dir = markdown_dir / "kelvin_papers"
        if exam_dir.exists():
            for md_file in exam_dir.glob("*.md"):
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if len(content) > 100:
                        section = f"=== EXAM_PAPER: {md_file.name} ===\n{content}\n=== END EXAM_PAPER ==="
                        content_sections.append(section)
                        total_chars += len(section)
                        
                        if total_chars > self.max_tokens * 0.3:
                            break
                            
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {md_file}: {e}")
        
        # Load lecture notes
        lectures_dir = markdown_dir / "lectures"
        if lectures_dir.exists():
            for md_file in list(lectures_dir.glob("*.md")): 
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if len(content) > 100:
                        section = f"=== LECTURE: {md_file.name} ===\n{content}\n=== END LECTURE ==="
                        content_sections.append(section)
                        total_chars += len(section)
                        
                        if total_chars > self.max_tokens:
                            break
                            
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {md_file}: {e}")
        
        fallback_content = "\n\n".join(content_sections)
        
        if len(fallback_content) > self.max_tokens:
            fallback_content = fallback_content[:self.max_tokens]
        
        logger.info(f"✅ Fallback content loaded: {len(fallback_content)} characters")
        return fallback_content
