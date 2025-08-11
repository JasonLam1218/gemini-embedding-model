#!/usr/bin/env python3
"""
Enhanced Content aggregation for optimal single-prompt generation.
Includes intelligent content curation and balanced selection algorithms.
Complete implementation with all existing functionality preserved.
"""

import re
import numpy as np
from typing import List, Dict, Any
from loguru import logger
from pathlib import Path

class EnhancedContentAggregator:
    """Enhanced content aggregator with intelligent balancing and quality filtering"""
    
    def __init__(self):
        self.content_weights = {
            'lecture_notes': 0.60,      # Primary educational content
            'exam_questions': 0.25,     # Example question patterns  
            'model_answers': 0.15       # Solution approaches
        }
        self.quality_thresholds = {
            'min_chunk_length': 200,
            'max_chunk_length': 2000,
            'min_concept_density': 0.1
        }
        self.max_tokens = 800000
        logger.info("✅ Enhanced ContentAggregator initialized with quality filtering")

    def aggregate_balanced_content(self, embeddings_data: List[Dict], 
                                 topic: str, max_tokens: int = 800000) -> str:
        """Aggregate content with intelligent balancing and quality filtering"""
        
        logger.info(f"📋 Enhanced content aggregation for topic: {topic}")
        logger.info(f"📊 Input embeddings data: {len(embeddings_data)} items")
        
        if not embeddings_data:
            logger.error("❌ No embeddings data provided")
            return self._fallback_load_content()
        
        # Step 1: Filter high-quality chunks
        quality_chunks = self._filter_quality_chunks(embeddings_data)
        logger.info(f"🔍 Filtered to {len(quality_chunks)} quality chunks")
        
        # Step 2: Group by content type with quality scoring
        content_groups = self._group_and_score_content(quality_chunks, topic)
        
        # Step 3: Apply balanced sampling
        selected_content = self._balanced_content_sampling(content_groups)
        
        # Step 4: Build optimized content structure
        return self._build_optimized_content_structure(selected_content)
    
    def _filter_quality_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Filter chunks based on quality metrics"""
        quality_chunks = []
        
        for chunk in chunks:
            chunk_text = chunk.get('chunk_text', '')
            
            # Quality checks
            if (len(chunk_text) >= self.quality_thresholds['min_chunk_length'] and
                len(chunk_text) <= self.quality_thresholds['max_chunk_length'] and
                self._calculate_concept_density(chunk_text) >= self.quality_thresholds['min_concept_density']):
                
                chunk['quality_score'] = self._calculate_quality_score(chunk)
                quality_chunks.append(chunk)
        
        return sorted(quality_chunks, key=lambda x: x['quality_score'], reverse=True)
    
    def _calculate_concept_density(self, text: str) -> float:
        """Calculate educational concept density in text"""
        concept_indicators = [
            'algorithm', 'method', 'approach', 'technique', 'principle',
            'theory', 'model', 'framework', 'analysis', 'implementation',
            'definition', 'concept', 'example', 'application', 'solution',
            'process', 'system', 'function', 'structure', 'pattern'
        ]
        
        words = text.lower().split()
        concept_count = sum(1 for word in words if any(indicator in word for indicator in concept_indicators))
        return concept_count / len(words) if words else 0
    
    def _calculate_quality_score(self, chunk: Dict) -> float:
        """Calculate comprehensive quality score for chunk"""
        text = chunk.get('chunk_text', '')
        content_type = chunk.get('content_type', '')
        
        # Base score from concept density
        base_score = self._calculate_concept_density(text)
        
        # Content type multiplier
        type_multipliers = {
            'lecture_notes': 1.0,
            'exam_questions': 0.8,  
            'model_answers': 0.9
        }
        
        # Length optimization (penalize too short or too long)
        length_score = 1.0
        if len(text) < 500:
            length_score = 0.8
        elif len(text) > 1500:
            length_score = 0.9
        
        # Academic language bonus
        academic_bonus = self._calculate_academic_language_score(text)
        
        return (base_score * type_multipliers.get(content_type, 0.5) * 
                length_score * (1 + academic_bonus))
    
    def _calculate_academic_language_score(self, text: str) -> float:
        """Calculate bonus for academic language usage"""
        academic_terms = [
            'analyze', 'evaluate', 'synthesize', 'demonstrate', 'investigate',
            'methodology', 'hypothesis', 'empirical', 'theoretical', 'framework'
        ]
        
        academic_count = sum(1 for term in academic_terms if term in text.lower())
        return min(0.2, academic_count * 0.02)  # Max 20% bonus
    
    def _group_and_score_content(self, quality_chunks: List[Dict], topic: str) -> Dict[str, List[Dict]]:
        """Group content by type and apply topic relevance scoring"""
        content_groups = {
            'lecture_notes': [],
            'exam_questions': [],
            'model_answers': []
        }
        
        topic_keywords = self._extract_topic_keywords(topic)
        
        for chunk in quality_chunks:
            content_type = chunk.get('content_type', 'lecture_notes')
            
            # Calculate topic relevance
            topic_relevance = self._calculate_topic_relevance(
                chunk.get('chunk_text', ''), topic_keywords
            )
            chunk['topic_relevance'] = topic_relevance
            
            # Adjust quality score with topic relevance
            chunk['final_score'] = chunk['quality_score'] * (1 + topic_relevance)
            
            if content_type in content_groups:
                content_groups[content_type].append(chunk)
            else:
                content_groups['lecture_notes'].append(chunk)
        
        # Sort each group by final score
        for content_type in content_groups:
            content_groups[content_type].sort(key=lambda x: x['final_score'], reverse=True)
        
        return content_groups
    
    def _extract_topic_keywords(self, topic: str) -> List[str]:
        """Extract keywords from topic for relevance scoring"""
        # Basic keyword extraction - can be enhanced with NLP
        keywords = topic.lower().replace('and', '').replace('&', '').split()
        
        # Add common variations
        expanded_keywords = keywords.copy()
        for keyword in keywords:
            if 'data' in keyword:
                expanded_keywords.extend(['analytics', 'analysis', 'mining'])
            elif 'ai' in keyword or 'artificial' in keyword:
                expanded_keywords.extend(['intelligence', 'machine', 'learning'])
        
        return list(set(expanded_keywords))
    
    def _calculate_topic_relevance(self, text: str, topic_keywords: List[str]) -> float:
        """Calculate relevance to topic based on keyword matching"""
        text_lower = text.lower()
        
        keyword_matches = sum(1 for keyword in topic_keywords if keyword in text_lower)
        max_possible_matches = len(topic_keywords)
        
        return keyword_matches / max_possible_matches if max_possible_matches > 0 else 0
    
    def _balanced_content_sampling(self, content_groups: Dict) -> List[Dict]:
        """Apply weighted sampling for balanced content selection"""
        selected_chunks = []
        total_target = 25  # Target total chunks
        
        for content_type, chunks in content_groups.items():
            target_weight = self.content_weights.get(content_type, 0.1)
            target_count = int(total_target * target_weight)
            
            # Ensure minimum representation
            min_count = 2 if chunks and content_type != 'model_answers' else 1
            target_count = max(target_count, min_count)
            
            # Select top chunks based on final scores
            selected = chunks[:min(target_count, len(chunks))]
            selected_chunks.extend(selected)
            
            logger.info(f"📊 Selected {len(selected)} {content_type} chunks (target: {target_count})")
        
        return selected_chunks
    
    def _build_optimized_content_structure(self, selected_content: List[Dict]) -> str:
        """Build optimized content structure with strict length limits"""
        
        # REDUCED maximum content size to prevent timeouts
        MAX_TOTAL_CHARS = 100000  # Reduced from 800000 to 100000
        
        # Group selected content by type
        grouped_content = {
            'exam_questions': [],
            'model_answers': [],
            'lecture_notes': []
        }
        
        for chunk in selected_content:
            content_type = chunk.get('content_type', 'lecture_notes')
            if content_type in grouped_content:
                grouped_content[content_type].append(chunk)
        
        # Build structured sections with length control
        sections = []
        current_length = 0
        
        # Add exam papers first (highest priority for pattern recognition)
        if grouped_content['exam_questions']:
            for chunk in grouped_content['exam_questions']:
                content_text = chunk.get('chunk_text', '')
                section = (
                    f"=== EXAM_PAPER: {chunk.get('source_file', 'Unknown')} ===\n"
                    f"QUALITY_SCORE: {chunk.get('final_score', 0):.2f}\n"
                    f"CONTENT:\n{content_text}\n"
                    f"=== END EXAM_PAPER ==="
                )
                
                if current_length + len(section) <= MAX_TOTAL_CHARS:
                    sections.append(section)
                    current_length += len(section)
                else:
                    break
            logger.info(f"📋 Added {len([s for s in sections if 'EXAM_PAPER' in s])} exam sections")
        
        # Add model answers
        if grouped_content['model_answers'] and current_length < MAX_TOTAL_CHARS:
            for chunk in grouped_content['model_answers']:
                content_text = chunk.get('chunk_text', '')
                section = (
                    f"=== MODEL_ANSWERS: {chunk.get('source_file', 'Unknown')} ===\n"
                    f"QUALITY_SCORE: {chunk.get('final_score', 0):.2f}\n"
                    f"CONTENT:\n{content_text}\n"
                    f"=== END MODEL_ANSWERS ==="
                )
                
                if current_length + len(section) <= MAX_TOTAL_CHARS:
                    sections.append(section)
                    current_length += len(section)
                else:
                    break
            logger.info(f"📝 Added {len([s for s in sections if 'MODEL_ANSWERS' in s])} answer sections")
        
        # Add lecture notes (primary educational content)
        if grouped_content['lecture_notes'] and current_length < MAX_TOTAL_CHARS:
            for chunk in grouped_content['lecture_notes']:
                content_text = chunk.get('chunk_text', '')
                section = (
                    f"=== LECTURE: {chunk.get('source_file', 'Unknown')} ===\n"
                    f"QUALITY_SCORE: {chunk.get('final_score', 0):.2f}\n"
                    f"TOPIC_RELEVANCE: {chunk.get('topic_relevance', 0):.2f}\n"
                    f"CONTENT:\n{content_text}\n"
                    f"=== END LECTURE ==="
                )
                
                if current_length + len(section) <= MAX_TOTAL_CHARS:
                    sections.append(section)
                    current_length += len(section)
                else:
                    break
            logger.info(f"📚 Added {len([s for s in sections if 'LECTURE' in s])} lecture sections")
        
        aggregated = "\n\n".join(sections)
        
        # Final safety check
        if len(aggregated) > MAX_TOTAL_CHARS:
            aggregated = aggregated[:MAX_TOTAL_CHARS]
            logger.warning(f"⚠️ Content truncated to {MAX_TOTAL_CHARS} characters")
        
        logger.info(f"✅ Optimized content aggregation complete: {len(aggregated)} characters")
        return aggregated

    
    def validate_aggregated_content(self, content: str) -> Dict[str, Any]:
        """Enhanced validation of aggregated content quality"""
        validation = {
            'length_adequate': len(content) > 2000,  # Increased threshold
            'has_exam_content': 'EXAM_PAPER' in content,
            'has_lecture_content': 'LECTURE' in content,
            'has_model_answers': 'MODEL_ANSWERS' in content,
            'content_sections': content.count('==='),
            'total_characters': len(content),
            'quality_indicators': self._assess_content_quality_indicators(content),
            'topic_coverage': self._assess_topic_coverage(content)
        }
        
        validation['overall_valid'] = (
            validation['length_adequate'] and
            validation['has_lecture_content'] and
            validation['content_sections'] >= 5 and  # Minimum sections
            validation['quality_indicators']['academic_language'] and
            validation['topic_coverage']['diverse_topics']
        )
        
        return validation
    
    def _assess_content_quality_indicators(self, content: str) -> Dict[str, bool]:
        """Assess various quality indicators in content"""
        return {
            'academic_language': len(re.findall(r'\b(analyze|evaluate|implement|demonstrate)\b', content, re.I)) >= 3,
            'technical_terms': len(re.findall(r'\b(algorithm|method|framework|system)\b', content, re.I)) >= 5,
            'examples_present': 'example' in content.lower() or 'instance' in content.lower(),
            'structured_content': content.count('===') >= 5
        }
    
    def _assess_topic_coverage(self, content: str) -> Dict[str, bool]:
        """Assess topic coverage diversity"""
        content_lower = content.lower()
        
        return {
            'diverse_topics': len(set(re.findall(r'===\s*\w+:', content))) >= 3,
            'sufficient_depth': len(content.split('\n')) >= 50,
            'varied_sources': len(set(re.findall(r'=== \w+: ([^=]+) ===', content))) >= 5
        }
    
    def _fallback_load_content(self) -> str:
        """Enhanced fallback content loading with STRICT size limits"""
        logger.info("🔄 Using enhanced fallback content loading")
        
        markdown_dir = Path("data/output/converted_markdown")
        if not markdown_dir.exists():
            logger.error("❌ No markdown directory found for fallback")
            return ""
        
        content_sections = []
        total_chars = 0
        MAX_FALLBACK_CHARS = 80000  # Even smaller for fallback
        
        for category, subdir in [("EXAM_PAPER", "kelvin_papers"), ("LECTURE", "lectures")]:
            category_dir = markdown_dir / subdir
            if not category_dir.exists():
                continue
            
            files_processed = 0
            for md_file in category_dir.glob("*.md"):
                if files_processed >= 5:  # Limit to 5 files per category
                    break
                
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Truncate individual files if too long
                    if len(content) > MAX_FALLBACK_CHARS // 10:
                        content = content[:MAX_FALLBACK_CHARS // 10]
                    
                    if len(content) > 200:
                        section = f"=== {category}: {md_file.name} ===\n{content}\n=== END {category} ==="
                        
                        # Check if adding this section would exceed limit
                        if total_chars + len(section) > MAX_FALLBACK_CHARS:
                            break
                        
                        content_sections.append(section)
                        total_chars += len(section)
                        files_processed += 1
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {md_file}: {e}")
        
        fallback_content = "\n\n".join(content_sections)
        logger.info(f"✅ Enhanced fallback content loaded: {len(fallback_content)} characters")
        return fallback_content


# Maintain backward compatibility with original interface
class ContentAggregator(EnhancedContentAggregator):
    """Backward compatible wrapper for enhanced content aggregator"""
    
    def __init__(self):
        super().__init__()
        # Keep original max_tokens and token_ratio for compatibility
        self.max_tokens = 800000
        self.token_ratio = 1.3
        logger.info("✅ ContentAggregator initialized with enhanced features")
    
    def aggregate_for_single_prompt(self, embeddings_data: List[Dict], 
                                   topic: str, max_tokens: int = 800000) -> str:
        """Legacy method - delegates to enhanced version"""
        return self.aggregate_balanced_content(embeddings_data, topic, max_tokens)
    
    def _extract_content_chunks_improved(self, embeddings_data: List[Dict]) -> List[Dict]:
        """Extract content chunks with improved data structure handling (legacy method)"""
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
        """Determine content type from item metadata or content analysis (legacy method)"""
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
        """Build comprehensive content for prompt with optimized limits (legacy method)"""
        # Optimized limits for stable API processing
        MAX_EXAM_SECTIONS = 4
        MAX_ANSWER_SECTIONS = 4
        MAX_LECTURE_SECTIONS = 10

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
