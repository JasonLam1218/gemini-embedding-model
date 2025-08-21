# File: src/core/content/content_aggregator.py

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
import math # NEW: Import math for ceil function

# Import TextLoader for content classification in fallback
from ..text.text_loader import TextLoader # Added import

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
            'max_chunk_length': 2000, # Corrected to max_chunk_length
            'min_concept_density': 0.1
        }
        self.max_tokens = 800000
        self.text_loader_for_fallback = TextLoader() # Initialize TextLoader for fallback
        logger.info("✅ Enhanced ContentAggregator initialized with quality filtering")

    def aggregate_balanced_content(self, embeddings_data: List[Dict], 
                                 topic: str, max_tokens: int = 800000) -> str:
        """Aggregate content with intelligent balancing and quality filtering"""
        
        logger.info(f"📋 Enhanced content aggregation for topic: {topic}")
        logger.info(f"📊 Input embeddings data: 13 items") # Fixed static log message
        
        if not embeddings_data:
            logger.error("❌ No embeddings data provided. Attempting fallback.")
            # Changed this to use the general fallback, which now processes flattened MD
            return self._fallback_load_content() 
        
        # Step 1: Filter high-quality chunks
        quality_chunks = self._filter_quality_chunks(embeddings_data)
        logger.info(f"🔍 Filtered to {len(quality_chunks)} quality chunks")
        
        if not quality_chunks: # If after filtering, no quality chunks remain, then also fallback
            logger.warning("⚠️ No quality chunks found after filtering. Attempting fallback.")
            return self._fallback_load_content()

        # Step 2: Group by content type with quality scoring
        content_groups = self._group_and_score_content(quality_chunks, topic)
        
        # Step 3: Apply balanced sampling
        selected_content = self._balanced_content_sampling(content_groups, max_tokens) # Pass max_tokens
        
        # Step 4: Build optimized content structure
        return self._build_optimized_content_structure(selected_content)
    
    def _filter_quality_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Filter chunks based on quality metrics"""
        quality_chunks = []
        
        for chunk in chunks:
            chunk_text = chunk.get('chunk_text', '')
            content_type = chunk.get('content_type', 'unknown') # Ensure 'unknown' is handled gracefully

            # Add a check here: if content_type is 'unknown' and it's not meant to be,
            # it indicates a classification issue upstream.
            # For this fix, the TextLoader is being modified to prevent 'unknown'.
            # So, assuming content_type is now correctly 'lecture_notes', 'exam_questions', 'model_answers'
            
            # Quality checks
            if (len(chunk_text) >= self.quality_thresholds['min_chunk_length'] and
                len(chunk_text) <= self.quality_thresholds['max_chunk_length'] and
                self._calculate_concept_density(chunk_text) >= self.quality_thresholds['min_concept_density']):
                
                chunk['quality_score'] = self._calculate_quality_score(chunk)
                quality_chunks.append(chunk)
            else:
                logger.debug(f"Skipping chunk due to quality filter (length {len(chunk_text)}, density {self._calculate_concept_density(chunk_text):.2f}): {chunk_text[:100]}...")

        return sorted(quality_chunks, key=lambda x: x['quality_score'], reverse=True)
    
    # NEWLY ADDED METHODS START HERE
    def _calculate_concept_density(self, text: str) -> float:
        """
        Calculate a basic concept density score.
        This is a placeholder; a more sophisticated method would use NLP to identify key terms.
        For now, it's based on text length and presence of common academic terms.
        """
        if not text:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0

        academic_keywords = [
            'algorithm', 'analysis', 'architecture', 'artificial intelligence', 'classification',
            'computational', 'concept', 'data', 'database', 'deep learning', 'design',
            'evaluation', 'framework', 'function', 'implementation', 'learning',
            'machine learning', 'model', 'neural network', 'optimization', 'performance',
            'prediction', 'programming', 'regression', 'statistical', 'supervised',
            'system', 'theory', 'unsupervised', 'validation', 'vector'
        ]
        
        keyword_count = sum(1 for word in words if word.lower() in academic_keywords)
        # Simple density: ratio of keywords to total words, with a bonus for longer text
        density = keyword_count / len(words)
        
        # Add a slight boost for longer texts, assuming more content = more concepts
        density += (len(text) / self.quality_thresholds['max_chunk_length']) * 0.1 
        
        return min(density, 1.0) # Cap at 1.0

    def _calculate_quality_score(self, chunk: Dict) -> float:
        """
        Calculate a quality score for a chunk.
        Combines length quality, concept density, and content type weighting.
        """
        chunk_text = chunk.get('chunk_text', '')
        content_type = chunk.get('content_type', 'lecture_notes') # Default to lecture_notes
        
        length_score = 0.0
        chunk_len = len(chunk_text)
        min_len = self.quality_thresholds['min_chunk_length']
        # CORRECTED LINE: Changed 'max_chunk_size' to 'max_chunk_length'
        max_len = self.quality_thresholds['max_chunk_length'] 
        
        if chunk_len >= min_len:
            length_score = min(chunk_len / max_len, 1.0) # Closer to max_len is better

        concept_density = self._calculate_concept_density(chunk_text)
        
        # Content type weight
        type_weight = self.content_weights.get(content_type, 0.5) # Default to 0.5 if unknown
        
        # Overall quality score (weighted sum)
        # This is a heuristic and can be refined
        quality = (length_score * 0.4) + (concept_density * 0.4) + (type_weight * 0.2)
        
        return quality
        
    def _group_and_score_content(self, quality_chunks: List[Dict], topic: str) -> Dict[str, List[Dict]]:
        """
        Groups chunks by content type and adds a relevance score to each chunk.
        """
        content_groups = {
            'lecture_notes': [],
            'exam_questions': [],
            'model_answers': []
        }
        
        topic_keywords = self._extract_topic_keywords(topic)
        
        for chunk in quality_chunks:
            chunk_text = chunk.get('chunk_text', '').lower()
            content_type = chunk.get('content_type', 'lecture_notes') # Default for safety
            
            # Calculate topic relevance
            relevance = self._calculate_topic_relevance(chunk_text, topic_keywords)
            chunk['topic_relevance'] = relevance
            
            # Combine quality and relevance for a final selection score
            chunk['final_score'] = (chunk.get('quality_score', 0) * 0.7) + (relevance * 0.3)
            
            if content_type in content_groups:
                content_groups[content_type].append(chunk)
            else:
                content_groups['lecture_notes'].append(chunk) # Default to lecture notes if type is unexpected

        # Sort chunks within each group by final_score (descending)
        for content_type in content_groups:
            content_groups[content_type] = sorted(content_groups[content_type], 
                                                  key=lambda x: x['final_score'], reverse=True)
            
        return content_groups

    def _extract_topic_keywords(self, topic: str) -> List[str]:
        """Extracts keywords from the topic string."""
        return [word.lower() for word in re.findall(r'\b\w+\b', topic) if len(word) > 2]

    def _calculate_topic_relevance(self, text: str, topic_keywords: List[str]) -> float:
        """Calculates how relevant a chunk is to the topic based on keyword presence."""
        if not topic_keywords:
            return 0.5 # Neutral if no keywords to compare
        
        text_lower = text.lower()
        matched_keywords = sum(1 for kw in topic_keywords if kw in text_lower)
        
        relevance = matched_keywords / len(topic_keywords)
        return relevance
        
    def _balanced_content_sampling(self, content_groups: Dict[str, List[Dict]], max_tokens: int) -> List[Dict]:
        """
        Samples content chunks from groups based on defined weights and overall token limit.
        Uses a dynamic approach to fill up the token budget.
        """
        selected_content = []
        current_tokens = 0
        target_tokens = max_tokens
        
        # Estimate average tokens per chunk (rough average)
        avg_tokens_per_chunk = 200 # A reasonable average for chunks around 1500 chars / 4 (char to token ratio)
        if content_groups.get('lecture_notes') and content_groups['lecture_notes'][0].get('chunk_text'):
             # Use actual first chunk to get a better estimate
             avg_tokens_per_chunk = math.ceil(len(content_groups['lecture_notes'][0]['chunk_text']) / 4)

        # Calculate target count for each type based on weights
        type_target_counts = {
            ctype: math.ceil(target_tokens * weight / avg_tokens_per_chunk)
            for ctype, weight in self.content_weights.items()
        }

        # Initialize current counts
        current_counts = {ctype: 0 for ctype in self.content_weights.keys()}

        # Simple greedy approach: take top 'N' from each type until token limit is hit
        # This might not perfectly hit the ratios but ensures some representation
        
        # First pass: try to get a base amount from each category
        for ctype, weight in self.content_weights.items():
            num_to_take = min(len(content_groups.get(ctype, [])), int(type_target_counts[ctype] * 0.5)) # Take 50% of target initially
            for i in range(num_to_take):
                chunk = content_groups[ctype][i]
                chunk_tokens = math.ceil(len(chunk['chunk_text']) / 4)
                if current_tokens + chunk_tokens <= target_tokens:
                    selected_content.append(chunk)
                    current_tokens += chunk_tokens
                    current_counts[ctype] += 1
                else:
                    logger.debug(f"Reached token limit for initial pass at {ctype}.")
                    break

        # Second pass: fill remaining budget by iterating through sorted chunks from all types,
        # prioritizing types that are underrepresented relative to their weight
        
        # Create a combined list of remaining chunks, sorted by final_score
        remaining_chunks = []
        for ctype, chunks in content_groups.items():
            for chunk_idx in range(current_counts[ctype], len(chunks)):
                remaining_chunks.append(chunks[chunk_idx])
        
        remaining_chunks = sorted(remaining_chunks, key=lambda x: x['final_score'], reverse=True)

        for chunk in remaining_chunks:
            chunk_tokens = math.ceil(len(chunk['chunk_text']) / 4)
            if current_tokens + chunk_tokens <= target_tokens:
                selected_content.append(chunk)
                current_tokens += chunk_tokens
                current_counts[chunk.get('content_type', 'lecture_notes')] += 1
            else:
                logger.debug(f"Reached overall token limit during second pass. Current tokens: {current_tokens}")
                break
        
        logger.info(f"Final selected content: {len(selected_content)} chunks, ~{current_tokens} tokens.")
        logger.info(f"Selected counts by type: {current_counts}")
        
        return selected_content

    def _build_optimized_content_structure(self, selected_content: List[Dict]) -> str:
        """
        Builds the final string representation of the aggregated content for the prompt,
        organized by content type.
        """
        
        structured_content = {
            'LECTURE_NOTES': [],
            'EXAM_QUESTIONS': [],
            'MODEL_ANSWERS': []
        }

        # Organize content by type
        for chunk in selected_content:
            content_type = chunk.get('content_type', 'lecture_notes')
            # Map internal content_type names to external format strings
            if content_type == 'lecture_notes':
                structured_content['LECTURE_NOTES'].append(chunk)
            elif content_type == 'exam_questions' or content_type == 'sample_paper':
                structured_content['EXAM_QUESTIONS'].append(chunk)
            elif content_type == 'model_answers':
                structured_content['MODEL_ANSWERS'].append(chunk)

        final_sections = []

        # Add lecture notes
        if structured_content['LECTURE_NOTES']:
            for chunk in structured_content['LECTURE_NOTES']:
                final_sections.append(f"""=== LECTURE: {chunk.get('source_file', 'unknown')} ===
SOURCE: {chunk.get('source_file', 'unknown')}
TYPE: {chunk.get('content_type', 'lecture_notes')}
LENGTH: {len(chunk['chunk_text'])} characters
CHUNK_INDEX: {chunk.get('chunk_index', 'N/A')}
CONTENT:
{chunk['chunk_text']}
=== END OF LECTURE: {chunk.get('source_file', 'unknown')} ===
""")
        
        # Add exam questions
        if structured_content['EXAM_QUESTIONS']:
            for chunk in structured_content['EXAM_QUESTIONS']:
                final_sections.append(f"""=== EXAM_PAPER: {chunk.get('source_file', 'unknown')} ===
SOURCE: {chunk.get('source_file', 'unknown')}
TYPE: {chunk.get('content_type', 'exam_questions')}
LENGTH: {len(chunk['chunk_text'])} characters
CHUNK_INDEX: {chunk.get('chunk_index', 'N/A')}
CONTENT:
{chunk['chunk_text']}
=== END OF EXAM_PAPER: {chunk.get('source_file', 'unknown')} ===
""")
        
        # Add model answers
        if structured_content['MODEL_ANSWERS']:
            for chunk in structured_content['MODEL_ANSWERS']:
                final_sections.append(f"""=== MODEL_ANSWERS: {chunk.get('source_file', 'unknown')} ===
SOURCE: {chunk.get('source_file', 'unknown')}
TYPE: {chunk.get('content_type', 'model_answers')}
LENGTH: {len(chunk['chunk_text'])} characters
CHUNK_INDEX: {chunk.get('chunk_index', 'N/A')}
CONTENT:
{chunk['chunk_text']}
=== END OF MODEL_ANSWERS: {chunk.get('source_file', 'unknown')} ===
""")

        return "\n\n".join(final_sections)

    def validate_aggregated_content(self, aggregated_content: str) -> Dict[str, Any]:
        """Assess the quality and coverage of the aggregated content."""
        
        validation_results = {
            "length_adequate": len(aggregated_content) > self.quality_thresholds['min_chunk_length'] * 3, # Minimum of 3 chunks worth of content
            "total_characters": len(aggregated_content),
            "content_sections": aggregated_content.count("===") // 2, # Count number of sections
            "has_exam_content": "=== EXAM_PAPER:" in aggregated_content,
            "has_lecture_content": "=== LECTURE:" in aggregated_content,
            "has_model_answers": "=== MODEL_ANSWERS:" in aggregated_content,
            "quality_indicators": self._assess_content_quality_indicators(aggregated_content),
            "topic_coverage_score": self._assess_topic_coverage(aggregated_content)
        }
        
        validation_results["overall_valid"] = (
            validation_results["length_adequate"] and
            validation_results["has_lecture_content"] and
            (validation_results["has_exam_content"] or validation_results["has_model_answers"]) and # At least one type of assessment content
            validation_results["content_sections"] >= 3 # At least 3 logical sections (lecture, exam, answers)
        )
        
        return validation_results

    def _assess_content_quality_indicators(self, content: str) -> Dict[str, Any]:
        """Assess quality indicators within the aggregated content."""
        # Simple heuristic examples
        return {
            "contains_figures_tables_markers": "![Image" in content or "<table>" in content or "|" in content,
            "contains_code_blocks": "```" in content,
            "average_line_length": np.mean([len(line) for line in content.split('\n') if line.strip()]),
            "readability_score": None # Placeholder for a real readability score if implemented
        }

    def _assess_topic_coverage(self, content: str) -> float:
        """Assess how well the aggregated content covers the relevant topic areas."""
        # This would ideally be based on keywords extracted from the topic
        # and checking their distribution in the content.
        # For simplicity, a basic check for now.
        if not content:
            return 0.0
        
        # Placeholder keywords for 'AI and Data Analytics'
        topic_specific_keywords = ['machine learning', 'artificial intelligence', 'data analysis', 'deep learning', 'supervised', 'unsupervised', 'neural network', 'python', 'algorithm']
        
        score = sum(1 for keyword in topic_specific_keywords if keyword in content.lower())
        
        return min(score / len(topic_specific_keywords), 1.0) # Normalized score
    # NEWLY ADDED METHODS END HERE

    def _fallback_load_content(self) -> str:
        """Enhanced fallback content loading with STRICT size limits, adapted for flattened structure."""
        logger.info("🔄 Using enhanced fallback content loading from flattened markdown directory.")
        
        markdown_dir = Path("data/output/converted_markdown")
        if not markdown_dir.exists():
            logger.error("❌ No markdown directory found for fallback")
            return ""
        
        content_sections = []
        total_chars = 0
        MAX_FALLBACK_CHARS = 80000  # Even smaller for fallback
        
        # Iterate directly over all markdown files in the root converted_markdown directory
        for md_file in markdown_dir.glob("*.md"): # Changed to glob
            if md_file.name == "README.md":
                continue
            
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Use TextLoader's classification logic for this file to get the type
                inferred_content_type = self.text_loader_for_fallback.classify_content_type(md_file)

                # Map to the desired section header format
                section_header_type = "UNKNOWN_TYPE"
                if "exam_questions" in inferred_content_type or "sample_paper" in inferred_content_type:
                    section_header_type = "EXAM_PAPER"
                elif "model_answers" in inferred_content_type:
                    section_header_type = "MODEL_ANSWERS"
                elif "lecture_notes" in inferred_content_type:
                    section_header_type = "LECTURE"
                
                if len(content) > MAX_FALLBACK_CHARS // 10: # Truncate individual files
                    content = content[:MAX_FALLBACK_CHARS // 10]
                
                if len(content) > 200: # Ensure substantial content
                    section = f"=== {section_header_type}: {md_file.name} ===\n{content}\n=== END {section_header_type} ==="
                    
                    if total_chars + len(section) > MAX_FALLBACK_CHARS:
                        logger.warning(f"⚠️ Fallback content limit reached. Skipping {md_file.name}.")
                        break
                    
                    content_sections.append(section)
                    total_chars += len(section)
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {md_file} for fallback: {e}")
        
        fallback_content = "\n\n".join(content_sections)
        logger.info(f"✅ Enhanced fallback content loaded: {len(fallback_content)} characters")
        return fallback_content

# Maintain backward compatibility with original interface
class ContentAggregator(EnhancedContentAggregator):
    """Backward compatible wrapper for enhanced content aggregator"""
    
    def __init__(self):
        super().__init__()
        self.max_tokens = 800000
        self.token_ratio = 1.3
        logger.info("✅ ContentAggregator initialized with enhanced features (compatibility layer)")
    
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