#!/usr/bin/env python3
"""
Content aggregation for optimal single-prompt generation.
Manages token limits and content prioritization.
"""

import re
from typing import List, Dict, Any
from loguru import logger

class ContentAggregator:
    """Aggregates content optimally for single-prompt generation"""
    
    def __init__(self):
        self.max_tokens = 800000  # Conservative limit for Gemini
        self.token_ratio = 1.3    # Approximate tokens per word
        
    def aggregate_for_single_prompt(self, embeddings_data: List[Dict], 
                                  topic: str, max_tokens: int = 800000) -> str:
        """Combine lecture notes + sample exams optimally for single prompt"""
        
        logger.info(f"📋 Aggregating content for topic: {topic}")
        
        # Separate content by type
        lecture_content = []
        exam_content = []
        model_answer_content = []
        
        for item in embeddings_data:
            content_type = item.get('content_type', 'unknown')
            chunk_text = item.get('chunk_text', '')
            
            if content_type == 'lecture_notes':
                lecture_content.append(chunk_text)
            elif content_type == 'exam_questions':
                exam_content.append(chunk_text)
            elif content_type == 'model_answers':
                model_answer_content.append(chunk_text)
        
        # Prioritize content relevance
        prioritized_content = self._prioritize_content_by_relevance(
            lecture_content, exam_content, model_answer_content, topic
        )
        
        # Manage token limits effectively
        final_content = self._manage_token_limits(prioritized_content, max_tokens)
        
        # Format for single prompt
        formatted_content = self._format_for_single_prompt(final_content)
        
        logger.info(f"✅ Aggregated content: {len(formatted_content)} characters")
        return formatted_content
    
    def _prioritize_content_by_relevance(self, lectures: List[str], 
                                       exams: List[str], 
                                       answers: List[str], 
                                       topic: str) -> Dict[str, List[str]]:
        """Prioritize content based on relevance to topic"""
        
        # Filter content by topic relevance
        topic_keywords = topic.lower().split()
        
        def calculate_relevance(text: str) -> float:
            text_lower = text.lower()
            score = 0
            for keyword in topic_keywords:
                score += text_lower.count(keyword)
            return score / len(text) if len(text) > 0 else 0
        
        # Sort each type by relevance
        lectures_sorted = sorted(lectures, key=calculate_relevance, reverse=True)
        exams_sorted = sorted(exams, key=calculate_relevance, reverse=True)
        answers_sorted = sorted(answers, key=calculate_relevance, reverse=True)
        
        return {
            'lectures': lectures_sorted,
            'exams': exams_sorted,
            'answers': answers_sorted
        }
    
    def _manage_token_limits(self, content_dict: Dict[str, List[str]], 
                           max_tokens: int) -> Dict[str, str]:
        """Manage token limits while maintaining content balance"""
        
        # Allocate tokens: 60% lectures, 25% exams, 15% answers
        lecture_tokens = int(max_tokens * 0.60)
        exam_tokens = int(max_tokens * 0.25)
        answer_tokens = int(max_tokens * 0.15)
        
        result = {}
        
        # Process each content type within limits
        for content_type, token_limit in [
            ('lectures', lecture_tokens),
            ('exams', exam_tokens), 
            ('answers', answer_tokens)
        ]:
            content_list = content_dict.get(content_type, [])
            combined_content = self._combine_within_token_limit(content_list, token_limit)
            result[content_type] = combined_content
        
        return result
    
    def _combine_within_token_limit(self, content_list: List[str], token_limit: int) -> str:
        """Combine content pieces within token limit"""
        combined = []
        current_tokens = 0
        
        for content in content_list:
            content_tokens = self._estimate_tokens(content)
            if current_tokens + content_tokens <= token_limit:
                combined.append(content)
                current_tokens += content_tokens
            else:
                # Try to include partial content
                remaining_tokens = token_limit - current_tokens
                if remaining_tokens > 100:  # Only if meaningful space left
                    partial_content = self._truncate_to_tokens(content, remaining_tokens)
                    combined.append(partial_content)
                break
        
        return "\n\n".join(combined)
    
    def _format_for_single_prompt(self, content_dict: Dict[str, str]) -> str:
        """Format aggregated content for single prompt"""
        
        sections = []
        
        # Add lecture notes section
        if content_dict.get('lectures'):
            sections.append(f"""
=== LECTURE NOTES CONTENT ===
{content_dict['lectures']}
=== END LECTURE NOTES ===
""")
        
        # Add sample exam papers section
        if content_dict.get('exams'):
            sections.append(f"""
=== SAMPLE EXAM PAPERS ===
{content_dict['exams']}
=== END SAMPLE EXAMS ===
""")
        
        # Add model answers section
        if content_dict.get('answers'):
            sections.append(f"""
=== MODEL ANSWERS REFERENCE ===
{content_dict['answers']}
=== END MODEL ANSWERS ===
""")
        
        return "\n".join(sections)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        word_count = len(text.split())
        return int(word_count * self.token_ratio)
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens"""
        max_words = int(max_tokens / self.token_ratio)
        words = text.split()
        
        if len(words) <= max_words:
            return text
        
        truncated_words = words[:max_words]
        truncated_text = " ".join(truncated_words)
        
        # Try to end at a sentence boundary
        last_period = truncated_text.rfind('.')
        if last_period > len(truncated_text) * 0.8:
            truncated_text = truncated_text[:last_period + 1]
        
        return truncated_text + "..."
