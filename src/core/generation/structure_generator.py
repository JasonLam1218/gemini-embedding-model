import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from pathlib import Path
from datetime import datetime
import random
import time
import re
import markdown
from markdown_pdf import MarkdownPdf
from weasyprint import HTML, CSS

from ..embedding.gemini_client import GeminiClient
from ..embedding.embedding_generator import EmbeddingGenerator

class StructureGenerator:
    def __init__(self):
        self.gemini_client = GeminiClient()
        self.embedding_gen = EmbeddingGenerator()
        self.enable_enhanced_variety = False
        
        # Cache for generated content to reduce API calls
        self.question_cache = {}
        self.model_answer_cache = {}
        self.marking_scheme_cache = {}
        
        # API request tracking
        self.api_requests_made = 0
        self.max_api_requests = 15  # Stay under the 50 limit
        
        # Standard exam structure template
        self.standard_structure = {
            "Q1": {
                "main_topic": "Fundamental concepts and definitions",
                "sub_parts": [
                    {"part": "(a)", "focus": "basic definitions", "marks": 8},
                    {"part": "(b)", "focus": "conceptual explanation", "marks": 9},
                    {"part": "(c)", "focus": "comparison and contrast", "marks": 8}
                ],
                "total_marks": 25
            },
            "Q2": {
                "main_topic": "Applications and implementations", 
                "sub_parts": [
                    {"part": "(a)(i)", "focus": "practical applications", "marks": 8},
                    {"part": "(a)(ii)", "focus": "real-world examples", "marks": 8},
                    {"part": "(b)", "focus": "implementation challenges", "marks": 9}
                ],
                "total_marks": 25
            },
            "Q3": {
                "main_topic": "Technical analysis and problem-solving",
                "sub_parts": [
                    {"part": "(a)", "focus": "technical analysis", "marks": 8},
                    {"part": "(b)", "focus": "problem-solving approach", "marks": 9},
                    {"part": "(c)", "focus": "evaluation and critique", "marks": 8}
                ],
                "total_marks": 25
            },
            "Q4": {
                "main_topic": "Critical evaluation and future perspectives",
                "sub_parts": [
                    {"part": "(i)", "focus": "critical evaluation", "marks": 10},
                    {"part": "(ii)", "focus": "limitations and challenges", "marks": 9},
                    {"part": "(iii)", "focus": "future developments", "marks": 6}
                ],
                "total_marks": 25
            }
        }
        
        logger.info("✅ Structure Generator initialized with AI-powered academic content generation")

    def generate_structured_exam(self, topic: str = "AI and Data Analytics",
                                structure_type: str = "standard",
                                num_main_questions: Optional[int] = None) -> Dict:
        """Generate structured exam with comprehensive AI-powered model answers and marking schemes"""
        try:
            logger.info(f"🔄 Generating exam for topic: {topic}")
            
            # Reset API request counter
            self.api_requests_made = 0
            
            # Load embeddings data
            embeddings_data = self._load_embeddings_data()
            if not embeddings_data:
                logger.error("❌ No embeddings data available")
                return self._create_empty_exam_structure(topic)

            # Generate topic embedding 
            topic_embedding = self.embedding_gen.generate_topic_embedding(topic)
            if not topic_embedding:
                logger.error("❌ Failed to generate topic embedding")
                return self._create_empty_exam_structure(topic)

            # Find relevant content
            relevant_content = self.embedding_gen.find_similar_chunks(
                topic_embedding, embeddings_data, top_k=50, min_similarity=0.15
            )

            if not relevant_content:
                logger.error("❌ No relevant content found")
                return self._create_empty_exam_structure(topic)

            # Use standard structure to manage API usage
            if num_main_questions is None:
                num_main_questions = 4

            # Generate exam structure with comprehensive AI-powered content
            exam_structure = self._generate_exam_with_comprehensive_ai_content(
                topic, relevant_content, self.standard_structure
            )

            # Format final exam paper
            formatted_exam = self._format_complete_exam_paper(
                exam_structure, topic, len(relevant_content)
            )

            logger.info(f"✅ Generated exam with {len(exam_structure['questions'])} questions")
            logger.info(f"📊 Total API requests made: {self.api_requests_made}")
            return formatted_exam

        except Exception as e:
            logger.error(f"❌ Exam generation failed: {e}")
            return self._create_empty_exam_structure(topic)

    def _generate_exam_with_comprehensive_ai_content(self, topic: str, relevant_content: List[Dict], 
                                                   structure_template: Dict) -> Dict:
        """Generate exam questions with comprehensive AI-powered model answers and marking schemes"""
        exam_questions = {
            "title": f"AI-Enhanced Comprehensive Examination - {topic}",
            "topic": topic,
            "total_questions": len(structure_template),
            "total_marks": 100,
            "questions": {}
        }

        # Group content into clusters
        content_clusters = self._cluster_content_by_similarity(relevant_content)

        for q_num, q_structure in structure_template.items():
            logger.info(f"🔄 Generating {q_num}: {q_structure['main_topic']}")
            
            # Find best content cluster
            best_cluster = self._select_best_content_cluster(
                q_structure, content_clusters, topic
            )

            # Generate question with comprehensive AI-powered content
            question_data = self._create_comprehensive_ai_enhanced_question(
                q_num, q_structure, best_cluster, topic
            )

            exam_questions["questions"][q_num] = question_data

        return exam_questions

    def _create_comprehensive_ai_enhanced_question(self, q_num: str, q_structure: Dict,
                                                 content_cluster: List[Dict], topic: str) -> Dict:
        """Create question with comprehensive AI-generated model answers and marking schemes"""
        try:
            main_topic = q_structure["main_topic"]
            sub_parts = q_structure["sub_parts"]

            # Select diverse content for different sub-parts
            selected_content = self._select_diverse_content_for_subparts(
                content_cluster, len(sub_parts)
            )

            question_data = {
                "question_number": q_num,
                "main_topic": main_topic,
                "total_marks": q_structure["total_marks"],
                "sub_questions": [],
                "content_sources": [item.get('source_file', 'Unknown') for item in selected_content[:3]]
            }

            # Generate each sub-part with comprehensive AI content
            for i, sub_part in enumerate(sub_parts):
                content_item = selected_content[i % len(selected_content)] if selected_content else {}
                
                # Generate comprehensive AI-enhanced sub-question
                sub_question = self._generate_comprehensive_ai_enhanced_subpart(
                    sub_part, content_item, topic, main_topic
                )

                question_data["sub_questions"].append(sub_question)

            return question_data

        except Exception as e:
            logger.error(f"❌ Failed to create comprehensive AI-enhanced question {q_num}: {e}")
            return self._create_fallback_question(q_num, q_structure, topic)

    def _generate_comprehensive_ai_enhanced_subpart(self, sub_part: Dict, content_item: Dict,
                                                  topic: str, main_topic: str) -> Dict:
        """Generate sub-part question with comprehensive AI-powered model answer and marking scheme"""
        part_label = sub_part["part"]
        focus = sub_part["focus"]
        marks = sub_part["marks"]

        # Extract content
        content_text = content_item.get('chunk_text', '')
        content_snippet = content_text[:1200] if len(content_text) > 1200 else content_text

        # Generate question text with quota awareness
        question_text = self._generate_question_with_quota_check(
            focus, topic, main_topic, content_snippet
        )
        
        # Generate comprehensive AI-powered model answer
        model_answer = self._generate_comprehensive_ai_model_answer(
            question_text, focus, topic, content_snippet, marks
        )
        
        # Generate comprehensive AI-powered marking scheme
        detailed_marking_scheme = self._generate_comprehensive_ai_marking_scheme(
            question_text, focus, topic, content_snippet, marks
        )

        # Generate basic marking criteria as backup
        basic_marking_criteria = self._generate_marking_criteria(focus, marks, content_text)

        return {
            "part": part_label,
            "question": question_text,
            "marks": marks,
            "focus": focus,
            "model_answer": model_answer,
            "detailed_marking_scheme": detailed_marking_scheme,
            "basic_marking_criteria": basic_marking_criteria,
            "source_content": content_snippet,
            "similarity_score": content_item.get('similarity_score', 0)
        }

    def _generate_question_with_quota_check(self, focus: str, topic: str, 
                                          main_topic: str, content_snippet: str) -> str:
        """Generate question text with API quota awareness"""
        try:
            # Check quota before making API call
            if self.api_requests_made >= self.max_api_requests:
                logger.warning("⚠️ API quota reached, using template for question generation")
                return self._create_template_question(focus, topic, main_topic)

            # Check cache first
            cache_key = f"q_{focus}_{topic}_{main_topic}"
            if cache_key in self.question_cache:
                logger.info(f"📋 Using cached question for {focus}")
                return self._customize_cached_question(
                    self.question_cache[cache_key], content_snippet
                )

            # Try AI generation with enhanced prompting
            prompt = f"""Create a comprehensive university-level exam question about {focus} in {topic}.

Context: This question is part of the {main_topic} section.

Reference content for context:
{content_snippet[:500]}

Requirements:
- Create a clear, specific question suitable for university examination
- Focus specifically on {focus} aspects within {topic}
- Make it relevant to {main_topic} context
- Use academic language appropriate for higher education
- The question should test deep understanding and analytical thinking
- Include specific examples or scenarios where relevant
- Ensure the question allows for comprehensive discussion and analysis
- Frame it to encourage detailed, structured responses

Generate only the question text, no additional formatting or instructions."""

            try:
                response = self.gemini_client.generate_content(
                    prompt, temperature=0.4, max_tokens=400
                )
                
                self.api_requests_made += 1
                generated_question = response.strip()
                
                if generated_question and len(generated_question) > 30:
                    # Cache successful generation
                    self.question_cache[cache_key] = generated_question
                    logger.info(f"✅ Generated AI question for {focus}")
                    return generated_question
                else:
                    raise ValueError("Generated question too short or empty")
                    
            except Exception as api_error:
                logger.warning(f"⚠️ Question generation API failed: {api_error}")
                return self._create_template_question(focus, topic, main_topic)

        except Exception as e:
            logger.error(f"❌ Question generation failed: {e}")
            return self._create_template_question(focus, topic, main_topic)

    def _generate_comprehensive_ai_model_answer(self, question_text: str, focus: str, topic: str,
                                              content_snippet: str, marks: int) -> str:
        """Generate comprehensive AI-powered model answer in academic style"""
        try:
            # Check quota
            if self.api_requests_made >= self.max_api_requests:
                logger.warning("⚠️ API quota reached, using template for model answer")
                return self._create_comprehensive_template_model_answer(question_text, focus, topic, marks)

            # Check cache
            cache_key = f"ans_{focus}_{marks}_{hash(question_text[:100])}"
            if cache_key in self.model_answer_cache:
                logger.info(f"📋 Using cached model answer for {focus}")
                return self.model_answer_cache[cache_key]

            prompt = f"""Generate a comprehensive, detailed model answer for this university exam question in academic style:

QUESTION: {question_text}

CONTEXT:
- Topic: {topic}
- Focus Area: {focus}
- Mark Value: {marks} marks
- Reference Content: {content_snippet[:600]}

REQUIREMENTS FOR MODEL ANSWER:
- Provide a detailed, comprehensive academic-style model answer
- Structure the answer with clear headings, subheadings, and bullet points where appropriate
- Include multiple paragraphs with logical flow and academic transitions
- Demonstrate deep understanding of {focus} within {topic}
- Include relevant examples, applications, and explanations from the reference content
- Use proper academic terminology and concepts throughout
- Match the depth and complexity to the {marks} mark allocation
- Include specific details, definitions, and explanations that would earn full marks
- Make connections between different concepts and show analytical thinking
- Use academic language with clear explanations
- Provide comprehensive coverage that shows mastery of the subject
- Include practical applications and real-world relevance where appropriate
- Structure with clear introductory statements and concluding insights

FORMAT STYLE:
- Write in paragraphs with academic structure
- Use bullet points for lists of key points or features
- Include clear transitions between ideas
- Demonstrate critical thinking and analysis
- Show depth of knowledge appropriate for university level

The model answer should be comprehensive enough that a student following this would receive full marks and demonstrate complete understanding of the topic."""

            try:
                response = self.gemini_client.generate_content(
                    prompt, temperature=0.2, max_tokens=1200
                )
                
                self.api_requests_made += 1
                model_answer = response.strip()
                
                if model_answer and len(model_answer) > 200:
                    # Enhance formatting for academic style
                    model_answer = self._enhance_model_answer_formatting(model_answer)
                    
                    # Cache successful generation
                    self.model_answer_cache[cache_key] = model_answer
                    logger.info(f"✅ Generated comprehensive AI model answer for {focus} ({len(model_answer)} chars)")
                    return model_answer
                else:
                    raise ValueError("Generated model answer too short")
                    
            except Exception as api_error:
                logger.warning(f"⚠️ Model answer API failed: {api_error}")
                return self._create_comprehensive_template_model_answer(question_text, focus, topic, marks)

        except Exception as e:
            logger.error(f"❌ Model answer generation failed: {e}")
            return self._create_comprehensive_template_model_answer(question_text, focus, topic, marks)

    def _generate_comprehensive_ai_marking_scheme(self, question_text: str, focus: str, topic: str,
                                                content_snippet: str, marks: int) -> Dict[str, Any]:
        """Generate detailed AI-powered marking scheme in academic style"""
        try:
            # Check quota
            if self.api_requests_made >= self.max_api_requests:
                logger.warning("⚠️ API quota reached, using template for marking scheme")
                return self._create_comprehensive_template_marking_scheme(focus, marks)

            # Check cache
            cache_key = f"mark_{focus}_{marks}_{hash(question_text[:100])}"
            if cache_key in self.marking_scheme_cache:
                logger.info(f"📋 Using cached marking scheme for {focus}")
                return self.marking_scheme_cache[cache_key]

            prompt = f"""Create a detailed, comprehensive marking scheme for this university exam question:

QUESTION: {question_text}

CONTEXT:
- Topic: {topic}
- Focus Area: {focus}
- Total Marks: {marks}
- Reference Content: {content_snippet[:500]}

REQUIREMENTS FOR MARKING SCHEME:
- Break down the {marks} marks into specific, measurable criteria
- Provide clear mark allocation for different aspects of the answer
- Specify what students must demonstrate to earn each mark band
- Include key concepts, examples, definitions, and explanations that should be covered
- Make it specific enough for consistent marking by different examiners
- Focus on {focus} within {topic} context
- Base criteria on the reference content and academic expectations
- Include both knowledge demonstration and analytical thinking requirements
- Specify partial mark allocation for incomplete but correct responses
- Include guidance for different levels of answer quality

FORMAT REQUIREMENTS:
- Provide detailed marking criteria with specific mark allocations
- Include descriptions of what constitutes excellent, good, and satisfactory responses
- Specify key terms, concepts, and examples that should be present
- Include mark breakdown that totals exactly {marks} marks
- Provide guidance for awarding partial marks
- Make it practical and clear for university-level assessment

STRUCTURE:
1. Overall marking approach
2. Detailed criteria with specific mark allocations
3. Key concepts and examples that should be covered
4. Partial marking guidance
5. Quality indicators for different mark bands

Make the marking scheme comprehensive and academically rigorous."""

            try:
                response = self.gemini_client.generate_content(
                    prompt, temperature=0.1, max_tokens=800
                )
                
                self.api_requests_made += 1
                scheme_text = response.strip()
                
                if scheme_text and len(scheme_text) > 100:
                    # Parse the response into structured format
                    mark_breakdown = self._parse_comprehensive_ai_marking_scheme(scheme_text, marks)
                    
                    marking_scheme = {
                        "detailed_scheme": scheme_text,
                        "mark_breakdown": mark_breakdown,
                        "total_marks": marks,
                        "generated_by": "comprehensive_gemini_ai",
                        "academic_style": True
                    }
                    
                    # Cache successful generation
                    self.marking_scheme_cache[cache_key] = marking_scheme
                    logger.info(f"✅ Generated comprehensive AI marking scheme for {focus} ({marks} marks)")
                    return marking_scheme
                else:
                    raise ValueError("Generated marking scheme too short")
                    
            except Exception as api_error:
                logger.warning(f"⚠️ Marking scheme API failed: {api_error}")
                return self._create_comprehensive_template_marking_scheme(focus, marks)

        except Exception as e:
            logger.error(f"❌ Marking scheme generation failed: {e}")
            return self._create_comprehensive_template_marking_scheme(focus, marks)

    def _enhance_model_answer_formatting(self, model_answer: str) -> str:
        """Enhance model answer formatting for academic style"""
        try:
            # Add proper paragraph spacing
            model_answer = re.sub(r'\n\s*\n', '\n\n', model_answer)
            
            # Ensure proper bullet point formatting
            model_answer = re.sub(r'\n\s*[-•]\s*', '\n• ', model_answer)
            
            # Clean up extra spaces
            model_answer = re.sub(r' +', ' ', model_answer)
            
            # Ensure proper sentence spacing
            model_answer = re.sub(r'\.(\w)', r'. \1', model_answer)
            
            return model_answer.strip()
            
        except Exception as e:
            logger.error(f"❌ Failed to enhance model answer formatting: {e}")
            return model_answer

    def _parse_comprehensive_ai_marking_scheme(self, scheme_text: str, total_marks: int) -> List[Dict]:
        """Parse comprehensive AI-generated marking scheme into structured breakdown"""
        try:
            lines = scheme_text.split('\n')
            breakdown = []
            
            for line in lines:
                # Look for various patterns of mark allocation
                mark_patterns = [
                    r'(\d+)\s*marks?',
                    r'(\d+)\s*pts?',
                    r'(\d+)\s*points?'
                ]
                
                for pattern in mark_patterns:
                    matches = re.findall(pattern, line.lower())
                    if matches and any(char in line for char in [':', '(', '-', '•']):
                        # Extract criterion text
                        criterion = line
                        # Remove numbering
                        criterion = re.sub(r'^\d+[\.\)]\s*', '', criterion)
                        # Remove mark information
                        criterion = re.sub(r'\(\d+\s*marks?\)', '', criterion)
                        criterion = re.sub(r':\s*\d+\s*marks?', '', criterion)
                        criterion = re.sub(r'-\s*\d+\s*marks?', '', criterion)
                        # Remove bullet points
                        criterion = re.sub(r'^[•\-\*]\s*', '', criterion)
                        criterion = criterion.strip()
                        
                        if criterion and len(criterion) > 5:
                            marks = int(matches[0])
                            breakdown.append({"criterion": criterion, "marks": marks})
                        break
            
            # Validate and adjust total marks
            if breakdown:
                total_parsed = sum(item['marks'] for item in breakdown)
                if total_parsed != total_marks:
                    # Try to adjust proportionally if close
                    if abs(total_parsed - total_marks) <= 2:
                        # Adjust the largest criterion
                        if breakdown:
                            largest_idx = max(range(len(breakdown)), key=lambda i: breakdown[i]['marks'])
                            breakdown[largest_idx]['marks'] += (total_marks - total_parsed)
                    else:
                        # Fall back to balanced breakdown
                        return self._create_comprehensive_balanced_breakdown(total_marks)
                return breakdown
            
            # Fallback if parsing fails
            return self._create_comprehensive_balanced_breakdown(total_marks)
            
        except Exception as e:
            logger.error(f"❌ Failed to parse comprehensive AI marking scheme: {e}")
            return self._create_comprehensive_balanced_breakdown(total_marks)

    def _create_comprehensive_balanced_breakdown(self, total_marks: int) -> List[Dict]:
        """Create comprehensive balanced mark breakdown when parsing fails"""
        if total_marks <= 5:
            return [{"criterion": "Overall understanding, explanation, and application", "marks": total_marks}]
        elif total_marks <= 8:
            half = total_marks // 2
            return [
                {"criterion": "Conceptual understanding and definitions", "marks": half},
                {"criterion": "Application, examples, and analysis", "marks": total_marks - half}
            ]
        else:
            # Distribute marks across multiple criteria for comprehensive assessment
            base = total_marks // 4
            remainder = total_marks % 4
            
            criteria = [
                {"criterion": "Conceptual understanding and key definitions", "marks": base + (1 if remainder > 0 else 0)},
                {"criterion": "Detailed explanations and theoretical foundations", "marks": base + (1 if remainder > 1 else 0)},
                {"criterion": "Practical applications and relevant examples", "marks": base + (1 if remainder > 2 else 0)},
                {"criterion": "Critical analysis, evaluation, and synthesis", "marks": base}
            ]
            
            return criteria

    def _customize_cached_question(self, base_question: str, content_snippet: str) -> str:
        """Customize a cached question with specific content reference"""
        if content_snippet and len(content_snippet) > 100:
            # Add context-specific elements
            if "example" in base_question.lower():
                return base_question + " Use relevant examples from the provided content to support your comprehensive analysis."
            elif "discuss" in base_question.lower():
                return base_question + " Reference specific concepts from the given material in your detailed discussion."
            elif "explain" in base_question.lower():
                return base_question + " Draw upon the provided content to support your comprehensive explanation."
        return base_question

    def _create_template_question(self, focus: str, topic: str, main_topic: str) -> str:
        """Create comprehensive template-based question as fallback"""
        templates = {
            "basic definitions": f"Define and comprehensively explain the key concepts of {focus} in {topic}. Provide clear, detailed definitions and explain their significance in the context of {main_topic}. Include relevant examples to demonstrate your understanding.",
            
            "conceptual explanation": f"Provide a comprehensive explanation of how {focus} relates to {main_topic} in the context of {topic}. Discuss the underlying principles, theoretical foundations, and practical implications. Include specific examples to illustrate your points.",
            
            "comparison and contrast": f"Compare and contrast different approaches to {focus} in {topic}. Provide a detailed analysis of their strengths, weaknesses, and appropriate use cases. Include specific examples and evaluate their effectiveness in various contexts.",
            
            "practical applications": f"Discuss in detail the practical applications of {focus} in {topic}. Provide specific, comprehensive examples and analyze their effectiveness in real-world scenarios. Evaluate the benefits and limitations of each application.",
            
            "real-world examples": f"Provide and analyze comprehensive real-world examples of {focus} in {topic}. Explain in detail how these examples demonstrate key principles and concepts. Evaluate their significance and impact in practical contexts.",
            
            "implementation challenges": f"Analyze comprehensively the main challenges in implementing {focus} within {topic}. Discuss potential solutions, mitigation strategies, and best practices. Include specific examples and evaluate the effectiveness of different approaches.",
            
            "technical analysis": f"Conduct a comprehensive technical analysis of {focus} in relation to {topic}. Examine in detail the technical aspects, requirements, and considerations involved. Include specific examples and evaluate different technical approaches.",
            
            "problem-solving approach": f"Describe and analyze a systematic approach to solving problems related to {focus} in {topic}. Outline in detail the steps, methodologies, and best practices. Include specific examples and evaluate the effectiveness of different problem-solving strategies.",
            
            "evaluation and critique": f"Critically evaluate the effectiveness of {focus} in {topic}. Provide a comprehensive analysis of its benefits, limitations, and overall impact. Include specific examples and discuss future implications and improvements.",
            
            "critical evaluation": f"Provide a comprehensive critical evaluation of {focus} considering its impact on {topic}. Assess in detail its significance, effectiveness, and future potential. Include specific examples and analyze both positive and negative aspects.",
            
            "limitations and challenges": f"Discuss comprehensively the limitations and challenges associated with {focus} in {topic}. Analyze the constraints in detail and propose potential solutions. Include specific examples and evaluate the feasibility of different approaches.",
            
            "future developments": f"Analyze comprehensively the potential future developments in {focus} within the field of {topic}. Consider emerging trends, technological advances, and anticipated changes. Include specific examples and evaluate the implications of these developments."
        }
        
        return templates.get(focus.lower(), f"Provide a comprehensive analysis and discussion of {focus} in the context of {topic} and {main_topic}. Include detailed explanations, specific examples, and critical evaluation of key aspects.")

    def _create_comprehensive_template_model_answer(self, question_text: str, focus: str, 
                                                  topic: str, marks: int) -> str:
        """Create comprehensive template-based model answer in academic style"""
        base_answer = f"**Comprehensive Model Answer: {focus} in {topic}**\n\n"
        
        # Introduction
        base_answer += f"A comprehensive understanding of {focus} within {topic} requires examination of multiple interconnected aspects. This analysis will explore the fundamental concepts, practical applications, and critical implications.\n\n"
        
        # Main content based on marks
        if marks >= 6:
            base_answer += f"**1. Fundamental Concepts and Definitions**\n"
            base_answer += f"The core concepts of {focus} encompass several key elements:\n"
            base_answer += f"• Clear and precise definitions of essential terminology\n"
            base_answer += f"• Understanding of underlying principles and theoretical foundations\n"
            base_answer += f"• Recognition of the relationship between {focus} and broader {topic} concepts\n\n"
            
            base_answer += f"**2. Practical Applications and Examples**\n"
            base_answer += f"The practical significance of {focus} in {topic} is demonstrated through:\n"
            base_answer += f"• Real-world implementations and use cases\n"
            base_answer += f"• Specific examples that illustrate key principles\n"
            base_answer += f"• Analysis of effectiveness in various contexts\n\n"
            
            base_answer += f"**3. Analysis and Relationships**\n"
            base_answer += f"The interconnected nature of {focus} within {topic} involves:\n"
            base_answer += f"• Examination of relationships between different elements\n"
            base_answer += f"• Analysis of cause-and-effect relationships\n"
            base_answer += f"• Understanding of systemic implications\n\n"
        
        if marks >= 9:
            base_answer += f"**4. Critical Evaluation and Assessment**\n"
            base_answer += f"A critical assessment of {focus} reveals:\n"
            base_answer += f"• Strengths and advantages in practical applications\n"
            base_answer += f"• Limitations and constraints that must be considered\n"
            base_answer += f"• Comparative analysis with alternative approaches\n"
            base_answer += f"• Evidence-based conclusions drawn from analysis\n\n"
            
            base_answer += f"**5. Real-World Applications and Impact**\n"
            base_answer += f"The practical impact of {focus} in {topic} includes:\n"
            base_answer += f"• Specific industry applications and implementations\n"
            base_answer += f"• Analysis of effectiveness and outcomes\n"
            base_answer += f"• Discussion of benefits and challenges encountered\n\n"
        
        if marks >= 12:
            base_answer += f"**6. Synthesis and Integration**\n"
            base_answer += f"The integration of {focus} concepts demonstrates:\n"
            base_answer += f"• Synthesis of multiple concepts and principles\n"
            base_answer += f"• Integration of theoretical knowledge with practical applications\n"
            base_answer += f"• Evidence of comprehensive understanding across different contexts\n\n"
            
            base_answer += f"**7. Future Perspectives and Developments**\n"
            base_answer += f"Looking toward future developments in {focus}:\n"
            base_answer += f"• Emerging trends and technological advances\n"
            base_answer += f"• Anticipated changes and their implications\n"
            base_answer += f"• Potential opportunities and challenges ahead\n\n"
            
            base_answer += f"**8. Original Insights and Advanced Analysis**\n"
            base_answer += f"Advanced understanding is demonstrated through:\n"
            base_answer += f"• Original thinking and innovative perspectives\n"
            base_answer += f"• Deep insights into complex relationships\n"
            base_answer += f"• Evidence of critical thinking and analytical skills\n\n"
        
        # Conclusion
        base_answer += f"**Conclusion**\n"
        base_answer += f"In conclusion, {focus} represents a fundamental aspect of {topic} that requires comprehensive understanding across multiple dimensions. The effective application of these concepts depends on thorough knowledge of underlying principles, practical implementation strategies, and critical evaluation of outcomes. Students demonstrating mastery of this topic should show evidence of deep understanding, practical application knowledge, and critical analytical thinking appropriate for university-level study.\n\n"
        
        base_answer += f"**Key Assessment Areas for {marks} marks:**\n"
        base_answer += f"• Demonstrate comprehensive understanding of {focus} concepts\n"
        base_answer += f"• Show clear relevance to {topic} with specific, detailed examples\n"
        base_answer += f"• Provide appropriate depth and analytical detail for {marks} marks\n"
        base_answer += f"• Use proper academic terminology and demonstrate critical thinking\n"
        base_answer += f"• Show evidence of synthesis and integration of knowledge\n"
        
        return base_answer

    def _create_comprehensive_template_marking_scheme(self, focus: str, marks: int) -> Dict[str, Any]:
        """Create comprehensive template-based marking scheme in academic style"""
        # Create detailed criteria based on marks and focus
        criteria = []
        remaining_marks = marks
        
        # Distribute marks comprehensively
        if marks >= 10:
            # For higher mark questions, use more detailed breakdown
            understanding_marks = max(2, marks // 4)
            criteria.append({
                "criterion": f"Comprehensive understanding and knowledge of {focus} concepts, including clear definitions and theoretical foundations",
                "marks": understanding_marks
            })
            remaining_marks -= understanding_marks
            
            explanation_marks = max(2, marks // 4)
            criteria.append({
                "criterion": "Detailed explanations, logical structure, and academic presentation with appropriate terminology",
                "marks": explanation_marks
            })
            remaining_marks -= explanation_marks
            
            application_marks = max(2, remaining_marks // 2)
            criteria.append({
                "criterion": "Relevant examples, practical applications, and real-world connections with specific details",
                "marks": application_marks
            })
            remaining_marks -= application_marks
            
            if remaining_marks > 0:
                criteria.append({
                    "criterion": "Critical analysis, evaluation, synthesis of ideas, and evidence of original thinking",
                    "marks": remaining_marks
                })
        else:
            # For lower mark questions, use simpler breakdown
            understanding_marks = max(2, marks // 2)
            criteria.append({
                "criterion": f"Understanding and knowledge of {focus} concepts with clear explanations",
                "marks": understanding_marks
            })
            remaining_marks -= understanding_marks
            
            if remaining_marks > 0:
                criteria.append({
                    "criterion": "Application, examples, and analysis appropriate to the question requirements",
                    "marks": remaining_marks
                })
        
        # Create comprehensive detailed scheme text
        scheme_text = f"**Comprehensive Marking Scheme for {focus} ({marks} marks)**\n\n"
        
        scheme_text += f"**Overall Marking Approach:**\n"
        scheme_text += f"This marking scheme assesses comprehensive understanding of {focus} through multiple criteria. "
        scheme_text += f"Award marks based on demonstration of knowledge, clarity of explanation, use of relevant examples, "
        scheme_text += f"and depth of analysis appropriate for university level.\n\n"
        
        scheme_text += f"**Detailed Marking Criteria:**\n\n"
        
        for i, criterion in enumerate(criteria, 1):
            scheme_text += f"{i}. **{criterion['criterion']}**: {criterion['marks']} marks\n"
            
            # Add detailed guidance based on marks
            if criterion['marks'] >= 4:
                scheme_text += f"   - **Excellent (Full marks)**: Comprehensive coverage with detailed explanations, specific examples, and clear understanding\n"
                scheme_text += f"   - **Good ({criterion['marks']-1}-{criterion['marks']} marks)**: Good understanding with adequate detail and relevant examples\n"
                scheme_text += f"   - **Satisfactory ({max(1, criterion['marks']//2)}-{criterion['marks']-2} marks)**: Basic understanding with some explanation but limited detail\n"
                scheme_text += f"   - **Poor (0-{max(1, criterion['marks']//2-1)} marks)**: Minimal understanding or incorrect information\n\n"
            elif criterion['marks'] >= 2:
                scheme_text += f"   - **Full marks**: Clear understanding with appropriate explanation\n"
                scheme_text += f"   - **Partial marks**: Some understanding but incomplete or unclear explanation\n"
                scheme_text += f"   - **No marks**: No evidence of understanding or incorrect information\n\n"
        
        scheme_text += f"**Key Concepts and Examples that should be covered:**\n"
        scheme_text += f"• Fundamental definitions and terminology related to {focus}\n"
        scheme_text += f"• Theoretical principles and underlying concepts\n"
        scheme_text += f"• Practical applications and real-world examples\n"
        scheme_text += f"• Analysis of relationships and implications\n"
        scheme_text += f"• Critical evaluation where appropriate\n\n"
        
        scheme_text += f"**Partial Marking Guidance:**\n"
        scheme_text += f"• Award partial marks for correct concepts even if explanation is incomplete\n"
        scheme_text += f"• Credit relevant examples even if not perfectly explained\n"
        scheme_text += f"• Recognize good understanding demonstrated through alternative approaches\n"
        scheme_text += f"• Consider the overall quality and coherence of the response\n\n"
        
        scheme_text += f"**Quality Indicators:**\n"
        scheme_text += f"• **Excellent responses** demonstrate comprehensive understanding, detailed explanations, relevant examples, and critical thinking\n"
        scheme_text += f"• **Good responses** show clear understanding with adequate detail and appropriate examples\n"
        scheme_text += f"• **Satisfactory responses** demonstrate basic understanding with some explanation and limited examples\n"
        scheme_text += f"• **Poor responses** show minimal understanding, incorrect information, or fail to address the question\n\n"
        
        scheme_text += f"**Total: {marks} marks**\n\n"
        scheme_text += f"**Final Assessment Note:**\n"
        scheme_text += f"Award marks based on evidence of understanding, clarity of communication, use of appropriate examples, "
        scheme_text += f"and depth of analysis suitable for university-level study. Consider the overall quality and coherence "
        scheme_text += f"of the response while following the specific criteria outlined above."
        
        return {
            "detailed_scheme": scheme_text,
            "mark_breakdown": criteria,
            "total_marks": marks,
            "generated_by": "comprehensive_enhanced_template",
            "academic_style": True
        }

    # Include all existing helper methods from the previous implementation
    def _load_embeddings_data(self) -> List[Dict]:
        """Load embeddings data from the processed file"""
        try:
            embeddings_path = Path("data/output/processed/embeddings.json")
            if not embeddings_path.exists():
                logger.error("❌ Embeddings file not found")
                return []
            return self.embedding_gen.load_embeddings_data(str(embeddings_path))
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings data: {e}")
            return []

    def _cluster_content_by_similarity(self, content: List[Dict], 
                                     similarity_threshold: float = 0.6) -> List[List[Dict]]:
        """Enhanced content clustering for better question distribution"""
        try:
            if len(content) < 2:
                return [content]
            
            clusters = self.embedding_gen.cluster_similar_chunks(content, similarity_threshold)
            
            if not clusters or any(len(cluster) > 8 for cluster in clusters):
                clusters = self._create_balanced_clusters(content, max_cluster_size=6)
            
            if len(clusters) < 3 and len(content) >= 6:
                clusters = self._create_diverse_clusters(content, target_clusters=min(4, len(content)//2))
            
            logger.info(f"🔗 Created {len(clusters)} balanced content clusters")
            return clusters
        except Exception as e:
            logger.error(f"❌ Failed to cluster content: {e}")
            return [content]

    def _create_balanced_clusters(self, content: List[Dict], max_cluster_size: int = 6) -> List[List[Dict]]:
        """Create balanced clusters based on content similarity"""
        clusters = []
        remaining_content = content.copy()
        
        while remaining_content:
            current_cluster = [remaining_content.pop(0)]
            items_to_remove = []
            
            for i, candidate in enumerate(remaining_content):
                if len(current_cluster) >= max_cluster_size:
                    break
                
                similarities = []
                for cluster_item in current_cluster:
                    cluster_emb = cluster_item.get('embedding', [])
                    candidate_emb = candidate.get('embedding', [])
                    if cluster_emb and candidate_emb:
                        sim = self.embedding_gen.calculate_similarity(cluster_emb, candidate_emb)
                        similarities.append(sim)
                
                if similarities and sum(similarities) / len(similarities) > 0.5:
                    current_cluster.append(candidate)
                    items_to_remove.append(i)
            
            for i in reversed(items_to_remove):
                remaining_content.pop(i)
            
            clusters.append(current_cluster)
        
        return clusters

    def _create_diverse_clusters(self, content: List[Dict], target_clusters: int) -> List[List[Dict]]:
        """Create diverse clusters to ensure good question variety"""
        if len(content) <= target_clusters:
            return [[item] for item in content]
        
        clusters = []
        used_indices = set()
        
        # Select diverse seed items
        seed_indices = [0]
        used_indices.add(0)
        
        for _ in range(target_clusters - 1):
            best_candidate_idx = -1
            max_min_distance = 0
            
            for i, candidate in enumerate(content):
                if i in used_indices:
                    continue
                
                candidate_emb = candidate.get('embedding', [])
                if not candidate_emb:
                    continue
                
                min_distance = float('inf')
                for seed_idx in seed_indices:
                    seed_emb = content[seed_idx].get('embedding', [])
                    if seed_emb:
                        similarity = self.embedding_gen.calculate_similarity(candidate_emb, seed_emb)
                        distance = 1 - similarity
                        min_distance = min(min_distance, distance)
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate_idx = i
            
            if best_candidate_idx != -1:
                seed_indices.append(best_candidate_idx)
                used_indices.add(best_candidate_idx)
        
        # Initialize clusters with seed items
        for seed_idx in seed_indices:
            clusters.append([content[seed_idx]])
        
        # Assign remaining items to closest clusters
        for i, item in enumerate(content):
            if i in used_indices:
                continue
            
            item_emb = item.get('embedding', [])
            if not item_emb:
                smallest_cluster = min(clusters, key=len)
                smallest_cluster.append(item)
                continue
            
            best_cluster_idx = 0
            best_similarity = 0
            
            for cluster_idx, cluster in enumerate(clusters):
                cluster_similarities = []
                for cluster_item in cluster:
                    cluster_emb = cluster_item.get('embedding', [])
                    if cluster_emb:
                        sim = self.embedding_gen.calculate_similarity(item_emb, cluster_emb)
                        cluster_similarities.append(sim)
                
                if cluster_similarities:
                    avg_similarity = sum(cluster_similarities) / len(cluster_similarities)
                    if avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_cluster_idx = cluster_idx
            
            clusters[best_cluster_idx].append(item)
        
        return clusters

    def _select_best_content_cluster(self, q_structure: Dict, clusters: List[List[Dict]], 
                                   topic: str) -> List[Dict]:
        """Select the best content cluster for a specific question structure"""
        try:
            question_focus = f"{q_structure['main_topic']} {topic}"
            focus_embedding = self.embedding_gen.generate_topic_embedding(question_focus)
            
            if not focus_embedding:
                return clusters[0] if clusters else []
            
            best_cluster = []
            best_similarity = 0
            
            for cluster in clusters:
                if not cluster:
                    continue
                
                similarities = []
                for item in cluster:
                    item_embedding = item.get('embedding', [])
                    if item_embedding:
                        sim = self.embedding_gen.calculate_similarity(focus_embedding, item_embedding)
                        similarities.append(sim)
                
                if similarities:
                    avg_similarity = sum(similarities) / len(similarities)
                    if avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_cluster = cluster
            
            logger.debug(f"🎯 Selected cluster with similarity score: {best_similarity:.3f}")
            return best_cluster if best_cluster else (clusters[0] if clusters else [])
            
        except Exception as e:
            logger.error(f"❌ Failed to select best content cluster: {e}")
            return clusters[0] if clusters else []

    def _select_diverse_content_for_subparts(self, content_cluster: List[Dict], 
                                           num_subparts: int) -> List[Dict]:
        """Select diverse content pieces for different sub-parts"""
        try:
            if not content_cluster:
                return []
            
            if len(content_cluster) <= num_subparts:
                return content_cluster
            
            selected = [content_cluster[0]]
            
            for i in range(1, min(num_subparts, len(content_cluster))):
                best_candidate = None
                max_min_distance = 0
                
                for candidate in content_cluster[1:]:
                    if candidate in selected:
                        continue
                    
                    candidate_emb = candidate.get('embedding', [])
                    if not candidate_emb:
                        continue
                    
                    min_distance = float('inf')
                    for selected_item in selected:
                        selected_emb = selected_item.get('embedding', [])
                        if selected_emb:
                            similarity = self.embedding_gen.calculate_similarity(
                                candidate_emb, selected_emb
                            )
                            distance = 1 - similarity
                            min_distance = min(min_distance, distance)
                    
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_candidate = candidate
                
                if best_candidate:
                    selected.append(best_candidate)
            
            return selected
            
        except Exception as e:
            logger.error(f"❌ Failed to select diverse content: {e}")
            return content_cluster[:num_subparts]

    def _generate_marking_criteria(self, focus: str, marks: int, content_text: str) -> List[str]:
        """Generate marking criteria based on focus and content"""
        try:
            base_criteria = {
                "basic definitions": [
                    "Clear and accurate definitions with comprehensive explanations",
                    "Understanding of key concepts and terminology", 
                    "Appropriate use of academic terminology and context"
                ],
                "conceptual explanation": [
                    "Clear and detailed explanation of concepts",
                    "Logical structure and coherent flow of ideas",
                    "Evidence of deep understanding and comprehension"
                ],
                "comparison and contrast": [
                    "Comprehensive identification of similarities and differences",
                    "Balanced and detailed analysis of multiple aspects",
                    "Clear conclusions with supporting evidence"
                ],
                "practical applications": [
                    "Relevant and detailed practical examples",
                    "Understanding of application contexts and implications",
                    "Analysis of effectiveness and real-world relevance"
                ],
                "critical evaluation": [
                    "Critical thinking and analytical reasoning demonstrated",
                    "Balanced evaluation with multiple perspectives considered",
                    "Evidence-based conclusions with supporting arguments",
                    "Consideration of multiple viewpoints and implications"
                ]
            }
            
            criteria = base_criteria.get(focus.lower(), [
                "Understanding of topic with clear explanations",
                "Comprehensive explanation with relevant details",
                "Appropriate examples and practical applications"
            ])
            
            if marks >= 8:
                criteria.append("Depth of analysis and detailed examination")
            if marks >= 10:
                criteria.append("Original thinking, insights, and critical evaluation")
            if marks >= 12:
                criteria.append("Synthesis of concepts and advanced analytical thinking")
            
            return criteria[:6]  # Return up to 6 criteria for comprehensive assessment
            
        except Exception as e:
            logger.error(f"❌ Failed to generate marking criteria: {e}")
            return ["Understanding of topic", "Clear explanation", "Relevant examples"]

    def _format_complete_exam_paper(self, exam_structure: Dict, topic: str, content_count: int) -> Dict:
        """Format exam structure into complete paper with comprehensive AI-generated content"""
        try:
            questions = exam_structure.get("questions", {})
            
            exam_metadata = {
                "title": f"AI-Enhanced Comprehensive Examination - {topic}",
                "subject": "SBS4115 Fundamentals of AI & Data Analytics",
                "topic": topic,
                "total_questions": len(questions),
                "total_marks": sum(q.get("total_marks", 0) for q in questions.values()),
                "time_allowed": "3 hours",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "content_sources": content_count,
                "ai_requests_made": self.api_requests_made,
                "generation_style": "comprehensive_academic"
            }
            
            instructions = [
                "Answer all questions in ALL sections.",
                "All questions carry marks as indicated.",
                f"This question paper has {len(questions)} questions.",
                "Write all answers in the answer book provided.",
                "Comprehensive AI-generated model answers and detailed marking schemes are provided separately for reference.",
                "Ensure your answers demonstrate depth of understanding appropriate for university level."
            ]
            
            formatted_questions = {}
            for q_num, question_data in questions.items():
                formatted_questions[q_num] = {
                    **question_data,
                    "formatted_question": self._format_question_display(question_data)
                }
            
            # Generate separate comprehensive documents
            question_paper = self._generate_question_paper_content(formatted_questions, exam_metadata, instructions)
            model_answers = self._generate_comprehensive_model_answers_document(formatted_questions, exam_metadata)
            marking_schemes = self._generate_comprehensive_marking_schemes_document(formatted_questions, exam_metadata)
            
            return {
                "exam_metadata": exam_metadata,
                "instructions": instructions,
                "questions": formatted_questions,
                "question_paper_content": question_paper,
                "model_answers_content": model_answers,
                "marking_schemes_content": marking_schemes,
                "generation_stats": {
                    "questions_generated": len(questions),
                    "total_marks": exam_metadata["total_marks"],
                    "content_sources_used": content_count,
                    "ai_requests_made": self.api_requests_made,
                    "ai_generated_components": ["questions", "comprehensive_model_answers", "detailed_marking_schemes"],
                    "generation_style": "comprehensive_academic"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to format complete exam paper: {e}")
            return self._create_empty_exam_structure(topic)

    def _format_question_display(self, question_data: Dict) -> str:
        """Format question for display in exam paper"""
        formatted = f"{question_data['main_topic'].upper()}\n"
        formatted += f"Total marks: {question_data['total_marks']}\n\n"
        
        for sub_q in question_data.get("sub_questions", []):
            formatted += f"{sub_q['part']} {sub_q['question']}\n"
            formatted += f"({sub_q['marks']} marks)\n\n"
        
        return formatted

    def _generate_question_paper_content(self, questions: Dict, metadata: Dict, instructions: List[str]) -> str:
        """Generate the comprehensive question paper content"""
        content = f"""COMPREHENSIVE AI-ENHANCED EXAMINATION PAPER

{metadata['title']}
Subject: {metadata['subject']}
Time Allowed: {metadata['time_allowed']}
Date: {metadata['generated_at'].split()[0]}

INSTRUCTIONS TO CANDIDATES:
"""
        
        for i, instruction in enumerate(instructions, 1):
            content += f"{i}. {instruction}\n"
        
        content += "\nDO NOT TURN THIS PAGE OVER UNTIL YOU ARE TOLD TO DO SO\n\n"
        content += "=" * 50 + "\n\n"
        
        for q_num, question_data in questions.items():
            content += f"{q_num}. {question_data.get('formatted_question', '')}\n"
            content += "-" * 30 + "\n\n"
        
        content += "END OF EXAMINATION PAPER"
        return content

    def _generate_comprehensive_model_answers_document(self, questions: Dict, metadata: Dict) -> str:
        """Generate separate comprehensive AI-powered model answers document"""
        content = f"""COMPREHENSIVE AI-GENERATED MODEL ANSWERS

{metadata['title']}
Subject: {metadata['subject']}
Generated: {metadata['generated_at']}
API Requests Used: {metadata.get('ai_requests_made', 0)}
Generation Style: Comprehensive Academic

INSTRUCTIONS FOR MARKERS:
- These are comprehensive AI-generated model answers using Gemini AI
- Model answers follow academic style with detailed explanations and structured content
- Use these as reference for consistent marking across all scripts
- Award partial marks for correct concepts even if wording differs from model answers
- Refer to the detailed marking schemes for specific mark allocation guidance
- Model answers demonstrate the expected depth, structure, and quality for full marks
- Answers include comprehensive coverage, relevant examples, and critical analysis
- Students should demonstrate similar depth and academic rigor in their responses

"""
        
        for q_num, question_data in questions.items():
            content += f"\n{q_num}. {question_data.get('main_topic', 'Question').upper()}\n"
            content += f"Total marks: {question_data.get('total_marks', 0)}\n\n"
            
            sub_questions = question_data.get('sub_questions', [])
            for sub_q in sub_questions:
                content += f"**{sub_q.get('part', '')} {sub_q.get('question', '')}**\n"
                content += f"**({sub_q.get('marks', 0)} marks)**\n\n"
                content += f"**COMPREHENSIVE AI-GENERATED MODEL ANSWER:**\n\n{sub_q.get('model_answer', 'Not available')}\n\n"
                content += "=" * 60 + "\n\n"
        
        content += "\nEND OF COMPREHENSIVE AI-GENERATED MODEL ANSWERS"
        return content

    def _generate_comprehensive_marking_schemes_document(self, questions: Dict, metadata: Dict) -> str:
        """Generate separate comprehensive AI-powered marking schemes document"""
        content = f"""COMPREHENSIVE AI-GENERATED MARKING SCHEMES

{metadata['title']}
Subject: {metadata['subject']}
Generated: {metadata['generated_at']}
API Requests Used: {metadata.get('ai_requests_made', 0)}
Generation Style: Comprehensive Academic

INSTRUCTIONS FOR MARKERS:
- These are comprehensive AI-generated detailed marking schemes using Gemini AI
- Marking schemes provide detailed criteria and guidance for consistent assessment
- Award marks according to the specific criteria outlined for each question part
- Partial marks can be awarded for partially correct answers - use professional judgment
- Ensure consistency across all answer scripts using these detailed guidelines
- Consider the overall quality and coherence of student responses
- Use the quality indicators provided to assess different levels of achievement
- Refer to model answers for examples of full-mark responses

GENERAL MARKING PRINCIPLES:
- Reward understanding demonstrated through clear explanations
- Credit relevant examples and practical applications appropriately
- Recognize good analysis and critical thinking skills
- Award marks for logical structure and academic presentation
- Consider alternative valid approaches and interpretations

"""
        
        for q_num, question_data in questions.items():
            content += f"\n{q_num}. **{question_data.get('main_topic', 'Question').upper()}**\n"
            content += f"**Total marks: {question_data.get('total_marks', 0)}**\n\n"
            
            sub_questions = question_data.get('sub_questions', [])
            for sub_q in sub_questions:
                content += f"**{sub_q.get('part', '')} {sub_q.get('question', '')}**\n"
                content += f"**({sub_q.get('marks', 0)} marks)**\n\n"
                
                marking_scheme = sub_q.get('detailed_marking_scheme', {})
                content += f"**COMPREHENSIVE AI-GENERATED MARKING SCHEME:**\n\n{marking_scheme.get('detailed_scheme', 'Not available')}\n\n"
                
                breakdown = marking_scheme.get('mark_breakdown', [])
                if breakdown:
                    content += "**SUMMARY OF MARK BREAKDOWN:**\n"
                    for item in breakdown:
                        content += f"• {item['criterion']}: **{item['marks']} marks**\n"
                    content += "\n"
                
                content += "=" * 60 + "\n\n"
        
        content += "\nEND OF COMPREHENSIVE AI-GENERATED MARKING SCHEMES"
        return content

    def save_multi_format_exam(self, exam_data: Dict, output_dir: Path,
                            formats: List[str] = None) -> List[str]:
        """Save comprehensive AI-enhanced exam in multiple formats including MD and PDF"""
        try:
            if formats is None:
                formats = ['txt', 'json', 'md', 'pdf']  # Updated default
            
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_files = []

            # Define content components
            components = {
                'questions': exam_data.get('question_paper_content', ''),
                'answers': exam_data.get('model_answers_content', ''),
                'schemes': exam_data.get('marking_schemes_content', '')
            }

            for format_type in formats:
                format_lower = format_type.lower()

                if format_lower == 'txt':
                    # Existing TXT saving logic
                    for comp_name, content in components.items():
                        file_name = f"comprehensive_ai_{comp_name}_{timestamp}.txt"
                        file_path = output_dir / file_name
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        saved_files.append(str(file_path))
                    logger.info("✅ Saved TXT files")

                elif format_lower == 'json':
                    # Existing JSON saving logic
                    json_file = output_dir / f"comprehensive_ai_complete_exam_{timestamp}.json"
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(exam_data, f, indent=2, ensure_ascii=False)
                    saved_files.append(str(json_file))
                    logger.info("✅ Saved JSON file")

                elif format_lower == 'md':
                    # NEW: Save as Markdown
                    for comp_name, content in components.items():
                        md_content = self._convert_to_markdown(content, comp_name)
                        file_name = f"comprehensive_ai_{comp_name}_{timestamp}.md"
                        file_path = output_dir / file_name
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(md_content)
                        saved_files.append(str(file_path))
                    logger.info("✅ Saved MD files")

                elif format_lower == 'pdf':
                    # NEW: Convert MD to PDF
                    for comp_name, content in components.items():
                        md_content = self._convert_to_markdown(content, comp_name)
                        file_name = f"comprehensive_ai_{comp_name}_{timestamp}"
                        pdf_path = output_dir / f"{file_name}.pdf"
                        
                        # Convert MD to PDF using weasyprint
                        html_content = markdown.markdown(md_content)
                        HTML(string=html_content).write_pdf(str(pdf_path))
                        
                        saved_files.append(str(pdf_path))
                    logger.info("✅ Saved PDF files")

            return saved_files
        except Exception as e:
            logger.error(f"❌ Failed to save exam: {e}")
            return []


    def _convert_to_markdown(self, text: str, component_type: str) -> str:
        """Convert plain text content to enhanced Markdown format."""
        if not text:
            return f"# {component_type.title()}\n\nNo content available."
        
        # Create appropriate header based on component type
        headers = {
            'questions': 'Examination Questions',
            'answers': 'Model Answers', 
            'schemes': 'Marking Schemes'
        }
        
        header = headers.get(component_type, component_type.title())
        md_text = f"# {header}\n\n"
        
        # Enhanced text processing for better markdown
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                md_text += "\n"
                continue
                
            # Convert question numbers to headers
            if line.startswith('Q') and any(char.isdigit() for char in line[:5]):
                md_text += f"\n## {line}\n\n"
            # Convert sub-questions to subheaders
            elif line.startswith(('(a)', '(b)', '(c)', '(i)', '(ii)', '(iii)')):
                md_text += f"\n### {line}\n\n"
            # Convert marks indicators
            elif 'marks' in line.lower() and '(' in line:
                md_text += f"\n**{line}**\n\n"
            # Convert bullet points
            elif line.startswith('•'):
                md_text += f"- {line[1:].strip()}\n"
            # Regular text
            else:
                md_text += f"{line}\n"
        
        return md_text


    def _create_empty_exam_structure(self, topic: str) -> Dict:
        """Create empty exam structure when content is not available"""
        return {
            "exam_metadata": {
                "title": f"Comprehensive AI-Enhanced Examination Paper - {topic}",
                "subject": topic,
                "total_questions": 0,
                "total_marks": 0,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": "No content available for comprehensive exam generation"
            },
            "instructions": [],
            "questions": {},
            "question_paper_content": "Error: No content available for comprehensive exam generation",
            "model_answers_content": "Error: No comprehensive AI model answers could be generated",
            "marking_schemes_content": "Error: No comprehensive AI marking schemes could be generated"
        }

    def _create_fallback_question(self, q_num: str, q_structure: Dict, topic: str) -> Dict:
        """Create fallback question when comprehensive AI generation fails"""
        return {
            "question_number": q_num,
            "main_topic": q_structure["main_topic"],
            "total_marks": q_structure["total_marks"],
            "sub_questions": [
                {
                    "part": sub_part["part"],
                    "question": self._create_template_question(sub_part['focus'], topic, q_structure["main_topic"]),
                    "marks": sub_part["marks"],
                    "focus": sub_part["focus"],
                    "model_answer": self._create_comprehensive_template_model_answer("", sub_part['focus'], topic, sub_part['marks']),
                    "detailed_marking_scheme": self._create_comprehensive_template_marking_scheme(sub_part['focus'], sub_part['marks']),
                    "basic_marking_criteria": [f"Comprehensive understanding of {sub_part['focus']}"],
                    "source_content": "Comprehensive fallback generation",
                    "similarity_score": 0
                }
                for sub_part in q_structure["sub_parts"]
            ],
            "content_sources": ["Comprehensive fallback generation"]
        }

    # Template-only generation method for quota exhaustion scenarios
    def generate_template_only_exam(self, topic: str = "AI and Data Analytics") -> Dict:
        """Generate comprehensive exam using only templates when API quota is exhausted"""
        logger.info("🔄 Generating comprehensive template-only exam (no API calls)")
        
        # Load content for context
        embeddings_data = self._load_embeddings_data()
        if not embeddings_data:
            return self._create_empty_exam_structure(topic)
        
        # Use comprehensive template generation
        exam_structure = {
            "title": f"Comprehensive Template-Enhanced Examination - {topic}",
            "topic": topic,
            "total_questions": 4,
            "total_marks": 100,
            "questions": {}
        }
        
        for q_num, q_structure in self.standard_structure.items():
            question_data = {
                "question_number": q_num,
                "main_topic": q_structure["main_topic"],
                "total_marks": q_structure["total_marks"],
                "sub_questions": [],
                "content_sources": ["Comprehensive template generation"]
            }
            
            for sub_part in q_structure["sub_parts"]:
                sub_question = {
                    "part": sub_part["part"],
                    "question": self._create_template_question(
                        sub_part["focus"], topic, q_structure["main_topic"]
                    ),
                    "marks": sub_part["marks"],
                    "focus": sub_part["focus"],
                    "model_answer": self._create_comprehensive_template_model_answer(
                        "", sub_part["focus"], topic, sub_part["marks"]
                    ),
                    "detailed_marking_scheme": self._create_comprehensive_template_marking_scheme(
                        sub_part["focus"], sub_part["marks"]
                    ),
                    "basic_marking_criteria": self._generate_marking_criteria(
                        sub_part["focus"], sub_part["marks"], ""
                    ),
                    "source_content": "Comprehensive template-based generation",
                    "similarity_score": 0
                }
                question_data["sub_questions"].append(sub_question)
            
            exam_structure["questions"][q_num] = question_data
        
        return self._format_complete_exam_paper(exam_structure, topic, len(embeddings_data))
