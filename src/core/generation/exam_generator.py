import json
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger
from pathlib import Path
import google.generativeai as genai
import re
from datetime import datetime
from ..embedding.gemini_client import GeminiClient

class ExamGenerator:
    def __init__(self):
        """Initialize exam generator"""
        self.gemini_client = GeminiClient()
        genai.configure(api_key=self.gemini_client.api_key)
        self.generation_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info("✅ Exam generator initialized")

    def generate_exam(self, topic: str = None, num_questions: int = 10,
                     difficulty: str = "intermediate", question_types: List[str] = None):
        """Generate exam questions from processed content"""
        try:
            logger.info(f"🔄 Generating exam: {topic} ({num_questions} questions)")
            
            # Load content chunks
            content_chunks = self._load_content_simple()
            
            if not content_chunks:
                logger.error("❌ No content found")
                return self._empty_exam_structure(topic, difficulty)
            
            logger.info(f"📊 Found {len(content_chunks)} content chunks")
            
            # Generate questions
            questions = []
            question_types = question_types or ["multiple_choice", "short_answer"]
            
            # Generate questions with enhanced validation
            for i in range(min(num_questions, len(content_chunks) * 3)):
                chunk = content_chunks[i % len(content_chunks)]
                q_type = question_types[i % len(question_types)]
                
                logger.info(f"📝 Generating question {i+1}/{num_questions} ({q_type})")
                
                if q_type == "multiple_choice":
                    question = self._generate_mc_simple(chunk, topic, difficulty)
                else:
                    question = self._generate_sa_simple(chunk, topic, difficulty)
                
                # Enhanced validation - check for parsing failures
                if question and question.get('question') and question.get('question') not in ["**", "", "None"]:
                    question_text = question.get('question', '')
                    if len(question_text.strip()) > 15:
                        questions.append(question)
                        logger.info(f"✅ Successfully generated question {len(questions)}")
                    else:
                        logger.warning(f"⚠️ Question quality check failed for question {i+1}")
                else:
                    logger.warning(f"⚠️ Failed to generate question {i+1} - retrying with different approach")
                    
                    # Retry with simplified prompt if parsing failed
                    if q_type == "multiple_choice":
                        retry_question = self._generate_mc_fallback(chunk, topic, difficulty)
                    else:
                        retry_question = self._generate_sa_fallback(chunk, topic, difficulty)
                        
                    if retry_question and retry_question.get('question'):
                        questions.append(retry_question)
                        logger.info(f"✅ Retry successful for question {len(questions)}")
                
                # Stop if we have enough questions
                if len(questions) >= num_questions:
                    break
            
            # Create exam structure
            exam = self._create_exam_structure(questions, topic, difficulty)
            logger.info(f"✅ Generated exam with {len(questions)} questions")
            
            return exam
            
        except Exception as e:
            logger.error(f"❌ Exam generation failed: {e}")
            raise

    def _load_content_simple(self) -> List[str]:
        """Simple content loading from embeddings file"""
        try:
            embeddings_file = Path("data/output/processed/embeddings.json")
            
            if not embeddings_file.exists():
                logger.error(f"❌ File not found: {embeddings_file}")
                return []
            
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract text chunks - handle different possible key names
            chunks = []
            for item in data:
                # Try different possible keys for the text content
                text = (item.get('chunk') or 
                       item.get('chunk_text') or 
                       item.get('content') or 
                       item.get('text') or '')
                
                if text and len(text.strip()) > 50:  # Only use substantial chunks
                    chunks.append(text.strip())
            
            logger.info(f"📄 Extracted {len(chunks)} text chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Failed to load content: {e}")
            return []

    def _generate_mc_simple(self, content: str, topic: str, difficulty: str) -> Dict:
        """Generate multiple choice with improved prompting for consistent format"""
        content_snippet = content[:800] if len(content) > 800 else content
        
        prompt = f"""Based on this educational content, create a {difficulty}-level multiple choice question about {topic}.

CONTENT:
{content_snippet}

STRICT FORMAT - Follow this exact structure:

Question: What is the main concept being discussed?
A) First option
B) Second option  
C) Third option
D) Fourth option
Answer: B
Explanation: Brief explanation why B is correct.

IMPORTANT REQUIREMENTS:
- Question must end with a question mark
- Do NOT use asterisks (*) or special formatting
- Provide exactly 4 options labeled A) B) C) D)
- Only one correct answer
- Include explanation
- Base question directly on the provided content
- Use plain text only, no markdown formatting"""

        try:
            response = self.generation_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for more consistent formatting
                    max_output_tokens=600,
                )
            )
            
            return self._parse_mc_simple(response.text)
            
        except Exception as e:
            logger.error(f"❌ MC generation failed: {e}")
            return {}

    def _generate_sa_simple(self, content: str, topic: str, difficulty: str) -> Dict:
        """Generate short answer with improved prompting for consistent format"""
        content_snippet = content[:800] if len(content) > 800 else content
        
        prompt = f"""Based on this educational content, create a {difficulty}-level short answer question about {topic}.

CONTENT:
{content_snippet}

STRICT FORMAT - Follow this exact structure:

Question: Explain the key concepts discussed in the content.
Marks: 8
Key Points: Main points that should be covered in a complete answer.

IMPORTANT REQUIREMENTS:
- Question must be clear and specific
- Do NOT use asterisks (*) or special formatting
- Assign appropriate marks (5-15)
- Provide key points for marking
- Question should require explanation, not just facts
- Use plain text only, no markdown formatting"""

        try:
            response = self.generation_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for consistency
                    max_output_tokens=500,
                )
            )
            
            return self._parse_sa_simple(response.text)
            
        except Exception as e:
            logger.error(f"❌ SA generation failed: {e}")
            return {}

    def _parse_mc_simple(self, response: str) -> Dict:
        """Improved parsing for multiple choice questions with flexible detection"""
        try:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            question = ""
            options = []
            answer = ""
            explanation = ""
            
            for i, line in enumerate(lines):
                line_lower = line.lower()
                
                # Enhanced question detection - try multiple patterns
                if any(starter in line_lower for starter in ['question:', 'q:', 'question -']):
                    question = line.split(':', 1)[-1].strip()
                    # Remove any leading/trailing asterisks or formatting
                    question = re.sub(r'^\*+|\*+$', '', question).strip()
                    if not question and i + 1 < len(lines):
                        question = lines[i + 1].strip()
                        question = re.sub(r'^\*+|\*+$', '', question).strip()
                
                # Enhanced option detection - flexible patterns
                elif re.match(r'^[A-Da-d][).]', line) or line.startswith(('A)', 'B)', 'C)', 'D)')):
                    options.append(line)
                
                # Enhanced answer detection
                elif any(starter in line_lower for starter in ['answer:', 'correct:', 'solution:']):
                    answer_text = line.split(':', 1)[-1].strip()
                    answer_match = re.search(r'[A-Da-d]', answer_text)
                    if answer_match:
                        answer = answer_match.group().upper()
                
                # Enhanced explanation detection
                elif any(starter in line_lower for starter in ['explanation:', 'because:', 'reason:']):
                    explanation = line.split(':', 1)[-1].strip()
                    explanation = re.sub(r'^\*+|\*+$', '', explanation).strip()
            
            # Fallback question extraction - look for lines with question marks
            if not question or question == "**":
                for line in lines:
                    cleaned_line = re.sub(r'^\*+|\*+$', '', line).strip()
                    if len(cleaned_line) > 20 and '?' in cleaned_line:
                        question = cleaned_line
                        break
            
            # Final cleanup - remove asterisks that might be formatting artifacts
            question = re.sub(r'\*+', '', question).strip()
            
            logger.info(f"🔍 Enhanced MC Parse: question='{question[:50]}...', options={len(options)}")
            
            if question and len(options) >= 4 and question != "**":
                return {
                    "type": "multiple_choice",
                    "question": question.strip(),
                    "options": options[:4],
                    "correct_answer": answer or "A",
                    "explanation": explanation.strip()
                }
            else:
                logger.warning(f"⚠️ MC parsing failed: question='{question}', options={len(options)}")
                return {}
                
        except Exception as e:
            logger.error(f"❌ MC parsing error: {e}")
            return {}

    def _parse_sa_simple(self, response: str) -> Dict:
        """Simple parsing for short answer with improved logic"""
        try:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            question = ""
            marks = "5"
            key_points = ""
            
            # More flexible parsing
            for line in lines:
                line_lower = line.lower()
                if line_lower.startswith('question:'):
                    question = line.split(':', 1)[1].strip()
                    # Remove any asterisks or formatting artifacts
                    question = re.sub(r'^\*+|\*+$', '', question).strip()
                elif line_lower.startswith('marks:'):
                    marks = line.split(':', 1)[1].strip()
                elif line_lower.startswith('key points:') or line_lower.startswith('answer:'):
                    key_points = line.split(':', 1)[1].strip()
                elif question and not marks and line.strip().isdigit():
                    marks = line.strip()
            
            # Fallback question extraction
            if not question or question == "**":
                for line in lines:
                    cleaned_line = re.sub(r'^\*+|\*+$', '', line).strip()
                    if len(cleaned_line) > 20 and not any(skip in cleaned_line.lower() for skip in ['marks:', 'points:', 'answer:']):
                        question = cleaned_line
                        break
            
            # Debug logging
            logger.info(f"🔍 Parsed SA: question='{question[:50]}...', marks='{marks}', key_points='{key_points[:50]}...'")
            
            if question and len(question) > 10 and question != "**":
                return {
                    "type": "short_answer",
                    "question": question,
                    "marks": marks,
                    "key_points": key_points
                }
            else:
                logger.warning(f"⚠️ Incomplete SA question: question='{question}', marks='{marks}'")
                return {}
                
        except Exception as e:
            logger.error(f"❌ SA parsing failed: {e}")
            return {}

    def _generate_sa_fallback(self, content: str, topic: str, difficulty: str) -> Dict:
        """Fallback method for generating short answer questions when primary method fails"""
        try:
            content_snippet = content[:600] if len(content) > 600 else content
            
            # Simplified prompt for fallback
            prompt = f"""Create a simple {difficulty} short answer question based on this content:

{content_snippet}

Use this exact format:
Question: [Write a clear question here]
Marks: 8
Key Points: [Main points for the answer]

Keep it simple and clear."""

            response = self.generation_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=300,
                )
            )
            
            return self._parse_sa_simple(response.text)
            
        except Exception as e:
            logger.error(f"❌ SA fallback generation failed: {e}")
            return self._create_fallback_sa_question(topic)

    def _generate_mc_fallback(self, content: str, topic: str, difficulty: str) -> Dict:
        """Fallback method for generating multiple choice questions when primary method fails"""
        try:
            content_snippet = content[:600] if len(content) > 600 else content
            
            # Simplified prompt for fallback
            prompt = f"""Create a simple {difficulty} multiple choice question based on this content:

{content_snippet}

Use this exact format:
Question: [Write a clear question here]
A) [First option]
B) [Second option] 
C) [Third option]
D) [Fourth option]
Answer: B
Explanation: [Brief explanation]

Keep it simple and clear."""

            response = self.generation_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=400,
                )
            )
            
            return self._parse_mc_simple(response.text)
            
        except Exception as e:
            logger.error(f"❌ MC fallback generation failed: {e}")
            return self._create_fallback_mc_question(topic)

    def _create_fallback_sa_question(self, topic: str) -> Dict:
        """Create a basic fallback short answer question"""
        return {
            "type": "short_answer",
            "question": f"Explain the key concepts and applications related to {topic or 'the subject matter'}.",
            "marks": "8",
            "key_points": "Key concepts, practical applications, and relevant examples should be discussed."
        }

    def _create_fallback_mc_question(self, topic: str) -> Dict:
        """Create a basic fallback multiple choice question"""
        return {
            "type": "multiple_choice",
            "question": f"Which of the following best describes {topic or 'the main concept'}?",
            "options": [
                "A) A fundamental approach with broad applications",
                "B) A specialized technique for specific use cases",
                "C) An outdated method with limited relevance", 
                "D) A theoretical concept with no practical applications"
            ],
            "correct_answer": "A",
            "explanation": "This represents the most comprehensive and applicable description."
        }

    def _create_exam_structure(self, questions: List[Dict], topic: str, difficulty: str) -> Dict:
        """Create final exam structure"""
        return {
            "title": f"Generated Exam - {topic or 'General'}",
            "topic": topic,
            "difficulty": difficulty,
            "num_questions": len(questions),
            "question_types": list(set(q.get("type", "unknown") for q in questions)),
            "generated_at": self._get_timestamp(),
            "questions": questions,
            "metadata": {
                "generation_method": "Gemini API + RAG",
                "content_based": True
            }
        }

    def _empty_exam_structure(self, topic: str, difficulty: str) -> Dict:
        """Return empty exam structure when content loading fails"""
        return {
            "title": f"Generated Exam - {topic or 'General'}",
            "topic": topic,
            "difficulty": difficulty,
            "num_questions": 0,
            "question_types": [],
            "generated_at": self._get_timestamp(),
            "questions": [],
            "error": "No content available for question generation"
        }

    def _get_timestamp(self) -> str:
        """Get timestamp"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    # Additional utility methods for enhanced functionality
    
    def generate_full_paper(self, topic: str = None, num_questions: int = 20, 
                           difficulty: str = "intermediate") -> Dict:
        """Generate a complete exam paper in academic format"""
        try:
            logger.info(f"📋 Generating full exam paper: {topic} ({num_questions} questions)")
            
            # Generate more questions for a full paper
            question_distribution = {
                "multiple_choice": num_questions // 2,
                "short_answer": num_questions // 2
            }
            
            all_questions = []
            content_chunks = self._load_content_simple()
            
            if not content_chunks:
                return self._empty_exam_structure(topic, difficulty)
            
            for q_type, count in question_distribution.items():
                for i in range(count):
                    chunk = content_chunks[i % len(content_chunks)]
                    
                    if q_type == "multiple_choice":
                        question = self._generate_mc_simple(chunk, topic, difficulty)
                    else:
                        question = self._generate_sa_simple(chunk, topic, difficulty)
                    
                    if question and question.get('question'):
                        all_questions.append(question)
            
            # Format as complete exam paper
            return self._format_complete_paper(all_questions, topic, difficulty)
            
        except Exception as e:
            logger.error(f"❌ Full paper generation failed: {e}")
            raise

    def _format_complete_paper(self, questions: List[Dict], topic: str, difficulty: str) -> Dict:
        """Format questions into a complete exam paper"""
        
        # Separate questions by type
        mc_questions = [q for q in questions if q.get("type") == "multiple_choice"]
        sa_questions = [q for q in questions if q.get("type") == "short_answer"]
        
        # Generate paper header
        paper_content = f"""INSTRUCTIONS TO CANDIDATES
1. Answer all questions in ALL sections.
2. All questions carry marks as indicated.
3. This question paper has {len(questions)} questions.
4. Write all answers in the answer book provided.

DO NOT TURN THIS PAGE OVER UNTIL YOU ARE TOLD TO DO SO

Department of Construction, Environment and Engineering
SBS4115 Fundamentals of AI & Data Analytics
Generated Examination Paper
Date: {datetime.now().strftime('%d %B %Y')}
Time: 2:00 p.m. - 5:00 p.m.
Time Allowed: 3 hours

Topic Focus: {topic.title() if topic else 'Comprehensive Review'}
Difficulty Level: {difficulty.title()}

"""

        # Section A: Multiple Choice Questions
        if mc_questions:
            paper_content += "SECTION A: MULTIPLE CHOICE QUESTIONS\n"
            paper_content += "Choose the best answer for each question.\n\n"
            
            for i, q in enumerate(mc_questions, 1):
                paper_content += f"A.{i} {q['question']}\n"
                for option in q.get('options', []):
                    paper_content += f"     {option}\n"
                paper_content += "\n"
        
        # Section B: Short Answer Questions  
        if sa_questions:
            paper_content += "SECTION B: SHORT ANSWER QUESTIONS\n"
            paper_content += "Answer all questions. Show your working where appropriate.\n\n"
            
            for i, q in enumerate(sa_questions, 1):
                marks = q.get('marks', '8')
                paper_content += f"B.{i} {q['question']}\n"
                paper_content += f"     ({marks} marks)\n\n"
        
        paper_content += "\n--- END OF EXAMINATION PAPER ---"
        
        return {
            "title": f"Complete Exam Paper - {topic or 'Comprehensive'}",
            "topic": topic,
            "difficulty": difficulty,
            "total_questions": len(questions),
            "sections": {
                "multiple_choice": len(mc_questions),
                "short_answer": len(sa_questions)
            },
            "generated_at": self._get_timestamp(),
            "paper_content": paper_content,
            "questions": questions,
            "format": "complete_exam_paper"
        }
