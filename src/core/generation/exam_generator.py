import json
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger
from pathlib import Path
import google.generativeai as genai
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
            
            # Generate questions one by one
            for i in range(min(num_questions, len(content_chunks))):
                chunk = content_chunks[i % len(content_chunks)]
                q_type = question_types[i % len(question_types)]
                
                logger.info(f"📝 Generating question {i+1}/{num_questions} ({q_type})")
                
                if q_type == "multiple_choice":
                    question = self._generate_mc_simple(chunk, topic, difficulty)
                else:
                    question = self._generate_sa_simple(chunk, topic, difficulty)
                
                if question and question.get('question'):
                    questions.append(question)
                    logger.info(f"✅ Successfully generated question {i+1}")
                else:
                    logger.warning(f"⚠️ Failed to generate question {i+1}")
            
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
        """Generate multiple choice question with simpler approach"""
        
        # Limit content length for API
        content_snippet = content[:800] if len(content) > 800 else content
        
        prompt = f"""Create a {difficulty} multiple choice question based on this content:

{content_snippet}

Requirements:
- Question should test understanding of the content
- Provide 4 options (A, B, C, D)
- Only one option should be correct
- Include brief explanation

Format:
Question: [question text]
A) [option A]
B) [option B]
C) [option C]
D) [option D]
Answer: [A, B, C, or D]
Explanation: [brief explanation]"""

        try:
            response = self.generation_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=500,
                )
            )
            
            return self._parse_mc_simple(response.text)
            
        except Exception as e:
            logger.error(f"❌ MC generation failed: {e}")
            return {}
    
    def _generate_sa_simple(self, content: str, topic: str, difficulty: str) -> Dict:
        """Generate short answer question with simpler approach"""
        
        content_snippet = content[:800] if len(content) > 800 else content
        
        prompt = f"""Create a {difficulty} short answer question based on this content:

{content_snippet}

Requirements:
- Question should require explanation or analysis
- Provide key points for the answer
- Assign appropriate marks

Format:
Question: [question text]
Marks: [number]
Key Points: [bullet points of main answer elements]"""

        try:
            response = self.generation_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=400,
                )
            )
            
            return self._parse_sa_simple(response.text)
            
        except Exception as e:
            logger.error(f"❌ SA generation failed: {e}")
            return {}
    
    def _parse_mc_simple(self, response: str) -> Dict:
        """Simple parsing for multiple choice"""
        try:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            question = ""
            options = []
            answer = ""
            explanation = ""
            
            for line in lines:
                if line.lower().startswith('question:'):
                    question = line.split(':', 1)[1].strip()
                elif line.startswith(('A)', 'B)', 'C)', 'D)')):
                    options.append(line)
                elif line.lower().startswith('answer:'):
                    answer = line.split(':', 1)[1].strip()
                elif line.lower().startswith('explanation:'):
                    explanation = line.split(':', 1)[1].strip()
            
            if question and len(options) >= 4:
                return {
                    "type": "multiple_choice",
                    "question": question,
                    "options": options[:4],
                    "correct_answer": answer,
                    "explanation": explanation
                }
            else:
                logger.warning("⚠️ Incomplete MC question parsed")
                return {}
                
        except Exception as e:
            logger.error(f"❌ MC parsing failed: {e}")
            return {}
    
    def _parse_sa_simple(self, response: str) -> Dict:
        """Simple parsing for short answer"""
        try:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            question = ""
            marks = "5"
            key_points = ""
            
            for line in lines:
                if line.lower().startswith('question:'):
                    question = line.split(':', 1)[1].strip()
                elif line.lower().startswith('marks:'):
                    marks = line.split(':', 1)[1].strip()
                elif line.lower().startswith('key points:'):
                    key_points = line.split(':', 1)[1].strip()
            
            if question:
                return {
                    "type": "short_answer",
                    "question": question,
                    "marks": marks,
                    "key_points": key_points
                }
            else:
                logger.warning("⚠️ Incomplete SA question parsed")
                return {}
                
        except Exception as e:
            logger.error(f"❌ SA parsing failed: {e}")
            return {}
    
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
                "generation_method": "Simplified Gemini API",
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
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
