import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from pathlib import Path
from datetime import datetime
import random

# Add new imports for multi-format support
import markdown
from markdown_pdf import MarkdownPdf, Section

from ..embedding.gemini_client import GeminiClient
from ..embedding.embedding_generator import EmbeddingGenerator

class StructureGenerator:
    def __init__(self):
        self.gemini_client = GeminiClient()
        self.embedding_gen = EmbeddingGenerator()
        
        # Define standard exam structure template
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
        
        logger.info("✅ Structure Generator initialized with multi-format output support")

    def generate_structured_exam(self, topic: str = "AI and Data Analytics",
                                structure_type: str = "standard",
                                num_main_questions: int = 4) -> Dict:
        """Generate a structured exam paper using embedding similarity"""
        try:
            logger.info(f"🔄 Generating structured exam for topic: {topic}")

            # Load embeddings data
            embeddings_data = self._load_embeddings_data()
            if not embeddings_data:
                logger.error("❌ No embeddings data available")
                return self._create_empty_exam_structure(topic)

            # Generate topic embedding for similarity matching
            topic_embedding = self.embedding_gen.generate_topic_embedding(topic)
            if not topic_embedding:
                logger.error("❌ Failed to generate topic embedding")
                return self._create_empty_exam_structure(topic)

            # Find content clusters relevant to the topic
            relevant_content = self.embedding_gen.find_similar_chunks(
                topic_embedding, embeddings_data, top_k=20, min_similarity=0.2
            )

            if not relevant_content:
                logger.error("❌ No relevant content found for the topic")
                return self._create_empty_exam_structure(topic)

            logger.info(f"📚 Found {len(relevant_content)} relevant content chunks")

            # Generate structured questions
            exam_structure = self._generate_exam_questions(
                topic, relevant_content, self.standard_structure
            )

            # Format final exam paper
            formatted_exam = self._format_complete_exam_paper(
                exam_structure, topic, len(relevant_content)
            )

            logger.info(f"✅ Generated structured exam with {len(exam_structure['questions'])} main questions")
            return formatted_exam

        except Exception as e:
            logger.error(f"❌ Structured exam generation failed: {e}")
            return self._create_empty_exam_structure(topic)

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

    def _generate_exam_questions(self, topic: str, relevant_content: List[Dict],
                               structure_template: Dict) -> Dict:
        """Generate exam questions using similarity-based content matching"""
        exam_questions = {
            "title": f"Structured Examination - {topic}",
            "topic": topic,
            "total_questions": len(structure_template),
            "total_marks": 100,
            "questions": {}
        }

        # Group content into clusters for different question types
        content_clusters = self._cluster_content_by_similarity(relevant_content)

        for q_num, q_structure in structure_template.items():
            logger.info(f"🔄 Generating {q_num}: {q_structure['main_topic']}")

            # Find best content cluster for this question type
            best_cluster = self._select_best_content_cluster(
                q_structure, content_clusters, topic
            )

            # Generate question with sub-parts
            question_data = self._create_question_with_subparts(
                q_num, q_structure, best_cluster, topic
            )

            exam_questions["questions"][q_num] = question_data

        return exam_questions

    def _cluster_content_by_similarity(self, content: List[Dict]) -> List[List[Dict]]:
        """Cluster content based on embedding similarity"""
        try:
            if len(content) < 2:
                return [content]

            # Use embedding generator's clustering method
            clusters = self.embedding_gen.cluster_similar_chunks(content, similarity_threshold=0.6)

            # If no clusters formed, create individual clusters
            if not clusters:
                clusters = [[item] for item in content]

            logger.info(f"🔗 Created {len(clusters)} content clusters for question generation")
            return clusters

        except Exception as e:
            logger.error(f"❌ Failed to cluster content: {e}")
            return [content]  # Return all content as single cluster

    def _select_best_content_cluster(self, q_structure: Dict,
                                   clusters: List[List[Dict]],
                                   topic: str) -> List[Dict]:
        """Select the best content cluster for a specific question structure"""
        try:
            # Generate embedding for question focus
            question_focus = f"{q_structure['main_topic']} {topic}"
            focus_embedding = self.embedding_gen.generate_topic_embedding(question_focus)

            if not focus_embedding:
                # Fallback to first available cluster
                return clusters[0] if clusters else []

            best_cluster = []
            best_similarity = 0

            # Find cluster with highest average similarity to question focus
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

    def _create_question_with_subparts(self, q_num: str, q_structure: Dict,
                                     content_cluster: List[Dict], topic: str) -> Dict:
        """Create a structured question with multiple sub-parts using embedding similarity"""
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

            # Generate each sub-part
            for i, sub_part in enumerate(sub_parts):
                content_item = selected_content[i % len(selected_content)] if selected_content else {}
                sub_question = self._generate_subpart_question(
                    sub_part, content_item, topic, main_topic
                )
                question_data["sub_questions"].append(sub_question)

            return question_data

        except Exception as e:
            logger.error(f"❌ Failed to create question {q_num}: {e}")
            return self._create_fallback_question(q_num, q_structure, topic)

    def _select_diverse_content_for_subparts(self, content_cluster: List[Dict],
                                           num_subparts: int) -> List[Dict]:
        """Select diverse content pieces for different sub-parts"""
        try:
            if not content_cluster:
                return []

            if len(content_cluster) <= num_subparts:
                return content_cluster

            # Select diverse items based on embedding differences
            selected = [content_cluster[0]]  # Always include the first item

            for i in range(1, min(num_subparts, len(content_cluster))):
                best_candidate = None
                max_min_distance = 0

                # Find item most different from already selected items
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
                            distance = 1 - similarity  # Convert similarity to distance
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

    def _generate_subpart_question(self, sub_part: Dict, content_item: Dict,
                                 topic: str, main_topic: str) -> Dict:
        """Generate a specific sub-part question using embedding-based content"""
        try:
            part_label = sub_part["part"]
            focus = sub_part["focus"]
            marks = sub_part["marks"]

            # Extract relevant content text
            content_text = content_item.get('chunk_text', '')
            content_snippet = content_text[:300] + "..." if len(content_text) > 300 else content_text

            # Generate question based on focus and content
            question_text = self._create_question_text(focus, topic, main_topic, content_snippet)

            # Generate marking criteria based on content
            marking_criteria = self._generate_marking_criteria(focus, marks, content_text)

            return {
                "part": part_label,
                "question": question_text,
                "marks": marks,
                "focus": focus,
                "marking_criteria": marking_criteria,
                "source_content": content_snippet,
                "similarity_score": content_item.get('similarity_score', 0)
            }

        except Exception as e:
            logger.error(f"❌ Failed to generate subpart question: {e}")
            return {
                "part": sub_part.get("part", ""),
                "question": f"Discuss {sub_part.get('focus', 'the topic')} in relation to {topic}.",
                "marks": sub_part.get("marks", 5),
                "focus": sub_part.get("focus", ""),
                "marking_criteria": [f"Understanding of {sub_part.get('focus', 'concept')}"],
                "source_content": "Content not available",
                "similarity_score": 0
            }

    def _create_question_text(self, focus: str, topic: str, main_topic: str,
                            content_snippet: str) -> str:
        """Create question text based on focus and content"""
        try:
            # Use Gemini to generate contextual question
            prompt = f"""Based on this educational content about {topic}, create a specific exam question that focuses on "{focus}" within the broader topic of "{main_topic}".

Content context:
{content_snippet}

Requirements:
- Create a clear, specific question suitable for university-level examination
- The question should focus on: {focus}
- Make it relevant to {topic}
- Ensure it can be answered based on the content context
- Use academic language appropriate for an exam

Generate only the question text, no additional formatting or explanations."""

            response = self.gemini_client.generation_model.generate_content(
                prompt,
                generation_config=self.gemini_client._get_generation_config(temperature=0.3)
            )

            generated_question = response.text.strip()

            # Clean up the generated question
            if generated_question and len(generated_question) > 20:
                return generated_question
            else:
                # Fallback to template-based question
                return self._create_template_question(focus, topic, main_topic)

        except Exception as e:
            logger.error(f"❌ Failed to create question text with Gemini: {e}")
            return self._create_template_question(focus, topic, main_topic)

    def _create_template_question(self, focus: str, topic: str, main_topic: str) -> str:
        """Create template-based question as fallback"""
        templates = {
            "basic definitions": f"Define and explain the key concepts of {focus} in {topic}.",
            "conceptual explanation": f"Explain how {focus} relates to {main_topic} in the context of {topic}.",
            "comparison and contrast": f"Compare and contrast different approaches to {focus} in {topic}.",
            "practical applications": f"Discuss the practical applications of {focus} in {topic}.",
            "real-world examples": f"Provide and analyze real-world examples of {focus} in {topic}.",
            "implementation challenges": f"Analyze the main challenges in implementing {focus} within {topic}.",
            "technical analysis": f"Conduct a technical analysis of {focus} in relation to {topic}.",
            "problem-solving approach": f"Describe a systematic approach to solving problems related to {focus} in {topic}.",
            "evaluation and critique": f"Critically evaluate the effectiveness of {focus} in {topic}.",
            "critical evaluation": f"Provide a critical evaluation of {focus} considering its impact on {topic}.",
            "limitations and challenges": f"Discuss the limitations and challenges associated with {focus} in {topic}.",
            "future developments": f"Analyze potential future developments in {focus} within the field of {topic}."
        }

        return templates.get(focus.lower(), f"Analyze and discuss {focus} in the context of {topic}.")

    def _generate_marking_criteria(self, focus: str, marks: int, content_text: str) -> List[str]:
        """Generate marking criteria based on focus and content"""
        try:
            base_criteria = {
                "basic definitions": [
                    "Clear and accurate definitions",
                    "Understanding of key concepts",
                    "Appropriate use of terminology"
                ],
                "conceptual explanation": [
                    "Clear explanation of concepts",
                    "Logical structure and flow",
                    "Evidence of understanding"
                ],
                "comparison and contrast": [
                    "Identification of similarities and differences",
                    "Balanced analysis",
                    "Clear conclusions"
                ],
                "practical applications": [
                    "Relevant practical examples",
                    "Understanding of application contexts",
                    "Analysis of effectiveness"
                ],
                "critical evaluation": [
                    "Critical thinking demonstrated",
                    "Balanced evaluation",
                    "Evidence-based conclusions",
                    "Consideration of multiple perspectives"
                ]
            }

            criteria = base_criteria.get(focus.lower(), [
                "Understanding of topic",
                "Clear explanation",
                "Relevant examples"
            ])

            # Add mark distribution
            if marks >= 8:
                criteria.append("Depth of analysis and detail")
            if marks >= 10:
                criteria.append("Original thinking and insights")

            return criteria[:4]  # Limit to 4 criteria maximum

        except Exception as e:
            logger.error(f"❌ Failed to generate marking criteria: {e}")
            return ["Understanding of topic", "Clear explanation"]

    def _generate_markdown_content(self, exam_data: Dict, topic: str) -> str:
        """Generate formatted exam paper content in Markdown format"""
        try:
            exam_metadata = exam_data.get('exam_metadata', {})
            questions = exam_data.get('questions', {})
            
            markdown_content = f"""# {exam_metadata.get('title', f'Examination Paper - {topic}')}

**Subject:** {exam_metadata.get('subject', topic)}  
**Total Questions:** {exam_metadata.get('total_questions', 0)}  
**Total Marks:** {exam_metadata.get('total_marks', 0)}  
**Time Allowed:** {exam_metadata.get('time_allowed', '3 hours')}  
**Date:** {datetime.now().strftime('%d %B %Y')}

---

## Instructions to Candidates

"""
            
            # Add instructions
            instructions = exam_data.get('instructions', [])
            for i, instruction in enumerate(instructions, 1):
                markdown_content += f"{i}. {instruction}\n"
            
            markdown_content += "\n---\n\n**DO NOT TURN THIS PAGE OVER UNTIL YOU ARE TOLD TO DO SO**\n\n---\n\n"
            
            # Add questions
            for q_num, question_data in questions.items():
                markdown_content += f"## {q_num}. {question_data['main_topic'].upper()}\n\n"
                markdown_content += f"**Total marks:** {question_data['total_marks']}\n\n"
                
                for sub_q in question_data["sub_questions"]:
                    markdown_content += f"### {sub_q['part']} {sub_q['question']}\n\n"
                    markdown_content += f"*({sub_q['marks']} marks)*\n\n"
                    
                    # Add marking criteria as a collapsible section
                    if sub_q.get('marking_criteria'):
                        markdown_content += f"<details>\n<summary>Marking Criteria</summary>\n\n"
                        for criteria in sub_q['marking_criteria']:
                            markdown_content += f"- {criteria}\n"
                        markdown_content += f"\n</details>\n\n"
                
                markdown_content += "---\n\n"
            
            markdown_content += "\n**END OF EXAMINATION PAPER**\n"
            
            return markdown_content
            
        except Exception as e:
            logger.error(f"❌ Failed to generate markdown content: {e}")
            return f"# Error\n\nFailed to generate exam content: {str(e)}"

    def _convert_markdown_to_pdf(self, markdown_content: str, pdf_path: Path, 
                                 title: str = "Examination Paper") -> bool:
        """Convert markdown content to PDF format"""
        try:
            # Create PDF with table of contents
            pdf = MarkdownPdf(toc_level=2)
            
            # Add the exam content as a section
            pdf.add_section(Section(markdown_content, toc=True))
            
            # Set PDF metadata
            pdf.meta["title"] = title
            pdf.meta["author"] = "Exam Generation System"
            pdf.meta["subject"] = "Academic Examination"
            
            # Save to PDF
            pdf.save(str(pdf_path))
            
            logger.info(f"✅ Successfully converted to PDF: {pdf_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to convert markdown to PDF: {e}")
            return False

    def save_multi_format_exam(self, exam_data: Dict, base_path: Path, 
                              formats: List[str] = ['txt', 'md', 'pdf']) -> Dict[str, Path]:
        """Save exam paper in multiple formats (txt, md, pdf)"""
        saved_files = {}
        
        try:
            exam_metadata = exam_data.get('exam_metadata', {})
            topic = exam_metadata.get('subject', 'Unknown Topic')
            
            # Generate base filename without extension
            base_filename = base_path.stem
            output_dir = base_path.parent
            
            # 1. Save as TXT format (existing functionality)
            if 'txt' in formats:
                txt_path = output_dir / f"{base_filename}.txt"
                try:
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(exam_data.get('paper_content', 'No content available'))
                    saved_files['txt'] = txt_path
                    logger.info(f"✅ Saved TXT format: {txt_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to save TXT format: {e}")
            
            # 2. Save as Markdown format
            if 'md' in formats:
                md_path = output_dir / f"{base_filename}.md" 
                try:
                    markdown_content = self._generate_markdown_content(exam_data, topic)
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    saved_files['md'] = md_path
                    logger.info(f"✅ Saved Markdown format: {md_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to save Markdown format: {e}")
            
            # 3. Save as PDF format
            if 'pdf' in formats:
                pdf_path = output_dir / f"{base_filename}.pdf"
                try:
                    # Generate markdown content first
                    markdown_content = self._generate_markdown_content(exam_data, topic)
                    
                    # Convert markdown to PDF
                    title = exam_metadata.get('title', f'Examination Paper - {topic}')
                    if self._convert_markdown_to_pdf(markdown_content, pdf_path, title):
                        saved_files['pdf'] = pdf_path
                        logger.info(f"✅ Saved PDF format: {pdf_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to save PDF format: {e}")
            
            return saved_files
            
        except Exception as e:
            logger.error(f"❌ Failed to save multi-format exam: {e}")
            return {}

    def _format_complete_exam_paper(self, exam_structure: Dict, topic: str,
                                  content_count: int) -> Dict:
        """Format the complete exam paper with proper structure"""
        try:
            formatted_exam = {
                "exam_metadata": {
                    "title": f"Structured Examination Paper - {topic}",
                    "subject": topic,
                    "total_questions": exam_structure["total_questions"],
                    "total_marks": exam_structure["total_marks"],
                    "time_allowed": "3 hours",
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "content_sources": content_count,
                    "generation_method": "Embedding-based similarity matching"
                },
                "instructions": [
                    f"Answer all {exam_structure['total_questions']} questions.",
                    "All questions carry marks as indicated.",
                    "Write all answers in the answer book provided.",
                    "Show your working where appropriate."
                ],
                "questions": exam_structure["questions"],
                "paper_content": self._generate_formatted_paper_text(exam_structure, topic)
            }

            return formatted_exam

        except Exception as e:
            logger.error(f"❌ Failed to format complete exam paper: {e}")
            return exam_structure

    def _generate_formatted_paper_text(self, exam_structure: Dict, topic: str) -> str:
        """Generate formatted exam paper text"""
        try:
            paper_text = f"""
STRUCTURED EXAMINATION PAPER

{topic.upper()}

INSTRUCTIONS TO CANDIDATES:

1. Answer ALL questions in ALL sections.
2. All questions carry marks as indicated.
3. This question paper has {exam_structure['total_questions']} questions totaling 100 marks.
4. Write all answers in the answer book provided.
5. Show your working where appropriate.

DO NOT TURN THIS PAGE OVER UNTIL YOU ARE TOLD TO DO SO

---

Time Allowed: 3 hours
Date: {datetime.now().strftime('%d %B %Y')}

"""

            for q_num, question_data in exam_structure["questions"].items():
                paper_text += f"\n{q_num}. {question_data['main_topic'].upper()}\n"
                paper_text += f"Total marks: {question_data['total_marks']}\n\n"

                for sub_q in question_data["sub_questions"]:
                    paper_text += f"{sub_q['part']} {sub_q['question']}\n"
                    paper_text += f"({sub_q['marks']} marks)\n\n"

                paper_text += "\n---\n"

            paper_text += "\nEND OF EXAMINATION PAPER"

            return paper_text

        except Exception as e:
            logger.error(f"❌ Failed to generate formatted paper text: {e}")
            return "Error generating paper format"

    # Keep the original method for backward compatibility
    def save_formatted_exam(self, exam_data: Dict, output_path: Path):
        """Save formatted exam paper as text file (backward compatibility)"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(exam_data.get('paper_content', 'No content available'))
            logger.info(f"✅ Saved formatted exam to: {output_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save formatted exam: {e}")

    def _create_empty_exam_structure(self, topic: str) -> Dict:
        """Create empty exam structure when content is not available"""
        return {
            "exam_metadata": {
                "title": f"Structured Examination Paper - {topic}",
                "subject": topic,
                "total_questions": 0,
                "total_marks": 0,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": "No content available for exam generation"
            },
            "instructions": [],
            "questions": {},
            "paper_content": "Error: No content available for exam generation"
        }

    def _create_fallback_question(self, q_num: str, q_structure: Dict, topic: str) -> Dict:
        """Create fallback question when generation fails"""
        return {
            "question_number": q_num,
            "main_topic": q_structure["main_topic"],
            "total_marks": q_structure["total_marks"],
            "sub_questions": [
                {
                    "part": sub_part["part"],
                    "question": f"Discuss {sub_part['focus']} in relation to {topic}.",
                    "marks": sub_part["marks"],
                    "focus": sub_part["focus"],
                    "marking_criteria": [f"Understanding of {sub_part['focus']}"],
                    "source_content": "Content not available",
                    "similarity_score": 0
                }
                for sub_part in q_structure["sub_parts"]
            ],
            "content_sources": ["Fallback generation"]
        }
