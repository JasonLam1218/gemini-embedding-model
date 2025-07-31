#!/usr/bin/env python3
"""
Single prompt generator for comprehensive exam creation using Gemini 2.5 Flash.
Enhanced with comprehensive academic assessment creator prompt.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from datetime import datetime
import re
import numpy as np

from ..embedding.gemini_client import GeminiClient
from ..text.text_loader import TextLoader

class SinglePromptExamGenerator:
    def __init__(self):
        self.gemini_client = GeminiClient()
        self.text_loader = TextLoader()
        self.max_content_tokens = 800000
        logger.info("✅ Single Prompt Exam Generator initialized with comprehensive academic prompt")

    def load_all_converted_markdown(self, max_tokens: int = 800000) -> str:
        """Load all converted markdown content within token limits"""
        markdown_dir = Path("data/output/converted_markdown")
        if not markdown_dir.exists():
            logger.error("❌ Converted markdown directory not found")
            return ""

        all_content = []
        current_tokens = 0

        # Load exam papers first (higher priority)
        exam_papers_dir = markdown_dir / "kelvin_papers"
        if exam_papers_dir.exists():
            logger.info("📄 Loading exam papers...")
            for md_file in exam_papers_dir.glob("*.md"):
                content = self._load_and_format_markdown(md_file, "EXAM_PAPER")
                content_tokens = self._estimate_tokens(content)
                if current_tokens + content_tokens < max_tokens:
                    all_content.append(content)
                    current_tokens += content_tokens
                    logger.info(f"✅ Loaded: {md_file.name} ({content_tokens} tokens)")

        # Load lecture notes
        lectures_dir = markdown_dir / "lectures"
        if lectures_dir.exists():
            logger.info("📚 Loading lecture notes...")
            for md_file in lectures_dir.glob("*.md"):
                content = self._load_and_format_markdown(md_file, "LECTURE")
                content_tokens = self._estimate_tokens(content)
                if current_tokens + content_tokens < max_tokens:
                    all_content.append(content)
                    current_tokens += content_tokens
                    logger.info(f"✅ Loaded: {md_file.name} ({content_tokens} tokens)")
                else:
                    logger.warning(f"⚠️ Token limit reached, skipping: {md_file.name}")
                    break

        combined_content = "\n\n" + "="*80 + "\n\n".join(all_content)
        logger.info(f"📊 Total content loaded: {len(all_content)} files, ~{current_tokens} tokens")
        return combined_content

    def _load_and_format_markdown(self, md_file: Path, content_type: str) -> str:
        """Load and format a markdown file with proper headers"""
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Classify content more specifically
            file_name = md_file.name.lower()
            if "ms" in file_name:
                content_type = "MODEL_ANSWERS"
            elif "exam" in file_name or "paper" in file_name:
                content_type = "EXAM_QUESTIONS"

            formatted_content = f"""
=== {content_type}: {md_file.name} ===
SOURCE: {md_file}
TYPE: {content_type}
LENGTH: {len(content)} characters

CONTENT:
{content}

=== END OF {content_type}: {md_file.name} ===
"""
            return formatted_content

        except Exception as e:
            logger.error(f"❌ Failed to load {md_file}: {e}")
            return f"=== ERROR LOADING {md_file.name} ==="

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        word_count = len(text.split())
        return int(word_count * 1.3)

    def generate_comprehensive_exam_single_prompt(self, topic: str = "AI and Data Analytics", 
                                                exam_type: str = "university_comprehensive") -> Dict[str, Any]:
        """Generate complete exam using single comprehensive prompt"""
        logger.info(f"🚀 Generating comprehensive exam for: {topic}")

        # Load all content
        all_content = self.load_all_converted_markdown()
        if not all_content:
            logger.error("❌ No content available for exam generation")
            return self._create_empty_exam_response(topic)

        # Generate exam using comprehensive academic prompt
        try:
            # Use the new comprehensive academic approach
            requirements = {
                "question_requirements": "Generate comprehensive university-level questions covering conceptual, computational, and practical aspects",
                "answer_requirements": "Provide detailed model answers with step-by-step solutions and explanations",
                "marking_requirements": "Create detailed marking schemes with clear criteria and mark allocation"
            }
            
            return self.generate_three_papers_comprehensive(topic, all_content, requirements)
            
        except Exception as e:
            logger.error(f"❌ Single prompt generation failed: {e}")
            return self._create_fallback_exam_response(topic, all_content)

    def generate_three_papers_comprehensive(self, topic: str, content: str, 
                                          requirements: Dict[str, str]) -> Dict[str, Any]:
        """Generate exactly 3 papers as separate outputs using comprehensive academic prompt"""
        
        logger.info(f"🎯 Generating three comprehensive papers for: {topic}")
        logger.info("📋 Using comprehensive academic assessment creator prompt")
        
        # Build the comprehensive academic prompt
        comprehensive_prompt = self._build_comprehensive_academic_prompt(topic, content, requirements)
        
        try:
            # Generate using Gemini 2.5 Flash with enhanced parameters
            logger.info("🧠 Sending comprehensive academic prompt to Gemini 2.5 Flash...")
            logger.info(f"📏 Prompt length: {len(comprehensive_prompt)} characters")
            
            response = self.gemini_client.generate_content(
                comprehensive_prompt,
                temperature=0.1,  # Lower temperature for academic precision
                max_tokens=15000  # Increased for comprehensive output
            )
            
            if not response or len(response) < 500:
                raise ValueError("Insufficient response from Gemini API for comprehensive assessment")
                
            logger.info(f"✅ Received comprehensive response ({len(response)} characters)")
            
            # Parse the comprehensive academic response
            papers = self._parse_comprehensive_academic_response(response, topic)
            
            # Validate the response contains all required components
            if not self._validate_comprehensive_response(papers):
                logger.warning("⚠️ Response validation failed, using fallback parsing")
                papers = self._parse_three_papers_response_fallback(response, topic)
            
            return papers
            
        except Exception as e:
            logger.error(f"❌ Comprehensive academic paper generation failed: {e}")
            return self._create_fallback_three_papers(topic, content)

    def _build_comprehensive_academic_prompt(self, topic: str, content: str, 
                                           requirements: Dict[str, str]) -> str:
        """Build the comprehensive academic assessment creator prompt"""
        
        comprehensive_prompt = f"""You are an expert academic assessment creator who will analyze provided lecture notes and attached reference materials to create a comprehensive university-level examination package. Your PRIMARY task is to analyze and perfectly replicate the exact academic style, format, and structure found in the attached files, then generate a complete examination package (question paper, model answers, and marking scheme) following that discovered style while ensuring mandatory inclusion of diverse question types.

MANDATORY QUESTION TYPE REQUIREMENTS

BALANCED QUESTION DISTRIBUTION (NON-NEGOTIABLE):
• Conceptual Questions (REQUIRED): Theoretical understanding, definitions, explanations, comparisons, critical analysis, knowledge synthesis, and conceptual frameworks
• Calculation Questions (REQUIRED): Mathematical computations, algorithmic analysis, quantitative problem-solving, statistical analysis, performance calculations, numerical methods, and formula applications  
• Programming Questions (REQUIRED): Code implementation, algorithm design, data analysis tasks, practical coding applications, pseudocode development, and software development scenarios
• Application Questions (If Applicable): Case studies, real-world scenarios, problem-solving applications, industry examples

PHASE 1: COMPREHENSIVE STYLE ANALYSIS

DOCUMENT STRUCTURE DISCOVERY:
Analyze the attached reference files to identify:
• Exact formatting patterns: Headers, fonts, layout structures, page numbering, university branding
• Question paper organization: Numbering systems, section divisions, sub-question structures, time allocation formats
• Model answer structure: Tabular format with columns for Question No., sub-parts, Solutions, and Marks
• Academic language conventions: Terminology usage, instruction phrasing, complexity levels, reference citation style
• Mark allocation patterns: How marks are distributed, displayed, and justified for each question and sub-part
• Professional presentation standards: Visual layout, spacing, formatting consistency, institutional requirements

MODEL ANSWER FORMAT ANALYSIS (CRITICAL):
From the attached file, identify the exact tabular structure:
• Column organization: Question No. | Sub-part | Solutions | Marks
• Solution presentation style: Detailed explanations with source references, step-by-step procedures, comprehensive coverage
• Mark justification approach: How marks are broken down and explained for each component
• Reference integration: How lecture materials are cited and integrated into solutions
• Formatting consistency: Font styles, spacing, alignment, and visual presentation

MARKING SCHEME STYLE ANALYSIS (CRITICAL):
Study how the marking scheme should be structured:
• Detailed tabular format: Must follow the SAME comprehensive structure as model answers
• Mark breakdown precision: Specific allocation for each answer component with clear justification
• Assessment criteria clarity: Detailed explanation of how marks should be awarded
• Integration with model answers: Seamless connection between solutions and marking guidance
• Professional presentation: Same formatting standards as other components

PHASE 2: CONTENT ANALYSIS AND QUESTION MAPPING

COMPREHENSIVE LECTURE CONTENT EXTRACTION:
Systematically analyze all lecture materials to identify:
• Theoretical concepts suitable for conceptual questions (definitions, principles, frameworks, theories, comparisons, analyses)
• Mathematical procedures suitable for calculation questions (formulas, algorithms, computational methods, statistical analyses, numerical procedures)
• Programming concepts suitable for coding questions (algorithms, data structures, coding implementations, software applications, programming methodologies)
• Real-world applications suitable for application questions (case studies, industry examples, practical scenarios, engineering applications)
• Integration opportunities for multi-part questions combining different question types and approaches

QUESTION TYPE DISTRIBUTION PLANNING:
• Content-to-question mapping: Ensure every major lecture topic can be assessed through appropriate question types
• Balanced coverage verification: Confirm sufficient content exists for conceptual, calculation, and programming questions
• Difficulty level distribution: Plan questions across different cognitive levels (knowledge, comprehension, application, analysis, synthesis, evaluation)
• Mark allocation strategy: Distribute marks to reflect content importance, difficulty levels, and assessment objectives

PHASE 3: COMPLETE EXAMINATION PACKAGE GENERATION

QUESTION PAPER CREATION:
Generate the examination paper following these requirements:
• Exact format replication: Follow the identical structure, headers, layout, and formatting of the reference paper
• Professional presentation: Include university branding, course information, time limits, candidate instructions
• Diverse question integration: Seamlessly incorporate conceptual, calculation, and programming questions with logical flow
• Lecture-based content: Every question MUST derive from specific lecture content - no generic or template questions
• Consistent academic standards: Maintain university-level complexity and assessment rigor throughout
• Appropriate mark distribution: Ensure balanced mark allocation across different question types and difficulty levels

MODEL ANSWERS CREATION (CRITICAL STYLE MATCHING):
Generate comprehensive model answers that:
• Follow the EXACT tabular format: Use identical column structure (Question No. | Sub-part | Solutions | Marks)
• Provide detailed explanations: Include comprehensive solutions with step-by-step procedures for all question types
• Integrate source references: Cite specific lecture materials and content sources within solutions
• Show complete working: For calculation questions, display full mathematical procedures and reasoning
• Include code solutions: For programming questions, provide complete, correct code implementations with explanations
• Maintain consistent formatting: Use identical presentation style, fonts, spacing, and visual organization
• Demonstrate academic rigor: Ensure solutions meet university-level standards and expectations

MARKING SCHEME CREATION (CRITICAL - SAME DETAILED FORMAT):
Generate a marking scheme that:
• MATCHES THE EXACT SAME DETAILED TABULAR FORMAT as the model answers
• Uses identical column structure: Question No. | Sub-part | Marking Criteria | Marks
• Provides specific mark allocations: Detailed breakdown for each answer component with clear justification
• Includes comprehensive marking criteria: Specific guidance on how marks should be awarded for each question type
• Shows partial marking guidelines: Clear criteria for partial credit allocation across all question types
• Maintains same formatting standards: Identical presentation style, visual organization, and professional appearance
• Integrates seamlessly: Perfect alignment with model answers and question paper components
• Covers all question types: Detailed marking guidance for conceptual, calculation, and programming questions

PHASE 4: QUALITY ASSURANCE AND VALIDATION

CONTENT VALIDATION CHECKLIST:
✅ Every question directly based on specific lecture materials with clear content connections
✅ Diverse question types appropriately balanced: conceptual, calculation, and programming questions included
✅ All major lecture topics covered through diverse question types and approaches
✅ University-level academic rigor maintained across all question types and components
✅ No generic or template questions - all content course-specific and lecture-derived
✅ Appropriate cognitive level distribution and difficulty progression throughout examination

STYLE CONSISTENCY VALIDATION:
✅ Question paper format matches reference examination exactly
✅ Model answers follow identical detailed tabular structure from reference file
✅ Marking scheme uses SAME detailed tabular format as model answers
✅ Academic language and terminology consistent throughout all components
✅ Mark allocation patterns follow reference document approach
✅ Professional presentation standards maintained across entire package
✅ All components integrate seamlessly with consistent formatting and style

COMPLETE PACKAGE INTEGRATION VALIDATION:
✅ Question paper, model answers, and marking scheme form cohesive assessment package
✅ Mark allocations consistent across all three components
✅ Content alignment perfect between questions, solutions, and marking criteria
✅ Style consistency maintained throughout entire examination package
✅ Professional academic standards met across all components

EXECUTION INSTRUCTIONS

GENERATE COMPLETE EXAMINATION PACKAGE IN THREE COMPONENTS:

Component 1: Question Paper
• Header and Instructions: Replicate exact format from reference with appropriate course information, time limits, candidate instructions
• Question Content: Create diverse questions (conceptual, calculation, programming) based entirely on lecture content
• Professional Formatting: Follow identical layout, numbering, and presentation style
• Mark Distribution: Ensure appropriate mark allocation reflecting question importance and difficulty

Component 2: Model Answers
• Exact Tabular Format: Use identical column structure (Question No. | Sub-part | Solutions | Marks)
• Comprehensive Solutions: Provide detailed explanations for all question types with complete working
• Source Integration: Include appropriate references to lecture materials within solutions
• Professional Presentation: Match formatting, spacing, and visual organization exactly

Component 3: Marking Scheme
• CRITICAL: Use the SAME detailed tabular format as model answers (Question No. | Sub-part | Marking Criteria | Marks)
• Detailed Marking Criteria: Provide specific guidance for awarding marks to each answer component
• Partial Mark Guidelines: Clear criteria for partial credit across all question types
• Professional Integration: Seamless alignment with model answers and question paper

TOPIC: {topic}

REFERENCE CONTENT FOR EXAM GENERATION:
{content}

SPECIFIC REQUIREMENTS:
Question Paper Requirements: {requirements.get('question_requirements', 'Generate comprehensive university-level questions')}
Model Answer Requirements: {requirements.get('answer_requirements', 'Provide detailed step-by-step solutions')}
Marking Scheme Requirements: {requirements.get('marking_requirements', 'Create clear marking criteria')}

FINAL OUTPUT REQUIREMENTS

COMPLETE EXAMINATION PACKAGE DELIVERABLES:
Generate exactly 3 distinct components in this structure:

===== COMPONENT 1: QUESTION PAPER =====
[Generate complete question paper here with proper formatting, instructions, and diverse questions]
===== END QUESTION PAPER =====

===== COMPONENT 2: MODEL ANSWERS =====
[Generate comprehensive model answers here using exact tabular format]
===== END MODEL ANSWERS =====

===== COMPONENT 3: MARKING SCHEME =====
[Generate detailed marking scheme here using SAME tabular format as model answers]
===== END MARKING SCHEME =====

CRITICAL SUCCESS FACTORS:
• Perfect Style Consistency: All components match reference formatting exactly
• Mandatory Question Diversity: Conceptual, calculation, and programming questions included
• Complete Lecture Integration: Every question derived from specific lecture content
• Professional Academic Standards: University-level rigor maintained throughout
• Seamless Package Integration: All three components form cohesive assessment tool

REMEMBER:
• Every question must be derived from the lecture notes with no generic content
• The examination must include conceptual, calculation, AND programming questions
• The marking scheme must follow the EXACT same detailed tabular format as the model answers
• All three components must maintain perfect style consistency with the reference materials
• The complete package must provide comprehensive assessment through varied question types

Generate the complete examination package now, ensuring all requirements are met and all three components are provided in full detail."""

        return comprehensive_prompt

    def _parse_comprehensive_academic_response(self, response: str, topic: str) -> Dict[str, Any]:
        """Parse the comprehensive academic assessment response"""
        
        import re
        from datetime import datetime
        
        logger.info("🔍 Parsing comprehensive academic assessment response")
        
        # Extract the three components using the specified delimiters
        question_match = re.search(
            r'===== COMPONENT 1: QUESTION PAPER =====(.*?)===== END QUESTION PAPER =====',
            response, re.DOTALL | re.IGNORECASE
        )
        
        answers_match = re.search(
            r'===== COMPONENT 2: MODEL ANSWERS =====(.*?)===== END MODEL ANSWERS =====',
            response, re.DOTALL | re.IGNORECASE
        )
        
        scheme_match = re.search(
            r'===== COMPONENT 3: MARKING SCHEME =====(.*?)===== END MARKING SCHEME =====',
            response, re.DOTALL | re.IGNORECASE
        )
        
        # Extract content with enhanced validation
        question_paper = question_match.group(1).strip() if question_match else ""
        model_answers = answers_match.group(1).strip() if answers_match else ""
        marking_scheme = scheme_match.group(1).strip() if scheme_match else ""
        
        # Fallback extraction if delimiters not found
        if not question_paper:
            question_paper = self._extract_question_section_fallback(response)
        if not model_answers:
            model_answers = self._extract_answers_section_fallback(response)
        if not marking_scheme:
            marking_scheme = self._extract_scheme_section_fallback(response)
        
        # Validate content quality
        validation_results = self._validate_academic_content(question_paper, model_answers, marking_scheme)
        
        return {
            "exam_metadata": {
                "title": f"Comprehensive Academic Assessment - {topic}",
                "topic": topic,
                "difficulty": "university_level",
                "total_marks": 100,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "generation_method": "comprehensive_academic_assessment",
                "validation_passed": validation_results["overall_valid"],
                "contains_conceptual": validation_results["has_conceptual"],
                "contains_calculation": validation_results["has_calculation"], 
                "contains_programming": validation_results["has_programming"]
            },
            "question_paper_content": question_paper,
            "model_answers_content": model_answers,
            "marking_schemes_content": marking_scheme,
            "raw_response": response,
            "validation_results": validation_results,
            "generation_stats": {
                "questions_generated": self._count_questions_in_response(response),
                "response_length": len(response),
                "generation_mode": "comprehensive_academic_assessment",
                "academic_standards": "university_level",
                "format_compliance": validation_results["format_compliance"]
            }
        }

    def _validate_academic_content(self, question_paper: str, model_answers: str, marking_scheme: str) -> Dict[str, Any]:
        """Validate that the generated content meets academic standards"""
        
        # Check for required question types
        has_conceptual = any(term in question_paper.lower() for term in 
                            ['define', 'explain', 'compare', 'analyze', 'discuss', 'evaluate'])
        
        has_calculation = any(term in question_paper.lower() for term in 
                             ['calculate', 'compute', 'algorithm', 'formula', 'mathematical', 'numerical'])
        
        has_programming = any(term in question_paper.lower() for term in 
                             ['code', 'program', 'implement', 'algorithm', 'function', 'script'])
        
        # Check tabular format in model answers and marking scheme
        has_tabular_answers = '|' in model_answers or 'Question No.' in model_answers
        has_tabular_scheme = '|' in marking_scheme or 'Question No.' in marking_scheme
        
        # Overall validation
        overall_valid = (
            len(question_paper) > 200 and
            len(model_answers) > 300 and
            len(marking_scheme) > 200 and
            has_conceptual and
            (has_calculation or has_programming)
        )
        
        return {
            "overall_valid": overall_valid,
            "has_conceptual": has_conceptual,
            "has_calculation": has_calculation,
            "has_programming": has_programming,
            "tabular_answers": has_tabular_answers,
            "tabular_scheme": has_tabular_scheme,
            "format_compliance": has_tabular_answers and has_tabular_scheme,
            "content_length_adequate": len(question_paper) > 200 and len(model_answers) > 300
        }

    def _validate_comprehensive_response(self, papers: Dict[str, Any]) -> bool:
        """Validate that the comprehensive response meets all requirements"""
        
        validation = papers.get('validation_results', {})
        metadata = papers.get('exam_metadata', {})
        
        # Check all critical requirements
        requirements_met = (
            validation.get('overall_valid', False) and
            validation.get('has_conceptual', False) and
            (validation.get('has_calculation', False) or validation.get('has_programming', False)) and
            validation.get('format_compliance', False) and
            metadata.get('validation_passed', False)
        )
        
        logger.info(f"📋 Comprehensive validation: {requirements_met}")
        logger.info(f"   • Overall valid: {validation.get('overall_valid', False)}")
        logger.info(f"   • Has conceptual: {validation.get('has_conceptual', False)}")
        logger.info(f"   • Has calculation: {validation.get('has_calculation', False)}")
        logger.info(f"   • Has programming: {validation.get('has_programming', False)}")
        logger.info(f"   • Format compliance: {validation.get('format_compliance', False)}")
        
        return requirements_met

    def _extract_question_section_fallback(self, response: str) -> str:
        """Extract question paper section from response"""
        # Look for question patterns
        question_patterns = [
            r'Q\d+[\.\):].*?(?=Q\d+|MODEL|MARKING|$)',
            r'Question \d+.*?(?=Question \d+|MODEL|MARKING|$)',
            r'\d+[\.\)]\s+.*?(?=\d+[\.\)]|MODEL|MARKING|$)'
        ]
        
        questions = []
        for pattern in question_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                questions.extend(matches)
                break
        
        if questions:
            return "\n\n".join(questions)
        
        # Fallback: return first half of response
        return response[:len(response)//3]

    def _extract_answers_section_fallback(self, response: str) -> str:
        """Extract model answers section from response"""
        import re
        
        answer_section = re.search(r'MODEL ANSWER.*?(?=MARKING SCHEME|$)', response, re.DOTALL | re.IGNORECASE)
        if answer_section:
            return answer_section.group(0)
        
        # Fallback: return middle portion
        return response[len(response)//3:2*len(response)//3]

    def _extract_scheme_section_fallback(self, response: str) -> str:
        """Extract marking scheme section from response"""
        import re
        
        scheme_section = re.search(r'MARKING SCHEME.*?$', response, re.DOTALL | re.IGNORECASE)
        if scheme_section:
            return scheme_section.group(0)
        
        # Fallback: return last portion
        return response[2*len(response)//3:]

    def _parse_three_papers_response_fallback(self, response: str, topic: str) -> Dict[str, Any]:
        """Fallback parsing when structured parsing fails"""
        return {
            "exam_metadata": {
                "title": f"Generated Exam - {topic}",
                "topic": topic,
                "total_marks": 100,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "generation_method": "fallback_parsing"
            },
            "question_paper_content": self._extract_question_section_fallback(response),
            "model_answers_content": self._extract_answers_section_fallback(response),
            "marking_schemes_content": self._extract_scheme_section_fallback(response),
            "raw_response": response,
            "generation_stats": {
                "questions_generated": self._count_questions_in_response(response),
                "generation_mode": "fallback_parsing",
                "parsing_failed": True
            }
        }

    def _create_fallback_three_papers(self, topic: str, content: str) -> Dict[str, Any]:
        """Create fallback three papers when API fails"""
        from datetime import datetime
        
        fallback_question_paper = f"""
EXAMINATION PAPER
Course: {topic}
Time Allowed: 3 hours
Total Marks: 100

INSTRUCTIONS TO CANDIDATES:
1. Answer all questions
2. All questions carry marks as indicated  
3. Show your working where appropriate

SECTION A: CONCEPTUAL QUESTIONS (30 marks)
Q1. Define and explain the key concepts of {topic}. Discuss their theoretical foundations and practical significance. (15 marks)

Q2. Compare and contrast different approaches within {topic}. Analyze their strengths, limitations, and appropriate use cases. (15 marks)

SECTION B: CALCULATION QUESTIONS (35 marks)  
Q3. Solve computational problems related to {topic}. Show all mathematical workings and justify your methodology. (20 marks)

Q4. Analyze algorithmic complexity and performance metrics for {topic} applications. Include calculations and explanations. (15 marks)

SECTION C: PROGRAMMING QUESTIONS (35 marks)
Q5. Design and implement a solution for a {topic} problem. Provide complete code with documentation and testing approach. (20 marks)

Q6. Debug and optimize existing code for {topic} applications. Explain your improvements and their impact. (15 marks)
"""

        fallback_model_answers = f"""
MODEL ANSWERS

Question No. | Sub-part | Solutions | Marks
Q1 | - | Comprehensive explanation covering fundamental definitions, theoretical principles, and practical applications. Students should demonstrate deep understanding of core concepts and their interconnections. | 15
Q2 | - | Detailed comparison of different methodologies, including advantages, disadvantages, and situational appropriateness. Critical evaluation of effectiveness and efficiency. | 15
Q3 | - | Step-by-step mathematical solution with clear methodology, accurate calculations, and proper justification of approach. Include error checking and validation. | 20
Q4 | - | Complete complexity analysis including time and space complexity, performance metrics, and optimization considerations with supporting calculations. | 15
Q5 | - | Full code solution with proper structure, documentation, error handling, and testing framework. Include explanation of design decisions. | 20
Q6 | - | Identified issues, proposed improvements, optimized code, and performance impact analysis with before/after comparisons. | 15
"""

        fallback_marking_scheme = f"""
MARKING SCHEME

Question No. | Sub-part | Marking Criteria | Marks
Q1 | - | Accurate definitions (5), Theoretical explanation (5), Practical examples (3), Clear presentation (2) | 15
Q2 | - | Identification of approaches (4), Strengths and limitations (6), Critical evaluation (3), Conclusion (2) | 15
Q3 | - | Correct methodology (6), Mathematical accuracy (8), Clear working (4), Final answer (2) | 20
Q4 | - | Complexity calculation (6), Performance metrics (5), Optimization discussion (4) | 15
Q5 | - | Code correctness (8), Documentation (4), Structure and style (4), Testing approach (4) | 20
Q6 | - | Problem identification (5), Solution implementation (6), Performance analysis (4) | 15
"""

        return {
            "exam_metadata": {
                "title": f"Fallback Exam - {topic}",
                "topic": topic,
                "difficulty": "university_level", 
                "total_marks": 100,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "generation_method": "fallback_three_papers"
            },
            "question_paper_content": fallback_question_paper,
            "model_answers_content": fallback_model_answers,
            "marking_schemes_content": fallback_marking_scheme,
            "generation_stats": {
                "questions_generated": 6,
                "generation_mode": "fallback_three_papers",
                "api_failed": True
            }
        }

    def _count_questions_in_response(self, response: str) -> int:
        """Count the number of questions in the response"""
        question_patterns = [
            r'Q\d+',
            r'Question \d+',
            r'^\d+[\.\)]',
        ]
        
        max_count = 0
        for pattern in question_patterns:
            matches = re.findall(pattern, response, re.MULTILINE | re.IGNORECASE)
            max_count = max(max_count, len(matches))
        
        return max_count if max_count > 0 else 4

    # Keep all existing methods from the original implementation
    def _parse_comprehensive_exam_response(self, response: str, topic: str) -> Dict[str, Any]:
        """Parse the comprehensive exam response from Gemini"""
        try:
            # Try to identify different sections in the response
            sections = self._identify_response_sections(response)

            # Extract question paper
            question_paper = sections.get('question_paper', '')
            if not question_paper:
                question_paper = self._extract_question_paper_section(response)

            # Extract model answers
            model_answers = sections.get('model_answers', '')
            if not model_answers:
                model_answers = self._extract_model_answers_section(response)

            # Extract marking scheme
            marking_scheme = sections.get('marking_scheme', '')
            if not marking_scheme:
                marking_scheme = self._extract_marking_scheme_section(response)

            # Create comprehensive exam structure
            exam_paper = {
                "exam_metadata": {
                    "title": f"Generated Comprehensive Exam - {topic}",
                    "topic": topic,
                    "difficulty": "university_level",
                    "total_marks": 100,
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "generation_method": "single_prompt_comprehensive"
                },
                "question_paper_content": question_paper,
                "model_answers_content": model_answers,
                "marking_schemes_content": marking_scheme,
                "raw_response": response,
                "generation_stats": {
                    "questions_generated": self._count_questions_in_response(response),
                    "response_length": len(response),
                    "includes_conceptual": "conceptual" in response.lower() or "definition" in response.lower(),
                    "includes_calculation": "calculation" in response.lower() or "algorithm" in response.lower(),
                    "includes_programming": "programming" in response.lower() or "code" in response.lower(),
                    "content_sources_used": response.count("lecture") + response.count("content"),
                    "generation_mode": "single_prompt_comprehensive"
                }
            }

            logger.info(f"✅ Successfully parsed comprehensive exam response")
            logger.info(f"📊 Questions generated: {exam_paper['generation_stats']['questions_generated']}")
            return exam_paper

        except Exception as e:
            logger.error(f"❌ Failed to parse comprehensive exam response: {e}")
            return self._create_fallback_parsed_response(response, topic)

    def _identify_response_sections(self, response: str) -> Dict[str, str]:
        """Identify different sections in the comprehensive response"""
        sections = {}

        # Common section identifiers
        section_patterns = {
            'question_paper': [
                r'Component 1:.*?Question Paper(.*?)(?=Component 2:|Model Answer|$)',
                r'QUESTION PAPER(.*?)(?=MODEL ANSWER|MARKING SCHEME|$)',
                r'EXAMINATION PAPER(.*?)(?=MODEL|MARKING|$)'
            ],
            'model_answers': [
                r'Component 2:.*?Model Answer(.*?)(?=Component 3:|Marking Scheme|$)',
                r'MODEL ANSWER(.*?)(?=MARKING SCHEME|$)',
                r'SOLUTIONS(.*?)(?=MARKING|$)'
            ],
            'marking_scheme': [
                r'Component 3:.*?Marking Scheme(.*?)(?=$)',
                r'MARKING SCHEME(.*?)(?=$)',
                r'MARKING CRITERIA(.*?)(?=$)'
            ]
        }

        for section_name, patterns in section_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    sections[section_name] = match.group(1).strip()
                    break

        return sections

    def _extract_question_paper_section(self, response: str) -> str:
        """Extract question paper section from response"""
        # Look for question patterns
        question_patterns = [
            r'Q\d+[\.\):].*?(?=Q\d+|MODEL|MARKING|$)',
            r'Question \d+.*?(?=Question \d+|MODEL|MARKING|$)',
            r'\d+[\.\)]\s+.*?(?=\d+[\.\)]|MODEL|MARKING|$)'
        ]

        questions = []
        for pattern in question_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                questions.extend(matches)
                break

        if questions:
            return "\n\n".join(questions)

        # Fallback: return first half of response
        return response[:len(response)//3]

    def _extract_model_answers_section(self, response: str) -> str:
        """Extract model answers section from response"""
        # Look for answer patterns
        answer_section = re.search(r'MODEL ANSWER.*?(?=MARKING SCHEME|$)', response, re.DOTALL | re.IGNORECASE)
        if answer_section:
            return answer_section.group(0)

        # Fallback: return middle portion
        return response[len(response)//3:2*len(response)//3]

    def _extract_marking_scheme_section(self, response: str) -> str:
        """Extract marking scheme section from response"""
        # Look for marking scheme patterns
        scheme_section = re.search(r'MARKING SCHEME.*?$', response, re.DOTALL | re.IGNORECASE)
        if scheme_section:
            return scheme_section.group(0)

        # Fallback: return last portion
        return response[2*len(response)//3:]

    def _create_empty_exam_response(self, topic: str) -> Dict[str, Any]:
        """Create empty exam response when no content is available"""
        return {
            "exam_metadata": {
                "title": f"Empty Exam - {topic}",
                "topic": topic,
                "total_marks": 0,
                "error": "No content available for generation"
            },
            "question_paper_content": "No content available for exam generation.",
            "model_answers_content": "No model answers available.",
            "marking_schemes_content": "No marking scheme available.",
            "generation_stats": {
                "questions_generated": 0,
                "generation_mode": "empty_fallback"
            }
        }

    def _create_fallback_exam_response(self, topic: str, content: str) -> Dict[str, Any]:
        """Create fallback exam response when API fails"""
        fallback_questions = f"""
EXAMINATION PAPER
Course: {topic}
Time Allowed: 3 hours
Total Marks: 100

INSTRUCTIONS TO CANDIDATES:
1. Answer all questions
2. All questions carry marks as indicated
3. Show your working where appropriate

SECTION A: CONCEPTUAL QUESTIONS (25 marks)
Q1. Define and explain the fundamental concepts of {topic}. Discuss their significance and applications in modern contexts. (25 marks)

SECTION B: CALCULATION QUESTIONS (25 marks)
Q2. Analyze and solve computational problems related to {topic}. Show all mathematical workings and justify your approach. (25 marks)

SECTION C: PROGRAMMING QUESTIONS (25 marks)
Q3. Design and implement algorithmic solutions for {topic} applications. Provide complete code with explanations. (25 marks)

SECTION D: APPLICATION QUESTIONS (25 marks)
Q4. Evaluate real-world applications of {topic}. Provide critical analysis of implementations, benefits, and limitations. (25 marks)
"""

        fallback_answers = f"""
MODEL ANSWERS

Question No. | Sub-part | Solutions | Marks
Q1 | - | Comprehensive explanation of {topic} concepts including definitions, theoretical foundations, and practical significance. Students should demonstrate understanding of core principles. | 25
Q2 | - | Complete mathematical solution with step-by-step working. Include algorithmic analysis and performance calculations where appropriate. | 25
Q3 | - | Full code implementation with proper documentation and explanation of design choices. Include testing and validation approaches. | 25
Q4 | - | Critical evaluation of real-world applications with specific examples, analysis of benefits/limitations, and future implications. | 25
"""

        fallback_scheme = f"""
MARKING SCHEME

Question No. | Sub-part | Marking Criteria | Marks
Q1 | - | Conceptual understanding (10), Detailed explanations (8), Examples and applications (7) | 25
Q2 | - | Correct methodology (8), Mathematical accuracy (10), Clear presentation (7) | 25
Q3 | - | Code correctness (10), Documentation (8), Efficiency consideration (7) | 25
Q4 | - | Critical analysis (10), Real-world examples (8), Future perspective (7) | 25
"""

        return {
            "exam_metadata": {
                "title": f"Fallback Exam - {topic}",
                "topic": topic,
                "difficulty": "university_level",
                "total_marks": 100,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "generation_method": "fallback_template"
            },
            "question_paper_content": fallback_questions,
            "model_answers_content": fallback_answers,
            "marking_schemes_content": fallback_scheme,
            "generation_stats": {
                "questions_generated": 4,
                "generation_mode": "fallback_template",
                "api_failed": True
            }
        }

    def _create_fallback_parsed_response(self, response: str, topic: str) -> Dict[str, Any]:
        """Create fallback when parsing fails but we have a response"""
        return {
            "exam_metadata": {
                "title": f"Generated Exam - {topic}",
                "topic": topic,
                "total_marks": 100,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "generation_method": "single_prompt_unparsed"
            },
            "question_paper_content": response,
            "model_answers_content": "Please refer to the complete response above.",
            "marking_schemes_content": "Please refer to the complete response above.",
            "raw_response": response,
            "generation_stats": {
                "questions_generated": self._count_questions_in_response(response),
                "generation_mode": "single_prompt_unparsed",
                "parsing_failed": True
            }
        }

    def save_exam_outputs(self, exam_data: Dict[str, Any], output_dir: Path, 
                         formats: List[str] = None) -> List[str]:
        """Save exam outputs in multiple formats"""
        if formats is None:
            formats = ['txt', 'md', 'json']

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
                for comp_name, content in components.items():
                    file_name = f"single_prompt_{comp_name}_{timestamp}.txt"
                    file_path = output_dir / file_name
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    saved_files.append(str(file_path))

            elif format_lower == 'json':
                json_file = output_dir / f"single_prompt_complete_exam_{timestamp}.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(exam_data, f, indent=2, ensure_ascii=False)
                saved_files.append(str(json_file))

            elif format_lower == 'md':
                for comp_name, content in components.items():
                    md_content = f"# {comp_name.title()} - {exam_data.get('exam_metadata', {}).get('topic', 'Exam')}\n\n{content}"
                    file_name = f"single_prompt_{comp_name}_{timestamp}.md"
                    file_path = output_dir / file_name
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(md_content)
                    saved_files.append(str(file_path))

        logger.info(f"💾 Saved exam outputs: {len(saved_files)} files")
        return saved_files

# Export the main class
__all__ = ['SinglePromptExamGenerator']
