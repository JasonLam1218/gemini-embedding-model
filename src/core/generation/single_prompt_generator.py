#!/usr/bin/env python3
"""
Enhanced Single prompt generator for comprehensive exam creation using Gemini 2.5 Flash.
Includes quality enhancements: advanced prompts, validation, and detailed marking schemes.
Complete implementation with full comprehensive academic prompt - NO TRUNCATION.
"""

import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from datetime import datetime
import numpy as np

from ..embedding.gemini_client import GeminiClient
from ..text.text_loader import TextLoader

class EnhancedPromptBuilder:
    def __init__(self):
        self.quality_rules = {
            'question_construction': [
                "Avoid 'all of the above' or 'none of the above' options",
                "Ensure exactly one clearly correct answer per question", 
                "Use parallel structure in multiple choice options",
                "Keep question stems concise and focused",
                "Avoid negative phrasing unless testing critical concepts"
            ],
            'content_requirements': [
                "Every question must derive from specific lecture content",
                "Include mandatory question types: conceptual, calculation, programming",
                "Balance cognitive levels: 30% knowledge, 40% application, 30% analysis",
                "Distribute marks: 25% easy, 50% medium, 25% challenging"
            ],
            'format_standards': [
                "Use consistent numbering and formatting",
                "Include clear mark allocation for each question",
                "Provide complete working for all calculation questions",
                "Include code solutions with explanations"
            ]
        }
    
    def build_enhanced_prompt(self, topic: str, content: str, requirements: Dict) -> str:
        """Build quality-enhanced comprehensive prompt"""
        
        rules_section = self._build_quality_rules_section()
        templates_section = self._build_question_templates_section()
        validation_section = self._build_validation_requirements_section()
        
        enhanced_prompt = f"""You are an expert academic assessment creator with strict quality standards.

{rules_section}

{templates_section}

MANDATORY QUALITY REQUIREMENTS:
- Generate EXACTLY 6 questions: 2 conceptual, 2 calculation, 2 programming
- Each question must include: clear statement, appropriate difficulty, complete solution
- Total marks must equal 100 (distributed as: 15, 15, 20, 15, 15, 20)
- All content must derive from provided lecture materials

{validation_section}

TOPIC: {topic}
LECTURE CONTENT: {content}

GENERATE THREE COMPONENTS:
===== COMPONENT 1: QUESTION PAPER =====
[Generate here with strict adherence to quality rules]
===== END QUESTION PAPER =====

===== COMPONENT 2: MODEL ANSWERS =====
[Generate detailed tabular answers with complete solutions]
===== END MODEL ANSWERS =====

===== COMPONENT 3: MARKING SCHEME =====
[Generate detailed marking criteria with partial credit guidelines]
===== END MARKING SCHEME ====="""

        return enhanced_prompt
    
    def _build_quality_rules_section(self) -> str:
        """Build comprehensive quality rules section"""
        rules_text = "QUALITY STANDARDS (NON-NEGOTIABLE):\n\n"
        
        for category, rules in self.quality_rules.items():
            rules_text += f"**{category.replace('_', ' ').title()}:**\n"
            for rule in rules:
                rules_text += f"• {rule}\n"
            rules_text += "\n"
            
        return rules_text
    
    def _build_question_templates_section(self) -> str:
        """Build question type templates"""
        return """
QUESTION TYPE TEMPLATES:

**Conceptual Questions:**
Template: "Explain [concept] and discuss its significance in [context]. Include [specific aspects] in your answer. (15 marks)"

**Calculation Questions:**  
Template: "Given [data/parameters], calculate [specific requirement]. Show all working and justify your methodology. (20 marks)"

**Programming Questions:**
Template: "Implement [specific functionality] that [requirements]. Provide complete code with documentation and test cases. (20 marks)"
"""
    
    def _build_validation_requirements_section(self) -> str:
        """Build validation requirements"""
        return """
VALIDATION CHECKLIST:
✅ All questions derived from lecture content with clear connections
✅ Balanced question types: conceptual (2), calculation (2), programming (2)
✅ Appropriate cognitive distribution across Bloom's taxonomy
✅ Complete solutions with step-by-step working
✅ Detailed marking criteria with partial credit guidelines
✅ Professional formatting consistent throughout
"""

class QualityValidator:
    def __init__(self):
        self.validation_criteria = {
            'content_completeness': {
                'min_questions': 4,
                'required_sections': ['question paper', 'model answers', 'marking scheme'],
                'min_total_length': 2000
            },
            'question_diversity': {
                'required_types': ['conceptual', 'calculation', 'programming'],
                'min_per_type': 1
            },
            'academic_standards': {
                'mark_allocation': {'min_total': 80, 'max_total': 120},
                'difficulty_distribution': {'easy': 0.2, 'medium': 0.6, 'hard': 0.2}
            }
        }
    
    def comprehensive_validation(self, exam_data: Dict) -> Dict[str, Any]:
        """Perform comprehensive exam quality validation"""
        
        validation_results = {
            'overall_valid': False,
            'validation_score': 0.0,
            'detailed_results': {},
            'improvement_suggestions': []
        }
        
        # Content completeness validation
        completeness = self._validate_content_completeness(exam_data)
        validation_results['detailed_results']['completeness'] = completeness
        
        # Question diversity validation  
        diversity = self._validate_question_diversity(exam_data)
        validation_results['detailed_results']['diversity'] = diversity
        
        # Academic standards validation
        standards = self._validate_academic_standards(exam_data)
        validation_results['detailed_results']['standards'] = standards
        
        # Format consistency validation
        formatting = self._validate_formatting_consistency(exam_data)
        validation_results['detailed_results']['formatting'] = formatting
        
        # Calculate overall score
        validation_results['validation_score'] = self._calculate_validation_score(
            completeness, diversity, standards, formatting
        )
        
        validation_results['overall_valid'] = validation_results['validation_score'] >= 0.8
        
        # Generate improvement suggestions
        if not validation_results['overall_valid']:
            validation_results['improvement_suggestions'] = self._generate_improvement_suggestions(
                validation_results['detailed_results']
            )
        
        return validation_results
    
    def _validate_question_diversity(self, exam_data: Dict) -> Dict[str, Any]:
        """Validate question type diversity"""
        question_content = exam_data.get('question_paper_content', '').lower()
        
        type_indicators = {
            'conceptual': ['explain', 'define', 'compare', 'analyze', 'discuss'],
            'calculation': ['calculate', 'compute', 'algorithm', 'formula', 'solve'],
            'programming': ['code', 'implement', 'program', 'function', 'algorithm']
        }
        
        detected_types = {}
        for q_type, indicators in type_indicators.items():
            detected_types[q_type] = any(indicator in question_content for indicator in indicators)
        
        diversity_score = sum(detected_types.values()) / len(type_indicators)
        
        return {
            'diversity_score': diversity_score,
            'detected_types': detected_types,
            'meets_requirement': diversity_score >= 0.67,
            'missing_types': [t for t, present in detected_types.items() if not present]
        }
    
    def _validate_academic_standards(self, exam_data: Dict) -> Dict[str, Any]:
        """Validate academic rigor and standards"""
        question_paper = exam_data.get('question_paper_content', '')
        model_answers = exam_data.get('model_answers_content', '')
        
        # Check mark allocation
        mark_matches = re.findall(r'(\d+)\s*marks?', question_paper.lower())
        total_marks = sum(int(match) for match in mark_matches)
        
        # Check solution completeness
        has_detailed_solutions = len(model_answers) > 1000 and ('step' in model_answers.lower() or 'solution' in model_answers.lower())
        
        return {
            'total_marks': total_marks,
            'marks_appropriate': 80 <= total_marks <= 120,
            'has_detailed_solutions': has_detailed_solutions,
            'academic_language': self._check_academic_language(question_paper),
            'standards_score': (
                (0.4 if 80 <= total_marks <= 120 else 0) +
                (0.4 if has_detailed_solutions else 0) +
                (0.2 if self._check_academic_language(question_paper) else 0)
            )
        }
    
    def _validate_content_completeness(self, exam_data: Dict) -> Dict[str, Any]:
        """Validate content completeness"""
        question_paper = exam_data.get('question_paper_content', '')
        model_answers = exam_data.get('model_answers_content', '')
        marking_scheme = exam_data.get('marking_schemes_content', '')
        
        return {
            'has_questions': len(question_paper) > 500,
            'has_answers': len(model_answers) > 500,
            'has_marking': len(marking_scheme) > 300,
            'completeness_score': (
                (0.4 if len(question_paper) > 500 else 0) +
                (0.4 if len(model_answers) > 500 else 0) +
                (0.2 if len(marking_scheme) > 300 else 0)
            )
        }
    
    def _validate_formatting_consistency(self, exam_data: Dict) -> Dict[str, Any]:
        """Validate formatting consistency"""
        model_answers = exam_data.get('model_answers_content', '')
        marking_scheme = exam_data.get('marking_schemes_content', '')
        
        has_tabular_format = '|' in model_answers or 'Question No.' in model_answers
        consistent_marking = '|' in marking_scheme or 'Question No.' in marking_scheme
        
        return {
            'has_tabular_format': has_tabular_format,
            'consistent_marking': consistent_marking,
            'formatting_score': (0.5 if has_tabular_format else 0) + (0.5 if consistent_marking else 0)
        }
    
    def _calculate_validation_score(self, completeness, diversity, standards, formatting) -> float:
        """Calculate overall validation score"""
        weights = {
            'completeness': 0.3,
            'diversity': 0.3,
            'standards': 0.3,
            'formatting': 0.1
        }
        
        scores = {
            'completeness': completeness['completeness_score'],
            'diversity': diversity['diversity_score'],
            'standards': standards['standards_score'],
            'formatting': formatting['formatting_score']
        }
        
        return sum(weights[key] * scores[key] for key in weights.keys())
    
    def _generate_improvement_suggestions(self, detailed_results: Dict) -> List[str]:
        """Generate improvement suggestions based on validation results"""
        suggestions = []
        
        if detailed_results['completeness']['completeness_score'] < 0.8:
            suggestions.append("Ensure all three components (questions, answers, marking) are substantial")
        
        if detailed_results['diversity']['diversity_score'] < 0.67:
            missing = detailed_results['diversity']['missing_types']
            suggestions.append(f"Include missing question types: {', '.join(missing)}")
        
        if detailed_results['standards']['standards_score'] < 0.8:
            suggestions.append("Improve academic standards with better mark allocation and detailed solutions")
        
        return suggestions
    
    def _check_academic_language(self, text: str) -> bool:
        """Check for appropriate academic language"""
        academic_terms = [
            'analyze', 'evaluate', 'synthesize', 'justify', 'demonstrate',
            'methodology', 'implementation', 'framework', 'principle'
        ]
        return sum(1 for term in academic_terms if term in text.lower()) >= 3

class SinglePromptExamGenerator:
    def __init__(self):
        self.gemini_client = GeminiClient()
        self.text_loader = TextLoader()
        self.prompt_builder = EnhancedPromptBuilder()
        self.quality_validator = QualityValidator()
        self.max_content_tokens = None  # Remove token limit
        logger.info("✅ Enhanced Single Prompt Exam Generator initialized")

    def load_all_converted_markdown(self, max_tokens: int = None) -> str:
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
                if max_tokens is None or current_tokens + content_tokens < max_tokens:
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
                if max_tokens is None or current_tokens + content_tokens < max_tokens:
                    all_content.append(content)
                    current_tokens += content_tokens
                    logger.info(f"✅ Loaded: {md_file.name} ({content_tokens} tokens)")
                else:
                    if max_tokens:
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

    def generate_three_papers_comprehensive(self, topic: str, content: str,
                                          requirements: Dict[str, str]) -> Dict[str, Any]:
        """Generate exactly 3 papers as separate outputs using comprehensive academic prompt"""
        logger.info(f"🎯 Generating comprehensive papers for: {topic}")
        
        # Build the COMPLETE comprehensive academic prompt - NO TRUNCATION
        comprehensive_prompt = self._build_complete_comprehensive_academic_prompt(topic, content, requirements)
        
        try:
            # Generate using Gemini 2.5 Flash with enhanced parameters
            logger.info("🧠 Sending comprehensive academic prompt to Gemini 2.5 Flash...")
            logger.info(f"📏 Prompt length: {len(comprehensive_prompt)} characters")
            
            response = self.gemini_client.generate_content(
                comprehensive_prompt,
                temperature=0.1,
                max_tokens=15000
            )
            
            if not response or len(response) < 500:
                raise ValueError("Insufficient response from Gemini API")
            
            logger.info(f"✅ Received comprehensive response ({len(response)} characters)")
            
            # Parse the comprehensive response
            papers = self._parse_comprehensive_academic_response(response, topic)
            
            # Validate the response with enhanced validation
            validation_results = self.quality_validator.comprehensive_validation(papers)
            papers['validation_results'] = validation_results
            
            if validation_results['overall_valid']:
                logger.info(f"✅ Quality validation passed (score: {validation_results['validation_score']:.2f})")
            else:
                logger.warning(f"⚠️ Quality validation issues (score: {validation_results['validation_score']:.2f})")
                logger.info(f"Suggestions: {validation_results['improvement_suggestions']}")
            
            return papers
            
        except Exception as e:
            logger.error(f"❌ Comprehensive paper generation failed: {e}")
            return self._create_fallback_three_papers(topic, content)

    def _build_complete_comprehensive_academic_prompt(self, topic: str, content: str, requirements: Dict[str, str]) -> str:
        """Build the COMPLETE comprehensive academic assessment creator prompt - NO TRUNCATION"""
        
        # THIS IS THE COMPLETE PROMPT - DO NOT TRUNCATE
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
        """Parse the comprehensive academic assessment response with enhanced validation"""
        logger.info("🔍 Parsing comprehensive academic response")
        
        # Extract the three components using specified delimiters
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
        
        # Enhanced validation
        validation_results = self._validate_academic_content(question_paper, model_answers, marking_scheme)
        
        return {
            "exam_metadata": {
                "title": f"Enhanced Academic Assessment - {topic}",
                "topic": topic,
                "difficulty": "university_level",
                "total_marks": 100,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "generation_method": "enhanced_comprehensive_academic",
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
                "generation_mode": "enhanced_comprehensive_academic",
                "academic_standards": "university_level",
                "format_compliance": validation_results["format_compliance"]
            }
        }

    def _validate_academic_content(self, question_paper: str, model_answers: str, marking_scheme: str) -> Dict[str, Any]:
        """Enhanced validation of academic content"""
        # Check for required question types
        has_conceptual = any(term in question_paper.lower() for term in
                           ['define', 'explain', 'compare', 'analyze', 'discuss', 'evaluate'])
        has_calculation = any(term in question_paper.lower() for term in
                            ['calculate', 'compute', 'algorithm', 'formula', 'mathematical', 'numerical'])
        has_programming = any(term in question_paper.lower() for term in
                            ['code', 'program', 'implement', 'algorithm', 'function', 'script'])
        
        # Enhanced format checking
        has_tabular_answers = '|' in model_answers and 'Question No.' in model_answers
        has_tabular_scheme = '|' in marking_scheme and 'Question No.' in marking_scheme
        
        # Content length validation
        adequate_length = (len(question_paper) > 500 and 
                          len(model_answers) > 800 and 
                          len(marking_scheme) > 400)
        
        # Overall validation with stricter requirements
        overall_valid = (
            adequate_length and
            has_conceptual and
            (has_calculation or has_programming) and
            has_tabular_answers and
            has_tabular_scheme
        )
        
        return {
            "overall_valid": overall_valid,
            "has_conceptual": has_conceptual,
            "has_calculation": has_calculation,
            "has_programming": has_programming,
            "tabular_answers": has_tabular_answers,
            "tabular_scheme": has_tabular_scheme,
            "format_compliance": has_tabular_answers and has_tabular_scheme,
            "content_length_adequate": adequate_length,
            "quality_score": self._calculate_quality_score(
                has_conceptual, has_calculation, has_programming, 
                has_tabular_answers, has_tabular_scheme, adequate_length
            )
        }
    
    def _calculate_quality_score(self, conceptual, calculation, programming, 
                               tabular_answers, tabular_scheme, adequate_length) -> float:
        """Calculate overall quality score"""
        weights = {
            'conceptual': 0.2,
            'calculation': 0.15,
            'programming': 0.15,
            'tabular_answers': 0.2,
            'tabular_scheme': 0.2,
            'adequate_length': 0.1
        }
        
        scores = {
            'conceptual': int(conceptual),
            'calculation': int(calculation),
            'programming': int(programming),
            'tabular_answers': int(tabular_answers),
            'tabular_scheme': int(tabular_scheme),
            'adequate_length': int(adequate_length)
        }
        
        return sum(weights[key] * scores[key] for key in weights.keys())

    def _extract_question_section_fallback(self, response: str) -> str:
        """Extract question paper section using fallback methods"""
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
        
        return "\n\n".join(questions) if questions else response[:len(response)//3]

    def _extract_answers_section_fallback(self, response: str) -> str:
        """Extract model answers section using fallback methods"""
        answer_section = re.search(r'MODEL ANSWER.*?(?=MARKING SCHEME|$)', response, re.DOTALL | re.IGNORECASE)
        return answer_section.group(0) if answer_section else response[len(response)//3:2*len(response)//3]

    def _extract_scheme_section_fallback(self, response: str) -> str:
        """Extract marking scheme section using fallback methods"""
        scheme_section = re.search(r'MARKING SCHEME.*?$', response, re.DOTALL | re.IGNORECASE)
        return scheme_section.group(0) if scheme_section else response[2*len(response)//3:]

    def _count_questions_in_response(self, response: str) -> int:
        """Count questions in response"""
        question_patterns = [r'Q\d+', r'Question \d+', r'^\d+[\.\)]']
        total_questions = 0
        
        for pattern in question_patterns:
            matches = re.findall(pattern, response, re.MULTILINE | re.IGNORECASE)
            if matches:
                total_questions = max(total_questions, len(matches))
        
        return total_questions

    def _create_fallback_three_papers(self, topic: str, content: str) -> Dict[str, Any]:
        """Create basic fallback papers when main generation fails"""
        logger.info("🚨 Creating basic fallback response")
        
        return {
            "exam_metadata": {
                "title": f"Basic Fallback Assessment - {topic}",
                "topic": topic,
                "difficulty": "university_level",
                "total_marks": 100,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "generation_method": "basic_fallback"
            },
            "question_paper_content": f"""
# EXAMINATION PAPER - {topic}
Time: 3 hours | Total Marks: 100

Q1. Explain key concepts of {topic}. (25 marks)
Q2. Calculate relevant metrics for {topic}. (25 marks)  
Q3. Implement solution for {topic}. (25 marks)
Q4. Analyze applications of {topic}. (25 marks)
""",
            "model_answers_content": """
| Question No. | Solutions | Marks |
|--------------|-----------|-------|
| Q1 | Comprehensive conceptual explanation required | 25 |
| Q2 | Complete calculations with working | 25 |
| Q3 | Full code implementation | 25 |
| Q4 | Critical analysis of applications | 25 |
""",
            "marking_schemes_content": """
| Question No. | Marking Criteria | Marks |
|--------------|------------------|-------|
| Q1 | Understanding (15) + Examples (10) | 25 |
| Q2 | Method (15) + Accuracy (10) | 25 |
| Q3 | Code (15) + Documentation (10) | 25 |
| Q4 | Analysis (15) + Examples (10) | 25 |
""",
            "generation_stats": {
                "questions_generated": 4,
                "generation_mode": "basic_fallback"
            }
        }

    # Keep all other existing methods from original implementation
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
            requirements = {
                "question_requirements": "Generate comprehensive university-level questions covering conceptual, computational, and practical aspects",
                "answer_requirements": "Provide detailed model answers with step-by-step solutions and explanations",
                "marking_requirements": "Create detailed marking schemes with clear criteria and mark allocation"
            }

            return self.generate_three_papers_comprehensive(topic, all_content, requirements)

        except Exception as e:
            logger.error(f"❌ Single prompt generation failed: {e}")
            return self._create_fallback_exam_response(topic, all_content)

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
        return self._create_fallback_three_papers(topic, content)

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
                    file_name = f"enhanced_{comp_name}_{timestamp}.txt"
                    file_path = output_dir / file_name
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    saved_files.append(str(file_path))

            elif format_lower == 'json':
                json_file = output_dir / f"enhanced_complete_exam_{timestamp}.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(exam_data, f, indent=2, ensure_ascii=False)
                saved_files.append(str(json_file))

            elif format_lower == 'md':
                for comp_name, content in components.items():
                    md_content = f"# Enhanced {comp_name.title()} - {exam_data.get('exam_metadata', {}).get('topic', 'Exam')}\n\n{content}"
                    file_name = f"enhanced_{comp_name}_{timestamp}.md"
                    file_path = output_dir / file_name
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(md_content)
                    saved_files.append(str(file_path))

        logger.info(f"💾 Saved enhanced exam outputs: {len(saved_files)} files")
        return saved_files

# Export the main class
__all__ = ['SinglePromptExamGenerator']
