#!/usr/bin/env python3
"""
Single-prompt workflow manager for comprehensive exam generation.
Handles the complete pipeline from PDFs to three separate papers with PDF output support.
Enhanced with all fixes for content aggregation, rate limiting, and error handling.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger
from datetime import datetime

from ..text.text_loader import TextLoader
from ..text.chunker import TextChunker
from ..embedding.embedding_generator import EmbeddingGenerator
from ..generation.single_prompt_generator import SinglePromptExamGenerator
from ..content.content_aggregator import ContentAggregator
# Safe import for PDF conversion
import sys
from pathlib import Path

def _import_convert_function():
    """Safely import the convert_all_pdfs function"""
    try:
        from scripts.direct_convert import convert_all_pdfs
        return convert_all_pdfs
    except ImportError:
        scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        try:
            from direct_convert import convert_all_pdfs
            return convert_all_pdfs
        except ImportError:
            logger.warning("⚠️ PDF conversion function not available")
            def fallback_convert():
                logger.info("Using existing markdown files - PDF conversion skipped")
                return
            return fallback_convert

convert_all_pdfs = _import_convert_function()

# PDF generation imports with fallback handling
try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    logger.warning("⚠️ WeasyPrint not available - PDF generation will use ReportLab only")

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("⚠️ ReportLab not available - PDF generation disabled")

class SinglePromptWorkflow:
    """Complete workflow manager for single-prompt exam generation with PDF support"""
    
    def __init__(self):
        self.text_loader = TextLoader()
        self.chunker = TextChunker()
        self.embedding_generator = EmbeddingGenerator()
        self.exam_generator = SinglePromptExamGenerator()
        self.content_aggregator = ContentAggregator()
        
        # Define paths - Use existing structure
        self.input_dir = Path("data/input")
        self.output_dir = Path("data/output")
        self.converted_dir = self.output_dir / "converted_markdown"
        self.embeddings_dir = self.output_dir / "processed"
        self.papers_dir = self.output_dir / "generated_exams"
        
        logger.info("✅ Single Prompt Workflow initialized with PDF generation support")

    def execute_full_workflow(self, topic: str, requirements_file: Optional[str] = None) -> Dict[str, Any]:
        """Execute the complete workflow from PDFs to final papers"""
        start_time = time.time()
        logger.info(f"🚀 Starting complete single-prompt workflow for topic: {topic}")
        
        try:
            # Step 1: Check if PDFs are already converted
            logger.info("📄 STEP 1/5: Checking PDF conversion")
            if not self.converted_dir.exists() or len(list(self.converted_dir.rglob("*.md"))) == 0:
                logger.info("Converting PDFs to Markdown...")
                convert_all_pdfs()
            else:
                logger.info("✅ Markdown files already exist")
            
            # Step 2: Load and process markdown content
            logger.info("📝 STEP 2/5: Processing markdown content")
            documents = self._process_markdown_content()
            
            # Step 3: Check for existing embeddings or generate new ones
            logger.info("🧠 STEP 3/5: Loading/generating embeddings")
            embeddings_data = self._load_or_generate_embeddings(documents)
            
            # Step 4: Aggregate content for single prompt
            logger.info("📋 STEP 4/5: Aggregating content for single prompt")
            aggregated_content = self._aggregate_content_for_prompt(embeddings_data, topic)
            
            # Step 5: Generate three papers using single prompt
            logger.info("🎯 STEP 5/5: Generating three papers")
            papers_result = self._generate_three_papers(topic, aggregated_content, requirements_file)
            
            # Calculate final statistics
            duration = time.time() - start_time
            
            # Save workflow results
            workflow_result = {
                "workflow_metadata": {
                    "topic": topic,
                    "duration_seconds": round(duration, 2),
                    "timestamp": datetime.now().isoformat(),
                    "success": True
                },
                "processing_stats": {
                    "documents_processed": len(documents),
                    "embeddings_generated": len(embeddings_data),
                    "content_sections": len(aggregated_content.split("==="))
                },
                "generated_papers": papers_result,
                "output_files": papers_result.get("saved_files", [])
            }
            
            logger.info(f"✅ Workflow completed successfully in {duration:.2f} seconds")
            return workflow_result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ Workflow failed after {duration:.2f} seconds: {e}")
            return {
                "workflow_metadata": {
                    "topic": topic,
                    "duration_seconds": round(duration, 2),
                    "timestamp": datetime.now().isoformat(),
                    "success": False,
                    "error": str(e)
                }
            }

    def _process_markdown_content(self) -> List:
        """Step 2: Load and process all markdown content"""
        documents = self.text_loader.process_directory(self.converted_dir)
        if not documents:
            raise ValueError("No markdown documents found to process")
        logger.info(f"📄 Processed {len(documents)} markdown documents")
        return documents

    def _load_or_generate_embeddings(self, documents: List) -> List[Dict]:
        """Step 3: Load existing embeddings or generate new ones"""
        embeddings_file = self.embeddings_dir / "embeddings.json"
        
        if embeddings_file.exists():
            logger.info("📥 Loading existing embeddings")
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
            logger.info(f"✅ Loaded {len(embeddings_data)} existing embeddings")
            return embeddings_data
        else:
            logger.info("🧠 Generating new embeddings...")
            return self._generate_content_embeddings(documents)

    def _generate_content_embeddings(self, documents: List) -> List[Dict]:
        """Generate embeddings for all content"""
        all_chunks = []
        
        for doc in documents:
            # Chunk the document
            chunks = self.chunker.chunk_text(doc.content)
            
            for i, chunk_text in enumerate(chunks):
                chunk_data = {
                    "id": f"{doc.paper_set}_{doc.paper_number}_{i}",
                    "chunk_text": chunk_text,
                    "chunk_index": i,
                    "source_file": doc.source_file,
                    "content_type": doc.content_type,
                    "paper_set": doc.paper_set,
                    "metadata": doc.metadata
                }
                all_chunks.append(chunk_data)
        
        # Generate embeddings using batch processing
        embeddings_data = []
        batch_size = 5  # Conservative batch size
        
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_texts = [chunk["chunk_text"] for chunk in batch_chunks]
            
            try:
                # Use the enhanced batch processing from embedding_generator
                batch_results = self.embedding_generator.process_chunks_batch(batch_texts, batch_size)
                
                for j, result in enumerate(batch_results):
                    if result.get('success', False):
                        chunk_data = batch_chunks[j].copy()
                        chunk_data["embedding"] = result['embedding']
                        chunk_data["embedding_model"] = "text-embedding-004"
                        embeddings_data.append(chunk_data)
                    else:
                        logger.warning(f"⚠️ Failed to generate embedding for chunk {batch_chunks[j]['id']}")
                        
            except Exception as e:
                logger.error(f"❌ Batch embedding generation failed: {e}")
                # Continue with next batch
                continue
        
        # Save embeddings
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        embeddings_file = self.embeddings_dir / "embeddings.json"
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"🧠 Generated and saved {len(embeddings_data)} embeddings")
        return embeddings_data

    def _aggregate_content_for_prompt(self, embeddings_data: List[Dict], topic: str) -> str:
        """Step 4: Aggregate content optimally for single prompt"""
        logger.info(f"📋 Starting content aggregation for {len(embeddings_data)} embeddings")
        
        # Use content aggregator with validation
        aggregated_content = self.content_aggregator.aggregate_for_single_prompt(
            embeddings_data, topic, max_tokens=800000
        )
        
        # Validate aggregated content
        validation = self.content_aggregator.validate_aggregated_content(aggregated_content)
        
        logger.info(f"📊 Content validation results:")
        logger.info(f"  • Length adequate: {validation['length_adequate']}")
        logger.info(f"  • Has exam content: {validation['has_exam_content']}")
        logger.info(f"  • Has lecture content: {validation['has_lecture_content']}")
        logger.info(f"  • Content sections: {validation['content_sections']}")
        logger.info(f"  • Total characters: {validation['total_characters']}")
        logger.info(f"  • Overall valid: {validation['overall_valid']}")
        
        if not validation['overall_valid']:
            logger.warning("⚠️ Content validation failed, attempting direct markdown loading")
            aggregated_content = self.content_aggregator._fallback_load_content()
        
        return aggregated_content

    def _generate_three_papers(self, topic: str, content: str, requirements_file: Optional[str]) -> Dict:
        """Step 5: Generate three papers using comprehensive academic prompt"""
        
        # Validate content before sending to API
        if not content or len(content.strip()) < 1000:
            logger.error(f"❌ Insufficient content for generation: {len(content)} characters")
            logger.info("🔄 Attempting direct content loading fallback")
            
            # Fallback: Load content directly
            try:
                content = self.exam_generator.load_all_converted_markdown()
                if not content or len(content.strip()) < 1000:
                    raise ValueError("Fallback content loading also failed")
                logger.info(f"✅ Fallback content loaded: {len(content)} characters")
            except Exception as e:
                logger.error(f"❌ All content loading methods failed: {e}")
                return self._create_emergency_fallback_response(topic)
        
        try:
            # Load requirements if provided
            requirements = self._load_requirements(requirements_file)
            
            logger.info("🎯 Using comprehensive academic assessment creator approach")
            
            # Generate comprehensive exam using the enhanced academic prompt
            exam_result = self.exam_generator.generate_three_papers_comprehensive(
                topic=topic,
                content=content,
                requirements=requirements
            )
            
            # Enhanced validation logging
            if exam_result.get('exam_metadata', {}).get('validation_passed', False):
                logger.info("✅ Comprehensive academic assessment validation passed")
            else:
                logger.warning("⚠️ Academic assessment validation had issues")
            
            # Save the three papers separately with PDF support
            saved_files = self._save_three_papers(exam_result, topic)
            exam_result["saved_files"] = saved_files
            
            return exam_result
            
        except Exception as e:
            logger.error(f"❌ Comprehensive academic paper generation failed: {e}")
            return self._create_emergency_fallback_response(topic)

    def _create_emergency_fallback_response(self, topic: str) -> Dict:
        """Create emergency fallback when all content loading fails"""
        logger.info("🚨 Creating emergency fallback response")
        
        return {
            "exam_metadata": {
                "title": f"Emergency Fallback - {topic}",
                "topic": topic,
                "difficulty": "university_level",
                "total_marks": 100,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "generation_method": "emergency_fallback"
            },
            "question_paper_content": f"""
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
            """,
            "model_answers_content": """
MODEL ANSWERS

Question No. | Sub-part | Solutions | Marks
Q1 | - | Comprehensive explanation covering fundamental definitions, theoretical principles, and practical applications. Students should demonstrate deep understanding of core concepts and their interconnections. | 15
Q2 | - | Detailed comparison of different methodologies, including advantages, disadvantages, and situational appropriateness. Critical evaluation of effectiveness and efficiency. | 15
Q3 | - | Step-by-step mathematical solution with clear methodology, accurate calculations, and proper justification of approach. Include error checking and validation. | 20
Q4 | - | Complete complexity analysis including time and space complexity, performance metrics, and optimization considerations with supporting calculations. | 15
Q5 | - | Full code solution with proper structure, documentation, error handling, and testing framework. Include explanation of design decisions. | 20
Q6 | - | Identified issues, proposed improvements, optimized code, and performance impact analysis with before/after comparisons. | 15
            """,
            "marking_schemes_content": """
MARKING SCHEME

Question No. | Sub-part | Marking Criteria | Marks
Q1 | - | Accurate definitions (5), Theoretical explanation (5), Practical examples (3), Clear presentation (2) | 15
Q2 | - | Identification of approaches (4), Strengths and limitations (6), Critical evaluation (3), Conclusion (2) | 15
Q3 | - | Correct methodology (6), Mathematical accuracy (8), Clear working (4), Final answer (2) | 20
Q4 | - | Complexity calculation (6), Performance metrics (5), Optimization discussion (4) | 15
Q5 | - | Code correctness (8), Documentation (4), Structure and style (4), Testing approach (4) | 20
Q6 | - | Problem identification (5), Solution implementation (6), Performance analysis (4) | 15
            """,
            "generation_stats": {
                "questions_generated": 6,
                "generation_mode": "emergency_fallback",
                "api_failed": True
            }
        }

    def _load_requirements(self, requirements_file: Optional[str]) -> Dict:
        """Load requirements from file or use defaults"""
        if requirements_file and Path(requirements_file).exists():
            try:
                with open(requirements_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load requirements file: {e}")
        
        # Return default requirements for comprehensive academic assessment
        return {
            "question_requirements": "Generate comprehensive university-level questions covering conceptual, computational, and practical aspects with mandatory inclusion of diverse question types",
            "answer_requirements": "Provide detailed model answers with step-by-step solutions, explanations, and exact tabular format as specified",
            "marking_requirements": "Create detailed marking schemes with clear criteria, mark allocation, and same tabular format as model answers"
        }

    def _save_three_papers(self, exam_result: Dict, topic: str) -> List[str]:
        """Save the three papers as separate files including PDF format"""
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        papers = {
            "question_paper": exam_result.get("question_paper_content", ""),
            "model_answers": exam_result.get("model_answers_content", ""),
            "marking_scheme": exam_result.get("marking_schemes_content", "")
        }
        
        saved_files = []
        formats = ["pdf"]  # Added PDF format
        
        for paper_type, content in papers.items():
            if not content.strip():
                logger.warning(f"⚠️ Empty content for {paper_type}, skipping...")
                continue
                
            for format_type in formats:
                filename = f"comprehensive_{paper_type}_{timestamp}.{format_type}"
                filepath = self.papers_dir / filename
                
                try:
                    # export pdf file only 
                    if format_type == "pdf":
                        # Generate PDF using the new method
                        self._generate_pdf_file(content, filepath, paper_type, topic)
                    
                    saved_files.append(str(filepath))
                    logger.info(f"💾 Saved: {filename}")
                except Exception as e:
                    logger.error(f"❌ Failed to save {filename}: {e}")
                    continue
        
        # Save complete JSON (unchanged)
        json_filename = f"comprehensive_complete_exam_{timestamp}.json"
        json_filepath = self.papers_dir / json_filename
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(exam_result, f, indent=2, ensure_ascii=False)
        saved_files.append(str(json_filepath))
        logger.info(f"💾 Saved: {json_filename}")
        
        return saved_files

    def _generate_pdf_file(self, content: str, filepath: Path, paper_type: str, topic: str):
        """Generate PDF file from content using multiple approaches"""
        try:
            # Try WeasyPrint first (better HTML/CSS support)
            if WEASYPRINT_AVAILABLE and self._generate_pdf_with_weasyprint(content, filepath, paper_type, topic):
                return
        except Exception as e:
            logger.warning(f"⚠️ WeasyPrint failed: {e}, trying ReportLab")
        
        try:
            # Fallback to ReportLab
            if REPORTLAB_AVAILABLE:
                self._generate_pdf_with_reportlab(content, filepath, paper_type, topic)
            else:
                logger.error("❌ No PDF generation libraries available")
                raise ImportError("Neither WeasyPrint nor ReportLab available")
        except Exception as e:
            logger.error(f"❌ All PDF generation methods failed: {e}")
            raise

    def _generate_pdf_with_weasyprint(self, content: str, filepath: Path, paper_type: str, topic: str) -> bool:
        """Generate PDF using WeasyPrint (preferred method)"""
        try:
            import weasyprint
            # Convert content to HTML with proper styling
            html_content = self._format_content_as_html(content, paper_type, topic)
            
            # Generate PDF with custom options
            html_doc = weasyprint.HTML(string=html_content)
            html_doc.write_pdf(
                str(filepath),
                stylesheets=None,
                presentational_hints=True,
                optimize_images=True
            )
            
            logger.info(f"✅ PDF generated with WeasyPrint: {filepath.name}")
            return True
        except ImportError:
            raise ImportError("WeasyPrint not installed")
        except Exception as e:
            logger.error(f"❌ WeasyPrint PDF generation failed: {e}")
            return False

    def _generate_pdf_with_reportlab(self, content: str, filepath: Path, paper_type: str, topic: str):
        """Generate PDF using ReportLab (fallback method)"""
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        
        # Create PDF document
        doc = SimpleDocTemplate(str(filepath), pagesize=A4, 
                               rightMargin=72, leftMargin=72, 
                               topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Create custom styles for academic papers
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1  # Center alignment
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        content_style = ParagraphStyle(
            'CustomContent',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            leftIndent=0
        )
        
        # Build PDF content
        story = []
        
        # Add header
        story.append(Paragraph(f"{paper_type.replace('_', ' ').title()}", title_style))
        story.append(Paragraph(f"Subject: {topic}", content_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", content_style))
        story.append(Spacer(1, 20))
        
        # Process content and add to story
        self._add_content_to_pdf_story(content, story, heading_style, content_style)
        
        # Build PDF
        doc.build(story)
        logger.info(f"✅ PDF generated with ReportLab: {filepath.name}")

    def _format_content_as_html(self, content: str, paper_type: str, topic: str) -> str:
        """Format content as HTML for WeasyPrint"""
        # CSS styling for academic papers
        css_styles = """
        <style>
        @page {
            size: A4;
            margin: 2cm;
            @top-center {
                content: string(document-title);
                font-size: 10pt;
                color: #666;
            }
            @bottom-center {
                content: "Page " counter(page) " of " counter(pages);
                font-size: 10pt;
                color: #666;
            }
        }
        
        body { 
            font-family: 'Times New Roman', serif; 
            font-size: 12pt;
            line-height: 1.4;
            color: #000;
            margin: 0;
            padding: 0;
        }
        
        h1 { 
            color: #1f4e79; 
            text-align: center; 
            font-size: 18pt;
            margin-bottom: 20pt;
            page-break-after: avoid;
            string-set: document-title content();
        }
        
        h2 { 
            color: #2f5f8f; 
            border-bottom: 2px solid #ccc; 
            font-size: 14pt;
            margin-top: 20pt;
            margin-bottom: 10pt;
            page-break-after: avoid;
        }
        
        h3 {
            color: #4f7f9f;
            font-size: 12pt;
            margin-top: 15pt;
            margin-bottom: 8pt;
            page-break-after: avoid;
        }
        
        .header { 
            background-color: #f5f5f5; 
            padding: 15pt; 
            margin-bottom: 20pt; 
            border: 1px solid #ddd;
            border-radius: 5pt;
        }
        
        .content {
            text-align: justify;
        }
        
        p {
            margin-bottom: 8pt;
            text-indent: 0;
        }
        
        ul, ol {
            margin-left: 20pt;
            margin-bottom: 10pt;
        }
        
        li {
            margin-bottom: 5pt;
        }
        
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin: 15pt 0;
            page-break-inside: avoid;
        }
        
        th, td { 
            border: 1px solid #ddd; 
            padding: 8pt; 
            text-align: left; 
            vertical-align: top;
        }
        
        th { 
            background-color: #f2f2f2; 
            font-weight: bold;
        }
        
        .question-number {
            font-weight: bold;
            color: #1f4e79;
        }
        
        .marks {
            font-style: italic;
            color: #666;
        }
        
        .page-break {
            page-break-before: always;
        }
        
        blockquote {
            margin-left: 20pt;
            padding-left: 10pt;
            border-left: 3pt solid #ccc;
            font-style: italic;
        }
        
        code {
            font-family: 'Courier New', monospace;
            background-color: #f5f5f5;
            padding: 2pt;
            border-radius: 2pt;
        }
        
        pre {
            font-family: 'Courier New', monospace;
            background-color: #f5f5f5;
            padding: 10pt;
            border-radius: 5pt;
            overflow-wrap: break-word;
            white-space: pre-wrap;
        }
        </style>
        """
        
        # Convert content to HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{paper_type.replace('_', ' ').title()} - {topic}</title>
            {css_styles}
        </head>
        <body>
            <h1>{paper_type.replace('_', ' ').title()}</h1>
            <div class="header">
                <p><strong>Subject:</strong> {topic}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <div class="content">
                {self._convert_content_to_html(content)}
            </div>
        </body>
        </html>
        """
        
        return html_content

    def _convert_content_to_html(self, content: str) -> str:
        """Convert plain text content to HTML with proper formatting"""
        import re
        
        # Escape HTML characters
        content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Convert line breaks to HTML
        content = content.replace('\n\n', '</p><p>')
        content = content.replace('\n', '<br>')
        
        # Wrap in paragraphs
        content = f'<p>{content}</p>'
        
        # Convert markdown-style headers
        content = re.sub(r'<p>#{3}\s*([^<]+?)</p>', r'<h3>\1</h3>', content)
        content = re.sub(r'<p>#{2}\s*([^<]+?)</p>', r'<h2>\1</h2>', content)
        content = re.sub(r'<p>#{1}\s*([^<]+?)</p>', r'<h1>\1</h1>', content)
        
        # Convert simple table format (pipe-separated)
        content = self._convert_tables_to_html(content)
        
        # Convert question numbers to styled format
        content = re.sub(r'<p>(Q\d+[\.:])', r'<p><span class="question-number">\1</span>', content)
        
        # Convert marks indicators
        content = re.sub(r'\((\d+\s*marks?)\)', r'<span class="marks">(\1)</span>', content, flags=re.IGNORECASE)
        
        return content

    def _convert_tables_to_html(self, content: str) -> str:
        """Convert pipe-separated tables to HTML tables"""
        import re
        
        # Find table patterns (lines with multiple |)
        table_pattern = r'<p>([^<]*\|[^<]*\|[^<]*)</p>'
        
        def convert_table_match(match):
            table_content = match.group(1)
            lines = table_content.split('<br>')
            
            if len(lines) < 2:
                return match.group(0)  # Not a table
            
            html_table = '<table>'
            
            # Process each line
            for i, line in enumerate(lines):
                if '|' not in line:
                    continue
                    
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                
                if not cells:
                    continue
                
                if i == 0:  # Header row
                    html_table += '<tr>'
                    for cell in cells:
                        html_table += f'<th>{cell}</th>'
                    html_table += '</tr>'
                else:  # Data row
                    html_table += '<tr>'
                    for cell in cells:
                        html_table += f'<td>{cell}</td>'
                    html_table += '</tr>'
            
            html_table += '</table>'
            return html_table
        
        return re.sub(table_pattern, convert_table_match, content)

    def _add_content_to_pdf_story(self, content: str, story: list, heading_style, content_style):
        """Add content to ReportLab story with proper formatting"""
        from reportlab.platypus import Paragraph
        import re
        
        # Split content into paragraphs
        paragraphs = content.split('\n\n')
        
        for para in paragraphs:
            if not para.strip():
                continue
            
            # Check if it's a heading
            if para.strip().startswith('#'):
                # Remove markdown hash symbols
                heading_text = re.sub(r'^#+\s*', '', para.strip())
                story.append(Paragraph(heading_text, heading_style))
            else:
                # Regular paragraph
                # Clean up the text for ReportLab
                clean_text = para.strip().replace('\n', ' ')
                
                # Handle simple formatting
                clean_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_text)  # Bold
                clean_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', clean_text)      # Italic
                
                try:
                    story.append(Paragraph(clean_text, content_style))
                except Exception as e:
                    # Fallback for problematic text
                    logger.warning(f"⚠️ Problem with paragraph formatting: {e}")
                    safe_text = re.sub(r'[^\w\s\.\,\?\!\:\;\(\)\-\+\=\%\$]', '', clean_text)
                    story.append(Paragraph(safe_text, content_style))
