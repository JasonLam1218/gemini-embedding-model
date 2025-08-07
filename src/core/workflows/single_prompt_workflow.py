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
from scripts.direct_convert import convert_all_pdfs

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
        formats = ["txt", "md", "pdf"]  # Added PDF format
        
        for paper_type, content in papers.items():
            for format_type in formats:
                filename = f"comprehensive_{paper_type}_{timestamp}.{format_type}"
                filepath = self.papers_dir / filename
                
                try:
                    if format_type == "txt":
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                            
                    elif format_type == "md":
                        md_content = f"# {paper_type.replace('_', ' ').title()} - {topic}\n\n{content}"
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(md_content)
                            
                    elif format_type == "pdf":
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
            
            # Generate PDF
            weasyprint.HTML(string=html_content).write_pdf(str(filepath))
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
            body {
                font-family: 'Times New Roman', serif;
                font-size: 12pt;
                line-height: 1.6;
                margin: 1in;
                color: #333;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid #333;
                padding-bottom: 20px;
            }
            .title {
                font-size: 18pt;
                font-weight: bold;
                color: #1a5490;
                margin-bottom: 10px;
            }
            .subtitle {
                font-size: 12pt;
                color: #666;
            }
            .section {
                margin-bottom: 20px;
            }
            .question {
                margin-bottom: 15px;
                page-break-inside: avoid;
            }
            .marks {
                font-weight: bold;
                color: #d9534f;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ccc;
                padding: 8px;
                text-align: left;
                vertical-align: top;
            }
            th {
                background-color: #f5f5f5;
                font-weight: bold;
            }
            .page-break {
                page-break-before: always;
            }
        </style>
        """
        
        # Convert content to HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{paper_type.replace('_', ' ').title()} - {topic}</title>
            {css_styles}
        </head>
        <body>
            <div class="header">
                <div class="title">{paper_type.replace('_', ' ').title()}</div>
                <div class="subtitle">Subject: {topic}</div>
                <div class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
            <div class="content">
                {self._convert_content_to_html(content)}
            </div>
        </body>
        </html>
        """
        
        return html_content

    def _convert_content_to_html(self, content: str) -> str:
        """Convert plain text content to formatted HTML"""
        import re
        
        # Convert line breaks
        html_content = content.replace('\n', '<br>')
        
        # Convert questions (Q1, Q2, etc.)
        html_content = re.sub(r'\b(Q\d+[.\)])', r'<strong>\1</strong>', html_content)
        
        # Convert marks notation
        html_content = re.sub(r'\((\d+)\s*marks?\)', r'<span class="marks">(\1 marks)</span>', html_content, flags=re.IGNORECASE)
        
        # Convert section headers (SECTION A, etc.)
        html_content = re.sub(r'\b(SECTION [A-Z][^<]*)', r'<h2>\1</h2>', html_content)
        
        # Convert tables (if content contains pipe-separated tables)
        if '|' in content and 'Question No.' in content:
            html_content = self._convert_tables_to_html(html_content)
        
        return html_content

    def _convert_tables_to_html(self, content: str) -> str:
        """Convert pipe-separated tables to HTML tables"""
        import re
        
        # Find table patterns
        table_pattern = r'(\|[^|]*\|(?:\s*\|[^|]*\|)*)'
        tables = re.findall(table_pattern, content, re.MULTILINE)
        
        for table_text in tables:
            rows = table_text.strip().split('\n')
            if len(rows) >= 2:  # Has header and at least one row
                html_table = '<table>'
                
                # Process header
                header_cells = [cell.strip() for cell in rows[0].split('|')[1:-1]]
                html_table += '<tr>' + ''.join(f'<th>{cell}</th>' for cell in header_cells) + '</tr>'
                
                # Process data rows
                for row in rows[1:]:
                    if '|' in row:
                        data_cells = [cell.strip() for cell in row.split('|')[1:-1]]
                        html_table += '<tr>' + ''.join(f'<td>{cell}</td>' for cell in data_cells) + '</tr>'
                
                html_table += '</table>'
                content = content.replace(table_text, html_table)
        
        return content

    def _add_content_to_pdf_story(self, content: str, story: list, heading_style, content_style):
        """Add formatted content to ReportLab story"""
        from reportlab.platypus import Paragraph, Spacer
        import re
        
        # Split content into sections
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if it's a section header
            if re.match(r'^(SECTION [A-Z]|Q\d+)', line):
                story.append(Spacer(1, 12))
                story.append(Paragraph(line, heading_style))
            else:
                # Regular content
                story.append(Paragraph(line, content_style))

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            "input_directory": str(self.input_dir),
            "output_directory": str(self.output_dir),
            "converted_files": len(list(self.converted_dir.rglob("*.md"))) if self.converted_dir.exists() else 0,
            "embedding_files": len(list(self.embeddings_dir.rglob("*.json"))) if self.embeddings_dir.exists() else 0,
            "generated_papers": len(list(self.papers_dir.rglob("*"))) if self.papers_dir.exists() else 0,
            "pdf_generation_available": WEASYPRINT_AVAILABLE or REPORTLAB_AVAILABLE
        }

    def cleanup_workflow_data(self):
        """Clean up intermediate workflow data"""
        try:
            # Clean up lock files
            lock_file = Path("data/output/pipeline.lock")
            if lock_file.exists():
                lock_file.unlink()
                logger.info("🧹 Cleaned up pipeline lock file")
            
            # Optional: Clean up old session logs (keep last 5)
            logs_dir = Path("data/output/logs")
            if logs_dir.exists():
                session_logs = list(logs_dir.glob("session_*.log"))
                if len(session_logs) > 5:
                    # Sort by modification time and remove oldest
                    session_logs.sort(key=lambda x: x.stat().st_mtime)
                    for old_log in session_logs[:-5]:
                        old_log.unlink()
                        logger.info(f"🧹 Cleaned up old log: {old_log.name}")
            
            logger.info("✅ Workflow cleanup completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Workflow cleanup failed: {e}")

    def validate_workflow_integrity(self) -> Dict[str, Any]:
        """Validate the integrity of the workflow"""
        validation = {
            "directories_exist": True,
            "required_files": True,
            "dependencies_available": True,
            "issues": []
        }
        
        # Check directories
        required_dirs = [self.input_dir, self.output_dir, self.converted_dir, self.embeddings_dir, self.papers_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                validation["directories_exist"] = False
                validation["issues"].append(f"Missing directory: {dir_path}")
        
        # Check PDF generation capabilities
        if not (WEASYPRINT_AVAILABLE or REPORTLAB_AVAILABLE):
            validation["dependencies_available"] = False
            validation["issues"].append("PDF generation not available - install weasyprint or reportlab")
        
        # Check for input files
        input_pdfs = len(list(self.input_dir.rglob("*.pdf"))) if self.input_dir.exists() else 0
        converted_mds = len(list(self.converted_dir.rglob("*.md"))) if self.converted_dir.exists() else 0
        
        if input_pdfs == 0 and converted_mds == 0:
            validation["required_files"] = False
            validation["issues"].append("No input PDF files or converted markdown files found")
        
        validation["overall_status"] = (
            validation["directories_exist"] and 
            validation["required_files"] and 
            validation["dependencies_available"]
        )
        
        return validation
