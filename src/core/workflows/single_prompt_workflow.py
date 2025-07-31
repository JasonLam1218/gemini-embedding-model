#!/usr/bin/env python3
"""
Single-prompt workflow manager for comprehensive exam generation.
Handles the complete pipeline from PDFs to three separate papers.
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

class SinglePromptWorkflow:
    """Complete workflow manager for single-prompt exam generation"""
    
    def __init__(self):
        self.text_loader = TextLoader()
        self.chunker = TextChunker()
        self.embedding_generator = EmbeddingGenerator()
        self.exam_generator = SinglePromptExamGenerator()
        self.content_aggregator = ContentAggregator()
        
        # Define paths - Use existing structure
        self.input_dir = Path("data/input")
        self.output_dir = Path("data/output")
        self.converted_dir = self.output_dir / "converted_markdown"  # Use existing
        self.embeddings_dir = self.output_dir / "processed"         # Use existing
        self.papers_dir = self.output_dir / "generated_exams"       # Use existing
        
        logger.info("✅ Single Prompt Workflow initialized")

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
        
        # Generate embeddings
        embeddings_data = []
        for chunk in all_chunks:
            try:
                embedding = self.embedding_generator.generate_single_embedding(chunk["chunk_text"])
                if embedding:
                    chunk["embedding"] = embedding
                    embeddings_data.append(chunk)
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate embedding for chunk {chunk['id']}: {e}")
        
        # Save embeddings
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        embeddings_file = self.embeddings_dir / "embeddings.json"
        
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"🧠 Generated and saved {len(embeddings_data)} embeddings")
        return embeddings_data

    def _aggregate_content_for_prompt(self, embeddings_data: List[Dict], topic: str) -> str:
        """Step 4: Aggregate content optimally for single prompt"""
        return self.content_aggregator.aggregate_for_single_prompt(
            embeddings_data, topic, max_tokens=800000
        )

    def _generate_three_papers(self, topic: str, content: str, requirements_file: Optional[str]) -> Dict:
        """Step 5: Generate three papers using comprehensive academic prompt"""
        try:
            # Load requirements if provided (optional for comprehensive prompt)
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
            
            # Save the three papers separately
            saved_files = self._save_three_papers(exam_result, topic)
            
            exam_result["saved_files"] = saved_files
            return exam_result
            
        except Exception as e:
            logger.error(f"❌ Comprehensive academic paper generation failed: {e}")
            raise

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
        """Save the three papers as separate files"""
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        papers = {
            "question_paper": exam_result.get("question_paper_content", ""),
            "model_answers": exam_result.get("model_answers_content", ""),
            "marking_scheme": exam_result.get("marking_schemes_content", "")
        }
        
        saved_files = []
        formats = ["txt", "md"]
        
        for paper_type, content in papers.items():
            for format_type in formats:
                filename = f"comprehensive_{paper_type}_{timestamp}.{format_type}"
                filepath = self.papers_dir / filename
                
                if format_type == "txt":
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                elif format_type == "md":
                    md_content = f"# {paper_type.replace('_', ' ').title()} - {topic}\n\n{content}"
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(md_content)
                
                saved_files.append(str(filepath))
                logger.info(f"💾 Saved: {filename}")
        
        # Also save complete JSON
        json_filename = f"comprehensive_complete_exam_{timestamp}.json"
        json_filepath = self.papers_dir / json_filename
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(exam_result, f, indent=2, ensure_ascii=False)
        saved_files.append(str(json_filepath))
        logger.info(f"💾 Saved: {json_filename}")
        
        return saved_files

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            "input_directory": str(self.input_dir),
            "output_directory": str(self.output_dir),
            "converted_files": len(list(self.converted_dir.rglob("*.md"))) if self.converted_dir.exists() else 0,
            "embedding_files": len(list(self.embeddings_dir.rglob("*.json"))) if self.embeddings_dir.exists() else 0,
            "generated_papers": len(list(self.papers_dir.rglob("*"))) if self.papers_dir.exists() else 0
        }
