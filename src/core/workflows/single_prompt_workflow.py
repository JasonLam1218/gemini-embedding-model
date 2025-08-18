#!/usr/bin/env python3
"""
Single-prompt workflow manager for comprehensive exam generation.
Handles the complete pipeline from PDFs to three separate papers with PDF output support.
Enhanced with all fixes for content aggregation, rate limiting, and error handling.
Complete implementation with all existing functionality preserved.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger
from datetime import datetime
import sys
import re

# Add project root to Python path if not already added, useful for imports like config.settings
project_root_dir = Path(__file__).parent.parent.parent.parent
if str(project_root_dir) not in sys.path:
    sys.path.insert(0, str(project_root_dir))

from src.core.text.text_loader import TextLoader
from src.core.text.chunker import TextChunker
from src.core.embedding.embedding_generator import EmbeddingGenerator
from src.core.generation.single_prompt_generator import SinglePromptExamGenerator
from src.core.content.content_aggregator import ContentAggregator
from src.core.storage.vector_store import VectorStore, Document, TextChunk, Embedding # Explicit import

# Import the new conversion functions
# This needs to be a relative import or sys.path adjusted correctly
def _import_convert_function():
    """Safely import the convert_all_pdfs function"""
    try:
        from scripts.direct_convert import convert_pdfs_from_vercel_blobs, convert_all_pdfs_enhanced_from_local
        return convert_pdfs_from_vercel_blobs, convert_all_pdfs_enhanced_from_local
    except ImportError:
        scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        try:
            from scripts.direct_convert import convert_pdfs_from_vercel_blobs, convert_all_pdfs_enhanced_from_local
            return convert_pdfs_from_vercel_blobs, convert_all_pdfs_enhanced_from_local
        except ImportError:
            logger.warning("⚠️ PDF conversion functions not available")
            def fallback_convert_blobs(*args, **kwargs):
                logger.info("Using existing markdown files - PDF conversion from blobs skipped")
                raise RuntimeError("PDF conversion from blobs not available.")
            def fallback_convert_local(*args, **kwargs):
                logger.info("Using existing markdown files - Local PDF conversion skipped")
                raise RuntimeError("Local PDF conversion not available.")
            return fallback_convert_blobs, fallback_convert_local

convert_pdfs_from_vercel_blobs, convert_all_pdfs_enhanced_from_local = _import_convert_function()

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
        self.vector_store = VectorStore() # Initialize VectorStore here
        
        # Define paths - Use existing structure
        self.input_dir = Path("data/input") # Still points to where original PDFs would be
        self.output_dir = Path("data/output")
        self.converted_dir = self.output_dir / "converted_markdown" # This is where all converted markdown will be stored
        self.embeddings_dir = self.output_dir / "processed"
        self.papers_dir = self.output_dir / "generated_exams"
        
        logger.info("✅ Single Prompt Workflow initialized with PDF generation support")

    async def execute_full_workflow(self, topic: str, requirements_file: Optional[str] = None,
                              pdf_blob_urls: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Execute the complete workflow from PDFs (local or Vercel Blobs) to final papers.
        
        Args:
            topic (str): The topic for the exam.
            requirements_file (Optional[str]): Path to a custom requirements JSON file.
            pdf_blob_urls (Optional[List[Dict]]): List of dictionaries containing Vercel Blob URLs
                                                and metadata for PDF files to process.
                                                E.g., [{'url': '...', 'category': 'lectures', 'original_filename': 'lecture1.pdf'}]
        """
        start_time = time.time()
        logger.info(f"🚀 Starting complete single-prompt workflow for topic: {topic}")

        try:
            # Step 1: Handle PDF conversion from local or Vercel Blobs
            logger.info("📄 STEP 1/5: Checking PDF conversion / Fetching from Blob Storage")
            if pdf_blob_urls:
                # Prioritize Vercel Blob URLs if provided
                logger.info(f"Initiating PDF conversion from {len(pdf_blob_urls)} Vercel Blob URLs...")
                await convert_pdfs_from_vercel_blobs(pdf_blob_urls) # Call the new blob conversion function
                logger.info("✅ PDFs downloaded and converted from Blob Storage to Markdown.")
            elif not self.converted_dir.exists() or len(list(self.converted_dir.rglob("*.md"))) == 0:
                # Fallback to local PDF conversion if no blob URLs and no markdown exists
                logger.info("No Vercel Blob URLs provided, and no converted markdown found locally.")
                logger.info("Attempting conversion from local data/input directory...")
                await convert_all_pdfs_enhanced_from_local() # Call the local PDF conversion function
                logger.info("✅ Local PDFs converted to Markdown.")
            else:
                logger.info("✅ Converted markdown files already exist locally, skipping PDF conversion.")

            # Step 2: Load and process markdown content (this now reads from self.converted_dir)
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
            
            # Save generated exam to Supabase
            if papers_result:
                try:
                    exam_id = self.vector_store.save_generated_exam(papers_result)
                    logger.info(f"✅ Generated exam saved to Supabase with ID: {exam_id}")
                    papers_result['exam_metadata']['supabase_exam_id'] = exam_id # Add ID to result
                except Exception as e:
                    logger.error(f"❌ Failed to save generated exam to Supabase: {e}")

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
                "generated_papers": papers_result, # Contains the content of papers and paths to local files
                "output_files": papers_result.get("saved_files", []) # Paths to locally saved PDFs/JSON
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
        
        # Get list of processed chunk IDs from Supabase that already have embeddings
        existing_embedded_chunk_ids = set()
        if self.vector_store:
            try:
                # Assuming vector_store.get_embeddings_count() returns total embeddings, not per chunk.
                # A better way would be to query supabase for chunk_ids that have embeddings.
                # For this, let's assume get_chunks_without_embeddings is robust.
                chunks_without_embeddings_from_db = self.vector_store.get_chunks_without_embeddings()
                # If this list is empty, it implies all chunks in DB have embeddings
                if len(chunks_without_embeddings_from_db) == 0 and self.vector_store.get_chunks_count() > 0:
                    logger.info("✅ All existing chunks in Supabase already have embeddings.")
                    # Load all embeddings from DB if available
                    # Note: There isn't a direct 'get_all_embeddings' in your VectorStore.
                    # For simplicity, we'll assume if local file exists and covers all, we use it.
                    # If you want to force re-fetch from Supabase here, you'd need a new method
                    # in VectorStore to get all embeddings or iterate through chunks and fetch.
                    if embeddings_file.exists():
                        with open(embeddings_file, 'r', encoding='utf-8') as f:
                            embeddings_data = json.load(f)
                        logger.info(f"📥 Loaded {len(embeddings_data)} existing embeddings from local file (Supabase implies complete).")
                        return embeddings_data
                    else:
                         # Fallback to generate if local file is missing even if DB is complete
                         logger.warning("No local embeddings file, generating even if Supabase is complete.")
                
            except Exception as e:
                logger.warning(f"⚠️ Could not check Supabase for existing embeddings: {e}. Proceeding with local file check/generation.")

        if embeddings_file.exists():
            logger.info("📥 Loading existing embeddings from local file.")
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
            logger.info(f"✅ Loaded {len(embeddings_data)} existing embeddings.")
            return embeddings_data
        else:
            logger.info("🧠 Generating new embeddings...")
            return self._generate_content_embeddings(documents)

    def _generate_content_embeddings(self, documents: List) -> List[Dict]:
        """Generate embeddings for all content WITH Supabase integration"""
        
        all_chunks_info = [] # Store chunk data including potential supabase_chunk_id
        chunks_to_embed_text = [] # Store only text for embedding generation
        chunks_to_embed_map = [] # Map index in chunks_to_embed_text back to all_chunks_info

        for doc in documents:
            logger.info(f"🔄 Processing document: {doc.source_file}")
            
            existing_doc = self.vector_store.document_exists_by_source_file(doc.source_file)
            doc_id = None
            if existing_doc:
                logger.info(f"📄 Document exists in Supabase: {doc.source_file}")
                doc_id = existing_doc['id']
                existing_db_chunks = self.vector_store.get_chunks_by_document(doc_id)
                
                # Check which existing chunks need embeddings
                for chunk_data in existing_db_chunks:
                    if not self.vector_store.embedding_exists_for_chunk(chunk_data['id']):
                        chunk_info = {
                            "id": f"{doc.paper_set}_{doc.paper_number}_{chunk_data['chunk_index']}",
                            "chunk_text": chunk_data['chunk_text'],
                            "chunk_index": chunk_data['chunk_index'],
                            "source_file": doc.source_file,
                            "content_type": doc.content_type,
                            "paper_set": doc.paper_set,
                            "metadata": doc.metadata,
                            "supabase_chunk_id": chunk_data['id']
                        }
                        all_chunks_info.append(chunk_info)
                        chunks_to_embed_text.append(chunk_info["chunk_text"])
                        chunks_to_embed_map.append(len(all_chunks_info) - 1) # Map to its position in all_chunks_info
                    else:
                        logger.debug(f"Chunk {chunk_data['id']} already has embedding, skipping.")

            else:
                logger.info(f"📝 Creating new document in Supabase: {doc.source_file}")
                
                supabase_doc = Document(
                    title=Path(doc.source_file).stem,
                    content=doc.content,
                    source_file=doc.source_file,
                    paper_set=doc.paper_set,
                    paper_number=doc.paper_number,
                    metadata=doc.metadata
                )
                
                try:
                    doc_id = self.vector_store.insert_document(supabase_doc)
                    chunks_text_list = self.chunker.chunk_text(doc.content)
                    
                    chunk_objects = [
                        TextChunk(
                            document_id=doc_id,
                            chunk_text=chunk_text,
                            chunk_index=i,
                            chunk_size=len(chunk_text)
                        ) for i, chunk_text in enumerate(chunks_text_list)
                    ]
                    
                    chunk_ids = self.vector_store.insert_text_chunks(chunk_objects)
                    
                    for i, (chunk_text, chunk_id) in enumerate(zip(chunks_text_list, chunk_ids)):
                        chunk_info = {
                            "id": f"{doc.paper_set}_{doc.paper_number}_{i}",
                            "chunk_text": chunk_text,
                            "chunk_index": i,
                            "source_file": doc.source_file,
                            "content_type": doc.content_type,
                            "paper_set": doc.paper_set,
                            "metadata": doc.metadata,
                            "supabase_chunk_id": chunk_id
                        }
                        all_chunks_info.append(chunk_info)
                        chunks_to_embed_text.append(chunk_info["chunk_text"])
                        chunks_to_embed_map.append(len(all_chunks_info) - 1)
                        
                    logger.info(f"✅ Created document and {len(chunks_text_list)} chunks in Supabase for {doc.source_file}")
                    
                except Exception as e:
                    logger.error(f"❌ Supabase document/chunk creation failed for {doc.source_file}: {e}. Proceeding with local-only chunking if possible.")
                    chunks_text_list = self.chunker.chunk_text(doc.content)
                    for i, chunk_text in enumerate(chunks_text_list):
                        chunk_info = {
                            "id": f"{doc.paper_set}_{doc.paper_number}_{i}",
                            "chunk_text": chunk_text,
                            "chunk_index": i,
                            "source_file": doc.source_file,
                            "content_type": doc.content_type,
                            "paper_set": doc.paper_set,
                            "metadata": doc.metadata,
                            "supabase_chunk_id": None
                        }
                        all_chunks_info.append(chunk_info)
                        chunks_to_embed_text.append(chunk_info["chunk_text"])
                        chunks_to_embed_map.append(len(all_chunks_info) - 1)

        logger.info(f"🧠 Generating embeddings for {len(chunks_to_embed_text)} new/unembedded chunks.")
        
        generated_embeddings_results = self.embedding_generator.process_chunks_batch(
            chunks_to_embed_text, batch_size=5
        )
        
        final_embeddings_data = []
        successful_embeddings_count = 0

        # Load existing embeddings first if they exist locally
        embeddings_file = self.embeddings_dir / "embeddings.json"
        if embeddings_file.exists():
            try:
                with open(embeddings_file, 'r', encoding='utf-8') as f:
                    final_embeddings_data = json.load(f)
                logger.info(f"Loaded {len(final_embeddings_data)} existing embeddings from local file.")
            except Exception as e:
                logger.warning(f"Failed to load existing embeddings file: {e}. Starting fresh.")
                final_embeddings_data = []
        
        # Keep track of IDs already in final_embeddings_data to avoid duplicates
        existing_ids_in_final_data = {item['id'] for item in final_embeddings_data}

        # Process generated embeddings
        for i, result in enumerate(generated_embeddings_results):
            # Get the original chunk info using the map
            original_chunk_info_index = chunks_to_embed_map[i]
            chunk_data = all_chunks_info[original_chunk_info_index]
            
            if result.get('success', False) and result.get('embedding'):
                if chunk_data['id'] not in existing_ids_in_final_data:
                    embedding_entry = {
                        **chunk_data,
                        "embedding": result['embedding'],
                        "embedding_model": "text-embedding-004"
                    }
                    final_embeddings_data.append(embedding_entry)
                    successful_embeddings_count += 1
                    
                    # Store embedding in Supabase if chunk has Supabase ID
                    supabase_chunk_id = chunk_data.get('supabase_chunk_id')
                    if supabase_chunk_id:
                        try:
                            embedding_obj = Embedding(
                                chunk_id=supabase_chunk_id,
                                embedding=result['embedding'],
                                model_name="text-embedding-004"
                            )
                            self.vector_store.insert_embeddings([embedding_obj])
                            logger.debug(f"✅ Stored embedding in Supabase for chunk {chunk_data['id']}")
                        except Exception as supabase_error:
                            logger.error(f"❌ Failed to store embedding in Supabase for chunk {chunk_data['id']}: {supabase_error}")
            else:
                logger.warning(f"⚠️ Failed to generate embedding for chunk {chunk_data['id']}: {result.get('error', 'Unknown error')}")

        # Save all embeddings (including newly generated ones and previously existing ones) locally
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(final_embeddings_data, f, indent=2, ensure_ascii=False)

        logger.info(f"🧠 Generated {successful_embeddings_count} new embeddings. Total local embeddings: {len(final_embeddings_data)}")
        logger.info(f"💾 Local backup saved to: {embeddings_file}")
        
        try:
            supabase_stats = self.vector_store.get_database_stats()
            logger.info(f"📊 Supabase verification - Embeddings count in DB: {supabase_stats['embeddings']}")
        except Exception as e:
            logger.warning(f"⚠️ Could not verify Supabase storage for embeddings: {e}")
        
        return final_embeddings_data


    def _aggregate_content_for_prompt(self, embeddings_data: List[Dict], topic: str) -> str:
        """Step 4: Aggregate content optimally for single prompt"""
        logger.info(f"📋 Starting content aggregation for {len(embeddings_data)} embeddings")

        aggregated_content = self.content_aggregator.aggregate_balanced_content( # Changed from aggregate_for_single_prompt
            embeddings_data, topic, max_tokens=800000
        )

        validation = self.content_aggregator.validate_aggregated_content(aggregated_content)
        logger.info(f"📊 Content validation results:")
        logger.info(f" • Length adequate: {validation['length_adequate']}")
        logger.info(f" • Has exam content: {validation['has_exam_content']}")
        logger.info(f" • Has lecture content: {validation['has_lecture_content']}")
        logger.info(f" • Content sections: {validation['content_sections']}")
        logger.info(f" • Total characters: {validation['total_characters']}")
        logger.info(f" • Overall valid: {validation['overall_valid']}")

        if not validation['overall_valid']:
            logger.warning("⚠️ Content validation failed, attempting direct markdown loading (fallback).")
            # Fallback to load all markdown if aggregation fails significantly
            aggregated_content = self.exam_generator.load_all_converted_markdown()
            logger.info(f"Fallback content loaded: {len(aggregated_content)} characters.")
            self.content_aggregator.conversion_stats["fallback_used"] += 1 # Not exactly, but for tracking

        if not aggregated_content or len(aggregated_content.strip()) < 1000:
            logger.error("❌ Aggregated content is still insufficient after fallback.")
            raise ValueError("Aggregated content too short or empty for prompt generation.")

        return aggregated_content

    def generate_three_papers_comprehensive_with_retry(self, topic: str, content: str,
                                                    requirements: Dict[str, str]) -> Dict[str, Any]:
        """Generate papers with progressive content reduction on timeout"""
        
        max_attempts = 3
        # Start with full content, reduce if attempts fail
        current_content_base = content 
        
        for attempt in range(max_attempts):
            try:
                # Reduce content size for each attempt after the first one
                if attempt > 0:
                    reduction_factor = 0.8 # Reduce content by 20% each retry
                    current_content = current_content_base[:int(len(current_content_base) * (reduction_factor ** attempt))]
                    logger.info(f"🔄 Attempt {attempt + 1}: Reducing content. Using {len(current_content)} characters.")
                else:
                    current_content = current_content_base
                    logger.info(f"🔄 Attempt {attempt + 1}: Using full content ({len(current_content)} characters).")

                if len(current_content.strip()) < 500: # Minimum content length for generation
                    logger.warning(f"Content too short for attempt {attempt + 1}, breaking retry loop.")
                    break
                
                comprehensive_prompt = self.exam_generator._build_complete_comprehensive_academic_prompt(
                    topic, current_content, requirements
                )
                
                response = self.exam_generator.gemini_client.generate_content( # Use exam_generator's client
                    comprehensive_prompt,
                    temperature=0.1,
                    max_tokens=15000,
                    timeout=180  # Extended timeout
                )
                
                if response and len(response) > 500:
                    logger.info(f"✅ Success on attempt {attempt + 1}")
                    return self.exam_generator._parse_comprehensive_academic_response(response, topic)
                    
            except Exception as e:
                logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    break
        
        # Fallback only after all attempts fail
        logger.error("❌ All retry attempts failed, using fallback exam generation.")
        return self.exam_generator._create_fallback_three_papers(topic, content) # Use exam_generator's fallback

    def _generate_three_papers(self, topic: str, content: str, requirements_file: Optional[str]) -> Dict:
        """Step 5: Generate three papers using comprehensive academic prompt"""
        if not content or len(content.strip()) < 1000:
            logger.error(f"❌ Insufficient content for generation: {len(content)} characters (after aggregation/fallback).")
            return self._create_emergency_fallback_response(topic)

        try:
            requirements = self._load_requirements(requirements_file)
            
            logger.info("🎯 Using comprehensive academic assessment creator approach")
            
            exam_result = self.generate_three_papers_comprehensive_with_retry( # Use the retry method
                topic=topic,
                content=content,
                requirements=requirements
            )

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
        """Create emergency fallback when all content loading or generation fails"""
        logger.info("🚨 Creating emergency fallback response due to critical failure.")
        return self.exam_generator._create_fallback_three_papers(topic, "No content available for detailed generation.")

    def _load_requirements(self, requirements_file: Optional[str]) -> Dict:
        """Load requirements from file or use defaults"""
        if requirements_file and Path(requirements_file).exists():
            try:
                with open(requirements_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load requirements file: {e}")

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
        formats = ["pdf", "json"] # Changed to primarily save PDF and the full JSON

        for paper_type, content in papers.items():
            if not content.strip():
                logger.warning(f"⚠️ Empty content for {paper_type}, skipping saving of this specific file type.")
                continue
            
            # Save as PDF
            if "pdf" in formats:
                filename = f"comprehensive_{paper_type}_{timestamp}.pdf"
                filepath = self.papers_dir / filename
                try:
                    self._generate_pdf_file(content, filepath, paper_type, topic)
                    saved_files.append(str(filepath))
                    logger.info(f"💾 Saved PDF: {filename}")
                except Exception as e:
                    logger.error(f"❌ Failed to save {filename} as PDF: {e}")
                    continue

        # Save complete JSON separately for the entire exam_result
        if "json" in formats:
            json_filename = f"comprehensive_complete_exam_{timestamp}.json"
            json_filepath = self.papers_dir / json_filename
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(exam_result, f, indent=2, ensure_ascii=False)
            saved_files.append(str(json_filepath))
            logger.info(f"💾 Saved JSON: {json_filename}")

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
            
            html_content = self._format_content_as_html(content, paper_type, topic)
            
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
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        doc = SimpleDocTemplate(str(filepath), pagesize=A4,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)

        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue,
            alignment=TA_LEFT
        )
        
        content_style = ParagraphStyle(
            'CustomContent',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            leftIndent=0,
            alignment=TA_LEFT
        )
        
        table_style = TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('BOX', (0,0), (-1,-1), 1, colors.black)
        ])


        story = []
        
        story.append(Paragraph(f"{paper_type.replace('_', ' ').title()}", title_style))
        story.append(Paragraph(f"Subject: {topic}", content_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", content_style))
        story.append(Spacer(1, 20))

        # Split content by lines/paragraphs and apply styles
        lines = content.split('\n')
        in_table = False
        current_table_lines = []

        for line in lines:
            if line.strip().startswith('|') and '|' in line.strip(): # Potential table line
                if not in_table:
                    # New table starts
                    in_table = True
                    current_table_lines = []
                current_table_lines.append(line.strip())
            else:
                if in_table:
                    # Table ended, process it
                    if current_table_lines:
                        # Clean and convert table data
                        table_data = []
                        for tbl_line in current_table_lines:
                            # Split by | and strip whitespace from each cell
                            row_cells = [cell.strip() for cell in tbl_line.strip('|').split('|')]
                            if any(cell for cell in row_cells): # Only add if row has content
                                table_data.append(row_cells)
                        
                        # Find the separator line (e.g., |---|---|)
                        header_idx = -1
                        for idx, row in enumerate(table_data):
                            if all(re.match(r'^-+$', cell) for cell in row if cell):
                                header_idx = idx
                                break
                        
                        if header_idx != -1:
                            header_row = table_data[0]
                            data_rows = table_data[header_idx + 1:]
                            
                            # Ensure all data rows have same number of columns as header
                            processed_table_data = [header_row]
                            processed_table_data.append(['-'*len(col) for col in header_row]) # Recreate separator
                            for row in data_rows:
                                if len(row) < len(header_row):
                                    row.extend([''] * (len(header_row) - len(row))) # Pad missing cells
                                processed_table_data.append(row)

                            story.append(Table(processed_table_data, style=table_style))
                            story.append(Spacer(1, 12))
                        else:
                            # If no separator found or malformed table, treat as normal text
                            for tbl_line in current_table_lines:
                                story.append(Paragraph(tbl_line, content_style))
                            
                    in_table = False
                    current_table_lines = []
                
                # Handle non-table lines
                if line.startswith('# '):
                    story.append(Paragraph(line[2:], heading_style))
                elif line.startswith('## '):
                    story.append(Paragraph(line[3:], heading_style))
                elif line.startswith('### '):
                    story.append(Paragraph(line[4:], heading_style))
                elif line.strip().startswith('Q') and (':' in line or '.' in line):
                    story.append(Spacer(1, 12))
                    story.append(Paragraph(f"<b>{line}</b>", content_style))
                elif line.strip(): # Avoid adding empty paragraphs
                    story.append(Paragraph(line.strip(), content_style))
        
        # After loop, check if still in table (e.g., table at end of content)
        if in_table and current_table_lines:
            table_data = []
            for tbl_line in current_table_lines:
                row_cells = [cell.strip() for cell in tbl_line.strip('|').split('|')]
                if any(cell for cell in row_cells):
                    table_data.append(row_cells)
            
            header_idx = -1
            for idx, row in enumerate(table_data):
                if all(re.match(r'^-+$', cell) for cell in row if cell):
                    header_idx = idx
                    break
            
            if header_idx != -1:
                header_row = table_data[0]
                data_rows = table_data[header_idx + 1:]
                processed_table_data = [header_row]
                processed_table_data.append(['-'*len(col) for col in header_row])
                for row in data_rows:
                    if len(row) < len(header_row):
                        row.extend([''] * (len(header_row) - len(row)))
                    processed_table_data.append(row)
                story.append(Table(processed_table_data, style=table_style))
            else:
                for tbl_line in current_table_lines:
                    story.append(Paragraph(tbl_line, content_style))

        story.append(Spacer(1, 12))

        doc.build(story)
        logger.info(f"✅ PDF generated with ReportLab: {filepath.name}")


    def _format_content_as_html(self, content: str, paper_type: str, topic: str) -> str:
        """Format content as HTML for WeasyPrint"""
        # CSS styling for academic papers
        css_styles = """
        <style>
        body { font-family: 'Times New Roman', serif; margin: 1in; line-height: 1.6; font-size: 11pt;}
        h1 { color: #1e3a8a; text-align: center; font-size: 20pt; margin-bottom: 30px;}
        h2 { color: #1e3a8a; font-size: 16pt; margin-top: 20px; }
        h3 { color: #374151; font-size: 14pt; }
        p { margin-bottom: 6px; }
        .question { font-weight: bold; margin: 15px 0; display: block; }
        .marks { font-weight: bold; color: #dc2626; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; page-break-inside: auto; }
        tr { page-break-inside: avoid; page-break-after: auto; }
        th, td { border: 1px solid #000; padding: 8px; text-align: left; vertical-align: top; }
        th { background-color: #f3f4f6; font-weight: bold; }
        .header { text-align: center; margin-bottom: 30px; }
        </style>
        """

        # Convert content to HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{paper_type.replace('_', ' ').title()}</title>
            {css_styles}
        </head>
        <body>
            <div class="header">
                <h1>{paper_type.replace('_', ' ').title()}</h1>
                <p><strong>Subject:</strong> {topic}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="content">
                {self._convert_markdown_to_html(content)}
            </div>
        </body>
        </html>
        """
        
        return html_content

    def _convert_markdown_to_html(self, content: str) -> str:
        """Convert markdown content to HTML, specifically handling tables for HTML."""
        import re
        
        # Replace code blocks first to protect their content
        def replace_code_blocks(match):
            lang = match.group(1) if match.group(1) else ''
            code = match.group(2)
            return f'<pre><code class="language-{lang}">{code}</code></pre>'
        content = re.sub(r'```(?P<lang>\w+)?\n(?P<code>.*?)\n```', replace_code_blocks, content, flags=re.DOTALL)

        # Convert headers
        content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)
        content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
        content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
        
        # Convert bold text
        content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
        
        # Convert italic text
        content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)

        # Convert unordered lists
        content = re.sub(r'^\s*[\*\-\+]\s+(.+)$', r'<li>\1</li>', content, flags=re.MULTILINE)
        content = re.sub(r'(<li>.+?</li>)(?!\\s*<li>)', r'<ul>\1</ul>', content, flags=re.DOTALL) # Wrap in <ul>

        # Convert ordered lists
        content = re.sub(r'^\s*\d+\.\s+(.+)$', r'<li>\1</li>', content, flags=re.MULTILINE)
        content = re.sub(r'(<li>.+?</li>)(?!\\s*<li>)', r'<ol>\1</ol>', content, flags=re.DOTALL) # Wrap in <ol>

        # Handle horizontal rules
        content = re.sub(r'^-{3,}$', '<hr>', content, flags=re.MULTILINE)

        # Convert markdown tables to HTML tables
        content = self._convert_tables_to_html(content)
        
        # Convert line breaks to paragraphs, ensuring tables/lists/code are not affected
        # This is tricky with simple regex. Better to split lines.
        lines = content.split('\n')
        processed_lines = []
        in_html_block = False # To avoid processing inside already converted HTML blocks
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('<') and stripped_line.endswith('>'): # Simple check for HTML tags
                in_html_block = True
                processed_lines.append(line)
            elif in_html_block and (stripped_line.startswith('</') or not stripped_line): # End of HTML block
                if stripped_line.startswith('</'):
                    in_html_block = False
                processed_lines.append(line)
            elif not stripped_line:
                if processed_lines and not processed_lines[-1].strip() == '</p>':
                    processed_lines.append('</p><p>')
            else:
                processed_lines.append(stripped_line)
        
        content = ''.join(processed_lines)
        content = content.replace('</p><p><ul>', '<ul>').replace('</ul></p><p>', '</ul>') # Clean up p tags around lists
        content = content.replace('</p><p><ol>', '<ol>').replace('</ol></p><p>', '</ol>') # Clean up p tags around lists
        content = content.replace('</p><p><pre>', '<pre>').replace('</pre></p><p>', '</pre>') # Clean up p tags around code
        content = content.replace('</p><p><table>', '<table>').replace('</table></p><p>', '</table>') # Clean up p tags around tables

        # Wrap remaining text in paragraphs
        if not content.startswith('<p>'):
            content = f'<p>{content}'
        if not content.endswith('</p>'):
            content = f'{content}</p>'

        # Convert question numbers
        content = re.sub(r'<p>(Q\d+[\.:])', r'<p class="question">\1', content)
        
        # Convert marks indicators
        content = re.sub(r'\((\d+\s*marks?)\)', r'<span class="marks">(\1)</span>', content, flags=re.IGNORECASE)
        
        return content

    def _convert_tables_to_html(self, content: str) -> str:
        """Convert pipe-separated tables to HTML tables"""
        # This regex tries to capture multi-line tables. It's a bit complex because Markdown tables
        # don't have clear start/end delimiters beyond consecutive lines.
        # It looks for lines starting and ending with '|' or lines with '---' (separator).
        # This assumes table lines are NOT separated by blank lines.

        # Regex to find a block of text that looks like a markdown table
        # It captures a sequence of lines that start and end with pipes, or are separator lines.
        # This is still a heuristic and might fail on malformed tables.
        table_block_pattern = r'(^\|.*(?:\|(?:\n\|.*)*?\n\|?\s*-{3,}[-|\s]*\|?(?:\n\|.*)*?)$)'
        
        def convert_table_match(match):
            markdown_table_block = match.group(1).strip()
            lines = markdown_table_block.split('\n')
            
            html_table = '<table>\n'
            header_processed = False
            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue # Skip empty lines inside the block
                
                if re.match(r'^\|?\s*-{3,}[-|\s]*\|?$', line): # This is the separator line
                    continue # Skip separator line itself, already used for header

                cells = [cell.strip() for cell in line.strip('|').split('|')]
                # Filter out empty strings from split, e.g., '||a|' splits to ['', '', 'a', '']
                cells = [cell for cell in cells if cell] 

                if not header_processed:
                    html_table += '<thead>\n<tr>\n'
                    for cell_content in cells:
                        html_table += f'<th>{cell_content}</th>\n'
                    html_table += '</tr>\n</thead>\n<tbody>\n'
                    header_processed = True
                else:
                    html_table += '<tr>\n'
                    for cell_content in cells:
                        html_table += f'<td>{cell_content}</td>\n'
                    html_table += '</tr>\n'
            
            html_table += '</tbody>\n</table>\n'
            return html_table
        
        # Use re.sub to find and replace all table blocks
        return re.sub(table_block_pattern, convert_table_match, content, flags=re.MULTILINE | re.DOTALL)


    def _add_content_to_pdf_story(self, content: str, story: list, heading_style, content_style):
        """Add content to PDF story with proper formatting"""
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        import re

        # Re-define styles here if necessary to ensure they are available
        styles = getSampleStyleSheet()
        table_style = TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('BOX', (0,0), (-1,-1), 1, colors.black)
        ])
        
        lines = content.split('\n')
        in_table_block = False
        current_table_lines = []
        
        for line_idx, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Check if this line is part of a Markdown table pattern
            is_table_line = stripped_line.startswith('|') and '|' in stripped_line
            is_table_separator = re.match(r'^\|?\s*-{3,}[-|\s]*\|?$', stripped_line) is not None

            if is_table_line or is_table_separator:
                current_table_lines.append(stripped_line)
                if not in_table_block:
                    in_table_block = True
            else:
                if in_table_block: # End of a table block
                    if current_table_lines:
                        # Process the accumulated table lines
                        table_data = []
                        header_line_idx = -1
                        for i, tbl_line in enumerate(current_table_lines):
                            if re.match(r'^\|?\s*-{3,}[-|\s]*\|?$', tbl_line):
                                header_line_idx = i
                                continue
                            cells = [cell.strip() for cell in tbl_line.strip('|').split('|')]
                            cells = [cell for cell in cells if cell] # Filter out empty strings from split
                            if cells: # Only add if row has content
                                table_data.append(cells)
                        
                        if table_data and header_line_idx != -1:
                            # Reconstruct table data with proper header and data separation
                            header_row = table_data[0] # Assumes first non-separator line is header
                            processed_table_data = [header_row]
                            # Recreate separator row for ReportLab's TableStyle
                            processed_table_data.append(['-'*len(col) if col else '' for col in header_row]) 
                            
                            # Add data rows, padding if necessary
                            for row in table_data[1:]:
                                if len(row) < len(header_row):
                                    row.extend([''] * (len(header_row) - len(row)))
                                processed_table_data.append(row)

                            story.append(Table(processed_table_data, style=table_style))
                            story.append(Spacer(1, 6)) # Spacer after table
                        else: # Malformed table, treat as paragraphs
                            for tbl_line in current_table_lines:
                                story.append(Paragraph(tbl_line, content_style))
                    
                    in_table_block = False
                    current_table_lines = []

                # Handle non-table, non-empty lines
                if stripped_line:
                    if stripped_line.startswith('# '):
                        story.append(Paragraph(stripped_line[2:], heading_style))
                    elif stripped_line.startswith('## '):
                        story.append(Paragraph(stripped_line[3:], heading_style))
                    elif stripped_line.startswith('### '):
                        story.append(Paragraph(stripped_line[4:], heading_style))
                    elif stripped_line.startswith('Q') and (':' in stripped_line or '.' in stripped_line):
                        story.append(Spacer(1, 12)) # Space before questions
                        story.append(Paragraph(f"<b>{stripped_line}</b>", content_style))
                    elif stripped_line.startswith('- ') or stripped_line.startswith('* '): # Basic list detection
                        story.append(Paragraph(f"• {stripped_line[2:].strip()}", content_style))
                    else:
                        story.append(Paragraph(stripped_line, content_style))
                elif not stripped_line and story and isinstance(story[-1], Paragraph):
                    # Add a spacer for empty lines, mimicking paragraph breaks
                    story.append(Spacer(1, 6))
        
        # After loop, if still in a table block (e.g., table at EOF)
        if in_table_block and current_table_lines:
            table_data = []
            header_line_idx = -1
            for i, tbl_line in enumerate(current_table_lines):
                if re.match(r'^\|?\s*-{3,}[-|\s]*\|?$', tbl_line):
                    header_line_idx = i
                    continue
                cells = [cell.strip() for cell in tbl_line.strip('|').split('|')]
                cells = [cell for cell in cells if cell]
                if cells:
                    table_data.append(cells)
            
            if table_data and header_line_idx != -1:
                header_row = table_data[0]
                processed_table_data = [header_row]
                processed_table_data.append(['-'*len(col) if col else '' for col in header_row])
                for row in table_data[1:]:
                    if len(row) < len(header_row):
                        row.extend([''] * (len(header_row) - len(row)))
                    processed_table_data.append(row)
                story.append(Table(processed_table_data, style=table_style))
            else:
                for tbl_line in current_table_lines:
                    story.append(Paragraph(tbl_line, content_style))

# Export the main class
__all__ = ['SinglePromptWorkflow']