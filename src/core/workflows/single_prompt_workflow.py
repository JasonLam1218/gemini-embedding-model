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
from config.settings import EMBEDDING_DIMENSIONS

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
        
        all_chunks_info_for_local_file = [] # Final list for local embeddings.json output
        chunks_to_embed_text_list = [] # List of texts to send to the embedding model
        # Map: index in chunks_to_embed_text_list -> {'doc_id': int, 'chunk_index': int, 'base_info': dict}
        # This will allow us to link generated embeddings back to the original chunk data and its Supabase ID
        chunks_to_embed_map_details = [] 

        for doc in documents:
            logger.info(f"🔄 Processing document: {doc.source_file}")
            
            # 1. Create or retrieve document in Supabase
            existing_doc = self.vector_store.document_exists_by_source_file(doc.source_file)
            doc_id = None
            if existing_doc:
                doc_id = existing_doc['id']
                logger.info(f"📄 Document already exists in Supabase (ID: {doc_id}): {doc.source_file}")
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
                    logger.info(f"✅ Document created in Supabase (ID: {doc_id}): {doc.source_file}")
                except Exception as e:
                    logger.error(f"❌ Failed to insert document {doc.source_file} into Supabase: {e}. Skipping this document.")
                    continue # Skip this document if parent document insertion failed

            # 2. Process chunks for this document: identify existing, new, and those needing embeddings
            current_doc_raw_chunks = self.chunker.chunk_text(doc.content)
            
            # Get current chunks from Supabase for this doc_id to match and get IDs
            current_db_chunks_map = {c['chunk_index']: c['id'] for c in self.vector_store.get_chunks_by_document(doc_id)}
            
            chunks_to_insert_into_db = [] # Chunks that are truly new and need to be inserted into text_chunks table
            
            # Placeholder for chunks to map after their Supabase ID is known
            chunks_pending_embedding_generation_details = [] 

            for i, chunk_text in enumerate(current_doc_raw_chunks):
                local_chunk_id = f"{doc.paper_set}_{doc.paper_number}_{i}"
                chunk_base_info = {
                    "id": local_chunk_id, # Local unique ID
                    "chunk_text": chunk_text,
                    "chunk_index": i,
                    "source_file": doc.source_file,
                    "content_type": doc.content_type,
                    "paper_set": doc.paper_set,
                    "metadata": doc.metadata,
                }
                
                db_chunk_id = current_db_chunks_map.get(i) # Get Supabase ID if it exists

                if db_chunk_id:
                    # Chunk exists in DB, check if it has an embedding
                    if self.vector_store.embedding_exists_for_chunk(db_chunk_id):
                        logger.debug(f"Chunk {db_chunk_id} (index {i}) already exists and has embedding. Loading existing.")
                        # Retrieve existing embedding and add to our local file output list
                        existing_embedding_data = self.vector_store.get_embedding_by_chunk_id(db_chunk_id)
                        if existing_embedding_data and existing_embedding_data.get('embedding'):
                            all_chunks_info_for_local_file.append({
                                **chunk_base_info,
                                "supabase_chunk_id": db_chunk_id,
                                "embedding": existing_embedding_data['embedding'],
                                "embedding_model": existing_embedding_data.get('model_name', 'gemini-embedding-001')
                            })
                        else:
                            # It exists, but embedding is missing or invalid. Needs re-embedding.
                            logger.warning(f"Chunk {db_chunk_id} exists but its embedding is invalid. Re-embedding.")
                            chunks_to_embed_text_list.append(chunk_text)
                            chunks_to_embed_map_details.append({'local_output_idx': len(all_chunks_info_for_local_file), 'supabase_chunk_id': db_chunk_id, 'base_info': chunk_base_info})
                            all_chunks_info_for_local_file.append(None) # Placeholder for later fill
                    else:
                        # Chunk exists but needs embedding
                        logger.debug(f"Chunk {db_chunk_id} (index {i}) exists but needs embedding. Adding to embedding queue.")
                        chunks_to_embed_text_list.append(chunk_text)
                        chunks_to_embed_map_details.append({'local_output_idx': len(all_chunks_info_for_local_file), 'supabase_chunk_id': db_chunk_id, 'base_info': chunk_base_info})
                        all_chunks_info_for_local_file.append(None) # Placeholder for later fill
                else:
                    # This is a new chunk, add to the DB insertion list
                    logger.debug(f"Chunk (index {i}) is new. Adding to DB insert queue and embedding queue.")
                    chunks_to_insert_into_db.append(TextChunk(
                        document_id=doc_id,
                        chunk_text=chunk_text,
                        chunk_index=i,
                        chunk_size=len(chunk_text)
                    ))
                    # Add to embedding queue, its supabase_chunk_id will be known after DB insert
                    chunks_to_embed_text_list.append(chunk_text)
                    # We store chunk_base_info here, and will update supabase_chunk_id after insertion
                    chunks_to_embed_map_details.append({'local_output_idx': len(all_chunks_info_for_local_file), 'base_info': chunk_base_info})
                    all_chunks_info_for_local_file.append(None) # Placeholder for later fill

            # 3. Insert new chunks into Supabase if any
            if chunks_to_insert_into_db:
                try:
                    inserted_chunk_ids = self.vector_store.insert_text_chunks(chunks_to_insert_into_db)
                    logger.info(f"✅ Inserted {len(inserted_chunk_ids)} new chunks for document {doc.source_file}.")
                    
                    # Update the chunks_to_embed_map_details with actual supabase_chunk_ids for newly inserted chunks
                    # This requires careful mapping of `chunks_to_insert_into_db` to `chunks_to_embed_map_details`.
                    # Assuming chunks_to_insert_into_db maintains order of chunks_to_embed_text_list for NEW chunks
                    
                    # Refined mapping: Iterate through map details and if `supabase_chunk_id` is missing, find it from `inserted_chunk_ids`
                    # This relies on the original `chunks_to_embed_map_details` preserving the `chunk_index` in `base_info`
                    
                    # Re-fetch the updated chunks to get correct IDs if new ones were inserted
                    # This ensures we always have the latest, correct DB IDs
                    updated_db_chunks_map_after_insert = {c['chunk_index']: c['id'] for c in self.vector_store.get_chunks_by_document(doc_id)}
                    
                    for map_entry in chunks_to_embed_map_details:
                        if 'supabase_chunk_id' not in map_entry or map_entry['supabase_chunk_id'] is None:
                            # This entry corresponds to a newly inserted chunk, find its ID
                            chunk_idx = map_entry['base_info']['chunk_index']
                            actual_db_id = updated_db_chunks_map_after_insert.get(chunk_idx)
                            if actual_db_id:
                                map_entry['supabase_chunk_id'] = actual_db_id
                            else:
                                logger.error(f"❌ Critical: Could not find Supabase ID for newly inserted chunk at index {chunk_idx}. This will cause a foreign key error later. Marking as invalid.")
                                map_entry['supabase_chunk_id'] = None # Explicitly set to None if we can't find it

                except Exception as e:
                    logger.error(f"❌ Failed to insert new chunks for document {doc.source_file} into Supabase: {e}. Chunks for this document will not have valid Supabase IDs for embedding.")
                    # Mark all relevant chunks in `chunks_to_embed_map_details` as having `supabase_chunk_id = None`
                    for map_entry in chunks_to_embed_map_details:
                        if map_entry['base_info']['source_file'] == doc.source_file: # Only for current doc
                            map_entry['supabase_chunk_id'] = None

        logger.info(f"🧠 Generating embeddings for {len(chunks_to_embed_text_list)} new/unembedded chunks.")
        
        # 4. Generate embeddings for the identified chunks
        generated_embeddings_results = self.embedding_generator.process_chunks_batch(
            chunks_to_embed_text_list, batch_size=5
        )
        
        embeddings_to_insert_into_supabase = [] 
        successful_embedding_generations = 0

        # 5. Process generated embeddings, populate local output, and prepare for Supabase insert
        for i, result in enumerate(generated_embeddings_results):
            mapped_info = chunks_to_embed_map_details[i]
            chunk_base_info = mapped_info['base_info']
            local_output_idx = mapped_info['local_output_idx']
            db_chunk_id = mapped_info.get('supabase_chunk_id') # This should now be reliable

            if not db_chunk_id:
                logger.warning(f"⚠️ Skipping embedding processing for local chunk {chunk_base_info['id']}: No valid Supabase chunk ID found. (This should have been handled earlier).")
                # Ensure placeholder is filled, even if with None
                all_chunks_info_for_local_file[local_output_idx] = {
                    **chunk_base_info,
                    "supabase_chunk_id": None,
                    "embedding": None,
                    "embedding_model": "gemini-embedding-001"
                }
                continue # Skip to next result if no valid db_chunk_id

            # Check if embedding generation was successful and the embedding data is valid (length check is crucial)
            if result.get('success', False) and result.get('embedding') and len(result['embedding']) == EMBEDDING_DIMENSIONS: 
                successful_embedding_generations += 1
                embedding_vector = result['embedding']
                
                # Create the entry for local file output
                embedding_entry = {
                    **chunk_base_info,
                    "supabase_chunk_id": db_chunk_id, 
                    "embedding": embedding_vector,
                    "embedding_model": "gemini-embedding-001"
                }
                all_chunks_info_for_local_file[local_output_idx] = embedding_entry
                
                # Add to list for batch Supabase insert
                embedding_obj = Embedding(
                    chunk_id=db_chunk_id,
                    embedding=embedding_vector,
                    model_name="gemini-embedding-001"
                )
                embeddings_to_insert_into_supabase.append(embedding_obj)
            else:
                logger.warning(f"⚠️ Failed to generate valid embedding for chunk {chunk_base_info['id']} (Supabase ID: {db_chunk_id}, Text: '{chunk_base_info['chunk_text'][:50]}...'): {result.get('error', 'Embedding data missing, empty, or wrong dimensions.')}. Setting embedding to None for local file.")
                # Create a placeholder entry for local file with None embedding
                all_chunks_info_for_local_file[local_output_idx] = {
                    **chunk_base_info,
                    "supabase_chunk_id": db_chunk_id,
                    "embedding": None, # Explicitly set to None
                    "embedding_model": "gemini-embedding-001"
                }
        
        # Filter out any remaining None placeholders if logic somehow failed (should ideally not be needed now)
        all_chunks_info_for_local_file = [c for c in all_chunks_info_for_local_file if c is not None]

        # 6. Perform a single batch insert of valid embeddings into Supabase
        if embeddings_to_insert_into_supabase:
            try:
                inserted_ids = self.vector_store.insert_embeddings(embeddings_to_insert_into_supabase)
                logger.info(f"✅ Stored {len(inserted_ids)} embeddings in Supabase in batch.")
            except Exception as supabase_error:
                logger.error(f"❌ Failed to store batch of embeddings in Supabase: {supabase_error}")
        else:
            logger.info("No new embeddings to insert into Supabase.")

        # 7. Save all embeddings (including newly generated ones and previously existing ones) locally
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        embeddings_file = self.embeddings_dir / "embeddings.json"
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks_info_for_local_file, f, indent=2, ensure_ascii=False)

        logger.info(f"🧠 Generated {successful_embedding_generations} new valid embeddings. Total local embeddings: {len(all_chunks_info_for_local_file)}")
        logger.info(f"💾 Local backup saved to: {embeddings_file}")
        
        try:
            supabase_stats = self.vector_store.get_database_stats()
            logger.info(f"📊 Supabase verification - Embeddings count in DB: {supabase_stats['embeddings']}")
        except Exception as e:
            logger.warning(f"⚠️ Could not verify Supabase storage for embeddings: {e}")
        
        return all_chunks_info_for_local_file


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
            # REMOVED: self.content_aggregator.conversion_stats["fallback_used"] += 1 # This line caused the error

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
                    max_tokens=10000,
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
            "answer_requirements": "Provide detailed model answers with step-by-step solutions and explanations",
            "marking_requirements": "Create detailed marking schemes with clear criteria and mark allocation"
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