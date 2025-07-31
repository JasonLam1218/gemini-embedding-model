#!/usr/bin/env python3

"""
Main pipeline controller for embedding-based exam generation system.
Enhanced with Supabase integration, quota-aware generation, and duplicate checking.
NOW WITH COMPREHENSIVE FILE LOGGING TO data/output/logs/
"""

import sys
import os
import click
from pathlib import Path
import json
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
import time

# Load environment variables from .env file
load_dotenv()

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ===== INITIALIZE COMPREHENSIVE LOGGING SYSTEM FIRST =====
# Import and setup logging BEFORE any other project imports
from src.core.utils.logging_config import (
    initialize_logging,
    log_system_information, 
    log_pipeline_start, 
    log_pipeline_end,
    create_operation_context,
    get_log_stats
)

# Initialize complete logging system
logs_dir = initialize_logging()

# NOW import loguru logger (after configuration)
from loguru import logger

# Log pipeline startup
logger.info("🚀 Starting gemini-embedding-model pipeline with comprehensive file logging")
logger.info(f"📁 All logs will be saved to: {logs_dir}")

# Import project modules (after logging setup)
from config.settings import BATCH_SIZE
from src.core.text.text_loader import TextLoader
from src.core.text.chunker import TextChunker
from src.core.embedding.embedding_generator import EmbeddingGenerator
from src.core.generation.structure_generator import StructureGenerator
from src.core.storage.vector_store import VectorStore, Document, TextChunk, Embedding

@click.group()
def cli():
    """Embedding-Based Exam Generation Pipeline CLI with Comprehensive File Logging"""
    logger.info("🎯 CLI initialized - all operations will be logged to files")

@cli.command()
@click.option('--input-dir', default='data/output/converted_markdown', help='Input directory (converted markdown files)')
@click.option('--use-supabase', is_flag=True, default=True, help='Store data in Supabase')
@click.option('--force-reprocess', is_flag=True, default=False, help='Force reprocess existing files')
@create_operation_context("Text Processing Pipeline")
def process_texts(input_dir, use_supabase, force_reprocess):
    """Load and process text files with duplicate checking and Supabase storage"""
    
    # Log pipeline start with parameters
    log_pipeline_start("process_texts", {
        "input_dir": input_dir,
        "use_supabase": use_supabase,
        "force_reprocess": force_reprocess
    })
    
    start_time = time.time()
    
    try:
        # Ensure output directory exists
        output_dir = Path("data/output/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Output directory: {output_dir}")

        # Initialize components
        logger.info("🔧 Initializing text processing components")
        loader = TextLoader()
        chunker = TextChunker()
        vector_store = VectorStore() if use_supabase else None
        logger.info("✅ Components initialized successfully")

        # Load documents
        logger.info(f"📂 Loading documents from: {input_dir}")
        documents = loader.process_directory(Path(input_dir))
        logger.info(f"📄 Found {len(documents)} documents to process")

        if not documents:
            logger.error(f"❌ No documents found in {input_dir}")
            raise ValueError(f"No documents found in {input_dir}")

        # Process documents into chunks with duplicate checking
        all_chunks = []
        supabase_results = []
        processed_count = 0
        skipped_count = 0

        for doc_index, doc in enumerate(documents, 1):
            source_file = doc.source_file
            logger.info(f"🔍 Processing document {doc_index}/{len(documents)}: {source_file}")

            # Check if document already exists in database
            if use_supabase and vector_store and not force_reprocess:
                logger.info(f"🔍 Checking for existing document: {source_file}")
                existing_doc = vector_store.document_exists_by_source_file(source_file)
                if existing_doc:
                    logger.info(f"⏭️ Document already exists (ID: {existing_doc['id']}): {source_file}, skipping processing")
                    skipped_count += 1
                    
                    # Retrieve existing chunks for local consistency
                    logger.info(f"📥 Retrieving existing chunks for {source_file}")
                    existing_chunks = vector_store.get_chunks_by_source_file(source_file)
                    logger.info(f"📥 Retrieved {len(existing_chunks)} existing chunks")
                    
                    for chunk in existing_chunks:
                        chunk_data = {
                            "id": f"{doc.paper_set}_{doc.paper_number}_{chunk['chunk_index']}",
                            "chunk_text": chunk['chunk_text'],
                            "chunk_index": chunk['chunk_index'],
                            "chunk_size": chunk['chunk_size'],
                            "source_file": source_file,
                            "paper_set": doc.paper_set,
                            "paper_number": doc.paper_number,
                            "metadata": doc.metadata
                        }
                        all_chunks.append(chunk_data)

                    supabase_results.append({
                        'document_id': existing_doc['id'],
                        'chunk_ids': [chunk['id'] for chunk in existing_chunks],
                        'source_file': source_file,
                        'status': 'existing'
                    })
                    continue

            # Process new document
            logger.info(f"🔄 Processing NEW document: {source_file}")
            processed_count += 1

            # Create chunks
            logger.info(f"✂️ Chunking document content ({len(doc.content)} characters)")
            chunks = chunker.chunk_text(doc.content)
            logger.info(f"📝 Created {len(chunks)} chunks for {doc.source_file}")

            if not chunks:
                logger.warning(f"⚠️ No chunks created for {source_file}")
                continue

            # Store in Supabase if enabled
            if use_supabase and vector_store:
                try:
                    logger.info("💾 Storing document and chunks in Supabase database")
                    
                    # Create Document object for Supabase
                    supabase_doc = Document(
                        title=Path(doc.source_file).stem,
                        content=doc.content,
                        source_file=doc.source_file,
                        paper_set=doc.paper_set,
                        paper_number=doc.paper_number,
                        metadata=doc.metadata
                    )

                    # Insert document to database
                    doc_id = vector_store.insert_document(supabase_doc)
                    logger.info(f"✅ Document stored in Supabase with ID: {doc_id}")

                    # Create and insert chunks to database
                    chunk_objects = []
                    for i, chunk_text in enumerate(chunks):
                        chunk_objects.append(TextChunk(
                            document_id=doc_id,
                            chunk_text=chunk_text,
                            chunk_index=i,
                            chunk_size=len(chunk_text),
                            overlap_size=0,
                            metadata={"source_file": doc.source_file}
                        ))

                    chunk_ids = vector_store.insert_text_chunks(chunk_objects)
                    logger.info(f"✅ Stored {len(chunk_ids)} chunks in Supabase database")

                    supabase_results.append({
                        'document_id': doc_id,
                        'chunk_ids': chunk_ids,
                        'source_file': doc.source_file,
                        'status': 'new'
                    })

                except Exception as e:
                    logger.error(f"❌ Failed to store document in Supabase: {e}")
                    logger.warning("⚠️ Continuing with local processing only")

            # Create local chunks data for JSON storage (keeping existing functionality)
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    "id": f"{doc.paper_set}_{doc.paper_number}_{i}",
                    "chunk_text": chunk,
                    "chunk_index": i,
                    "chunk_size": len(chunk),
                    "source_file": doc.source_file,
                    "paper_set": doc.paper_set,
                    "paper_number": doc.paper_number,
                    "metadata": doc.metadata
                }
                all_chunks.append(chunk_data)

        # Save processed chunks locally (keep existing functionality)
        chunks_path = output_dir / "processed_chunks.json"
        logger.info(f"💾 Saving {len(all_chunks)} chunks to: {chunks_path}")
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Chunks saved successfully to {chunks_path}")

        # Save processed documents locally
        docs_path = output_dir / "processed_documents.json"
        logger.info(f"💾 Saving documents metadata to: {docs_path}")
        loader.save_processed_documents(docs_path)
        logger.info(f"✅ Documents metadata saved to {docs_path}")

        # Save Supabase results
        if use_supabase and supabase_results:
            supabase_path = output_dir / "supabase_results.json"
            logger.info(f"💾 Saving Supabase operation results to: {supabase_path}")
            with open(supabase_path, "w", encoding="utf-8") as f:
                json.dump(supabase_results, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Supabase results saved to {supabase_path}")

        # Calculate final statistics
        duration = time.time() - start_time
        results = {
            "processed_documents": processed_count,
            "skipped_documents": skipped_count,
            "total_documents": len(documents),
            "total_chunks": len(all_chunks),
            "duration_seconds": round(duration, 2),
            "chunks_per_second": round(len(all_chunks) / duration, 2) if duration > 0 else 0
        }

        # Log comprehensive completion statistics
        logger.info("✅ Text processing completed successfully")
        logger.info(f"📊 FINAL PROCESSING STATISTICS:")
        logger.info(f"  • Total documents found: {len(documents)}")
        logger.info(f"  • New documents processed: {processed_count}")
        logger.info(f"  • Existing documents skipped: {skipped_count}")
        logger.info(f"  • Total chunks created: {len(all_chunks)}")
        logger.info(f"  • Processing duration: {duration:.2f} seconds")
        logger.info(f"  • Processing speed: {results['chunks_per_second']:.2f} chunks/second")

        # Show Supabase statistics
        if use_supabase and vector_store:
            try:
                stats = vector_store.get_database_stats()
                logger.info(f"📊 Supabase Database Status: {stats['documents']} documents, {stats['text_chunks']} chunks")
            except Exception as e:
                logger.warning(f"⚠️ Could not retrieve Supabase stats: {e}")

        # Log pipeline completion
        log_pipeline_end("process_texts", success=True, duration=duration, results=results)

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        logger.error(f"❌ Text processing failed after {duration:.2f} seconds: {error_msg}")
        log_pipeline_end("process_texts", success=False, duration=duration, error=error_msg)
        raise

@cli.command()
@click.option('--batch-size', default=BATCH_SIZE, help='Batch size for embedding generation')
@click.option('--use-supabase', is_flag=True, default=True, help='Store embeddings in Supabase')
@click.option('--force-regenerate', is_flag=True, default=False, help='Force regenerate existing embeddings')
@create_operation_context("Embedding Generation Pipeline")
def generate_embeddings(batch_size, use_supabase, force_regenerate):
    """Generate embeddings for all processed chunks with comprehensive logging"""
    
    log_pipeline_start("generate_embeddings", {
        "batch_size": batch_size,
        "use_supabase": use_supabase,
        "force_regenerate": force_regenerate
    })
    
    start_time = time.time()
    
    try:
        # Initialize components
        logger.info("🔧 Initializing embedding generator and vector store")
        generator = EmbeddingGenerator()
        vector_store = VectorStore() if use_supabase else None
        logger.info("✅ Components initialized successfully")

        # Load chunks from local file
        chunks_path = Path("data/output/processed/processed_chunks.json")
        if not chunks_path.exists():
            error_msg = "No processed chunks found. Run 'process-texts' first."
            logger.error(f"❌ {error_msg}")
            logger.error(f"Expected file: {chunks_path}")
            raise FileNotFoundError(error_msg)

        logger.info(f"📥 Loading chunks from: {chunks_path}")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        logger.info(f"📝 Loaded {len(chunks)} chunks for embedding generation")

        if not chunks:
            error_msg = "No chunks found in processed file"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        # Check quota status
        logger.info("🔍 Checking Gemini API quota status")
        quota_status = generator.check_quota_status()
        if quota_status:
            remaining = quota_status.get('requests_remaining', 'Unknown')
            total = quota_status.get('daily_limit', 'Unknown')
            logger.info(f"📊 API Quota Status: {remaining}/{total} requests remaining")
        else:
            logger.warning("⚠️ Could not check API quota status")

        # Get chunks that need embeddings
        chunks_needing_embeddings = []
        if use_supabase and vector_store and not force_regenerate:
            logger.info("🔍 Checking for existing embeddings in Supabase database")
            
            for chunk_index, chunk in enumerate(chunks, 1):
                if chunk_index % 100 == 0:
                    logger.info(f"🔍 Checked {chunk_index}/{len(chunks)} chunks for existing embeddings")
                
                # Find corresponding database chunk
                existing_doc = vector_store.document_exists_by_source_file(chunk['source_file'])
                if existing_doc:
                    db_chunks = vector_store.get_chunks_by_document(existing_doc['id'])
                    matching_chunk = next((c for c in db_chunks 
                                         if c['chunk_index'] == chunk['chunk_index']), None)
                    
                    if matching_chunk and vector_store.embedding_exists_for_chunk(matching_chunk['id']):
                        logger.debug(f"⏭️ Embedding already exists for chunk {chunk['id']}")
                        continue
                
                chunks_needing_embeddings.append(chunk)
        else:
            chunks_needing_embeddings = chunks

        logger.info(f"📝 Processing {len(chunks_needing_embeddings)} chunks needing embeddings")
        logger.info(f"⏭️ Skipping {len(chunks) - len(chunks_needing_embeddings)} chunks with existing embeddings")

        if not chunks_needing_embeddings:
            logger.info("✅ All chunks already have embeddings - no work needed")
            log_pipeline_end("generate_embeddings", success=True, duration=time.time() - start_time, 
                            results={"new_embeddings": 0, "total_embeddings": len(chunks)})
            return

        # Generate embeddings with comprehensive progress logging
        embeddings_data = []
        supabase_embeddings = []
        new_embeddings_count = 0
        failed_embeddings_count = 0

        for i, chunk in enumerate(chunks_needing_embeddings):
            try:
                progress_percent = (i + 1) / len(chunks_needing_embeddings) * 100
                logger.info(f"🧠 Generating embedding {i+1}/{len(chunks_needing_embeddings)} ({progress_percent:.1f}%) for chunk {chunk['id']}")
                
                # Generate embedding
                embedding = generator.generate_single_embedding(chunk["chunk_text"])
                
                if embedding:
                    # Create embedding data for local storage
                    chunk_with_embedding = {
                        **chunk,
                        "embedding": embedding,
                        "embedding_model": "text-embedding-004"
                    }
                    embeddings_data.append(chunk_with_embedding)

                    # Store in Supabase if enabled
                    if use_supabase and vector_store:
                        try:
                            # Find matching chunk in database
                            existing_doc = vector_store.document_exists_by_source_file(chunk['source_file'])
                            if existing_doc:
                                db_chunks = vector_store.get_chunks_by_document(existing_doc['id'])
                                matching_chunk = next((c for c in db_chunks 
                                                     if c['chunk_index'] == chunk['chunk_index']), None)
                                
                                if matching_chunk:
                                    # Create and insert embedding
                                    embedding_obj = Embedding(
                                        chunk_id=matching_chunk['id'],
                                        embedding=np.array(embedding, dtype=np.float32)
                                    )
                                    
                                    embedding_ids = vector_store.insert_embeddings([embedding_obj])
                                    supabase_embeddings.extend(embedding_ids)
                                    logger.debug(f"✅ Stored embedding in Supabase for chunk {matching_chunk['id']}")
                                else:
                                    logger.warning(f"⚠️ Could not find matching chunk in Supabase for {chunk['id']}")
                            else:
                                logger.warning(f"⚠️ Could not find document in Supabase for {chunk['source_file']}")
                                
                        except Exception as supabase_error:
                            logger.error(f"❌ Failed to store embedding in Supabase: {supabase_error}")

                    new_embeddings_count += 1
                    logger.info(f"✅ Generated embedding {new_embeddings_count}/{len(chunks_needing_embeddings)} ({len(embedding)} dimensions)")
                    
                    # Progress checkpoint every 10 embeddings
                    if new_embeddings_count % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = new_embeddings_count / elapsed
                        remaining = len(chunks_needing_embeddings) - new_embeddings_count
                        eta = remaining / rate if rate > 0 else 0
                        logger.info(f"📊 Progress: {new_embeddings_count} embeddings generated in {elapsed:.1f}s (rate: {rate:.2f}/s, ETA: {eta:.1f}s)")
                        
                        # Check quota status periodically
                        if new_embeddings_count % 50 == 0:
                            current_quota = generator.check_quota_status()
                            if current_quota:
                                remaining_requests = current_quota.get('requests_remaining', 'Unknown')
                                logger.info(f"📊 Current quota: {remaining_requests} requests remaining")
                    
                else:
                    failed_embeddings_count += 1
                    logger.warning(f"⚠️ Failed to generate embedding for chunk {chunk['id']} (attempt {i+1})")

            except Exception as e:
                error_str = str(e).lower()
                if "quota" in error_str or "429" in error_str:
                    logger.error(f"❌ API quota exhausted at chunk {i+1}. Generated {new_embeddings_count} embeddings so far.")
                    logger.error("💡 Try again later when quota resets, or continue with existing embeddings")
                    break
                else:
                    failed_embeddings_count += 1
                    logger.error(f"❌ Failed to generate embedding for chunk {chunk['id']}: {e}")
                    continue

        # Load existing embeddings for complete dataset
        if not force_regenerate:
            existing_embeddings_path = Path("data/output/processed/embeddings.json")
            if existing_embeddings_path.exists():
                logger.info("📥 Loading existing embeddings to merge with new ones")
                with open(existing_embeddings_path, "r", encoding="utf-8") as f:
                    existing_embeddings = json.load(f)
                
                # Merge with new embeddings (avoid duplicates)
                existing_ids = {emb['id'] for emb in existing_embeddings}
                merged_count = 0
                for new_emb in embeddings_data:
                    if new_emb['id'] not in existing_ids:
                        existing_embeddings.append(new_emb)
                        merged_count += 1
                embeddings_data = existing_embeddings
                logger.info(f"🔗 Merged {merged_count} new embeddings with {len(existing_embeddings) - merged_count} existing: {len(embeddings_data)} total")

        # Save embeddings locally
        output_path = Path("data/output/processed/embeddings.json")
        logger.info(f"💾 Saving {len(embeddings_data)} embeddings to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Embeddings saved successfully to {output_path}")

        # Calculate final statistics
        duration = time.time() - start_time
        results = {
            "new_embeddings_generated": new_embeddings_count,
            "failed_embeddings": failed_embeddings_count,
            "total_embeddings": len(embeddings_data),
            "supabase_embeddings_stored": len(supabase_embeddings),
            "duration_seconds": round(duration, 2),
            "embeddings_per_second": round(new_embeddings_count / duration, 2) if duration > 0 else 0
        }

        # Log comprehensive completion statistics
        logger.info("✅ Embedding generation completed successfully")
        logger.info(f"📊 FINAL EMBEDDING STATISTICS:")
        logger.info(f"  • New embeddings generated: {new_embeddings_count}")
        logger.info(f"  • Failed embedding attempts: {failed_embeddings_count}")
        logger.info(f"  • Total embeddings in dataset: {len(embeddings_data)}")
        logger.info(f"  • Supabase embeddings stored: {len(supabase_embeddings)}")
        logger.info(f"  • Processing duration: {duration:.2f} seconds")
        logger.info(f"  • Generation rate: {results['embeddings_per_second']:.2f} embeddings/second")

        # Show Supabase statistics
        if use_supabase and vector_store:
            try:
                stats = vector_store.get_database_stats()
                logger.info(f"📊 Supabase Database Status: {stats['embeddings']} total embeddings stored")
            except Exception as e:
                logger.warning(f"⚠️ Could not retrieve Supabase stats: {e}")

        # Warning if not all embeddings were generated
        if new_embeddings_count < len(chunks_needing_embeddings):
            missing = len(chunks_needing_embeddings) - new_embeddings_count
            logger.warning(f"⚠️ Only {new_embeddings_count}/{len(chunks_needing_embeddings)} new embeddings generated ({missing} missing)")
            logger.warning("💡 This may be due to API quota limits or network issues")

        log_pipeline_end("generate_embeddings", success=True, duration=duration, results=results)

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        logger.error(f"❌ Embedding generation failed after {duration:.2f} seconds: {error_msg}")
        log_pipeline_end("generate_embeddings", success=False, duration=duration, error=error_msg)
        raise

@cli.command()
@click.option('--topic', default='AI and Data Analytics', help='Exam topic')
@click.option('--structure-type', default='standard', help='Exam structure type')
@click.option('--formats', default='txt,md,pdf,json', help='Output formats (comma-separated)')
@click.option('--quota-aware', is_flag=True, default=True, help='Use quota-aware generation')
@click.option('--template-only', is_flag=True, default=False, help='Use template-only generation (no API calls)')
@click.option('--use-supabase', is_flag=True, default=True, help='Use Supabase for content retrieval')
@create_operation_context("Structured Exam Generation Pipeline")
def generate_structured_exam(topic, structure_type, formats, quota_aware, template_only, use_supabase):
    """Generate structured exam paper with model answers and marking schemes"""
    
    log_pipeline_start("generate_structured_exam", {
        "topic": topic,
        "structure_type": structure_type,
        "formats": formats,
        "quota_aware": quota_aware,
        "template_only": template_only,
        "use_supabase": use_supabase
    })
    
    start_time = time.time()
    
    try:
        logger.info(f"🔄 Generating structured exam paper for topic: {topic}")

        # Store the original Supabase intent
        original_use_supabase = use_supabase

        # Initialize components
        logger.info("🔧 Initializing structure generator and vector store")
        structure_gen = StructureGenerator()
        vector_store = VectorStore() if use_supabase else None

        # Log generation mode
        if template_only:
            logger.info("📝 Using template-only generation (no API calls)")
        elif quota_aware:
            logger.info("🛡️ Using quota-aware generation to prevent API exhaustion")
        else:
            logger.info("🚀 Using full AI generation mode")

        # Parse formats
        format_list = [f.strip().lower() for f in formats.split(',')]
        logger.info(f"📄 Output formats: {', '.join(format_list)}")

        # Check if we can use Supabase for CONTENT RETRIEVAL
        exam_paper = None
        use_supabase_for_retrieval = use_supabase

        if use_supabase and vector_store:
            try:
                logger.info("🔍 Testing Supabase connection and checking data availability")
                # Test Supabase connection and check data availability
                stats = vector_store.get_database_stats()
                if stats['documents'] > 0 and stats['embeddings'] > 0:
                    logger.info(f"✅ Supabase available: {stats['documents']} documents, {stats['embeddings']} embeddings")
                    
                    # Generate exam using Supabase data (if method exists)
                    if hasattr(structure_gen, 'generate_structured_exam_from_supabase'):
                        logger.info("🔄 Generating exam using Supabase data retrieval")
                        exam_paper = structure_gen.generate_structured_exam_from_supabase(
                            topic=topic,
                            structure_type=structure_type,
                            vector_store=vector_store
                        )
                        logger.info("✅ Exam generated using Supabase data")
                    else:
                        logger.warning("⚠️ Supabase integration method not found in StructureGenerator")
                        use_supabase_for_retrieval = False
                else:
                    logger.warning(f"⚠️ Supabase has insufficient data (docs: {stats['documents']}, embeddings: {stats['embeddings']})")
                    logger.warning("Falling back to local file generation")
                    use_supabase_for_retrieval = False

            except Exception as e:
                logger.error(f"❌ Supabase connection/data check failed: {e}")
                logger.warning("⚠️ Falling back to local file generation")
                use_supabase_for_retrieval = False

        # Fall back to local generation if Supabase retrieval not available or failed
        if not exam_paper:
            # Check if embeddings exist locally
            embeddings_path = Path("data/output/processed/embeddings.json")
            if not embeddings_path.exists():
                error_msg = "No embeddings found locally and Supabase unavailable. Run 'generate-embeddings' first."
                logger.error(f"❌ {error_msg}")
                raise FileNotFoundError(error_msg)

            logger.info(f"📁 Using local embeddings file: {embeddings_path}")

            # Generate exam based on mode
            if template_only:
                logger.info("📝 Generating template-only exam (no API calls)")
                exam_paper = structure_gen.generate_template_only_exam(topic=topic)
                logger.info("✅ Template-only exam generated successfully")
            else:
                logger.info("🧠 Generating AI-enhanced structured exam")
                exam_paper = structure_gen.generate_structured_exam(
                    topic=topic,
                    structure_type=structure_type
                )
                logger.info("✅ AI-enhanced exam generated successfully")

        # Check if generation was successful
        if not exam_paper:
            error_msg = "Exam generation returned no data"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        total_questions = exam_paper.get('exam_metadata', {}).get('total_questions', 0) or len(exam_paper.get('questions', {}))
        if total_questions == 0:
            logger.error("❌ No questions were generated. Check your content and API quota.")
            if not template_only:
                logger.info("💡 Try using --template-only flag for fallback generation")
            raise ValueError("No questions generated")

        logger.info(f"✅ Exam generated with {total_questions} questions")

        # === SUPABASE SAVE INTEGRATION ===
        if original_use_supabase and vector_store and exam_paper:
            try:
                logger.info("💾 Saving generated exam to Supabase database")
                
                # Ensure exam_paper has required metadata structure
                if not exam_paper.get('exam_metadata'):
                    logger.info("📋 Adding missing exam metadata")
                    exam_paper['exam_metadata'] = {
                        'title': f"Generated Exam - {topic}",
                        'topic': topic,
                        'difficulty': 'standard',
                        'total_marks': 100,
                        'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                if not exam_paper.get('generation_stats'):
                    logger.info("📊 Adding missing generation statistics")
                    questions = exam_paper.get('questions', {})
                    exam_paper['generation_stats'] = {
                        'questions_generated': len(questions),
                        'total_marks': 100,
                        'content_sources_used': 4,
                        'generation_mode': 'template' if template_only else 'ai_enhanced'
                    }

                # Validate and enhance metadata for Supabase compatibility
                exam_metadata = exam_paper.get('exam_metadata', {})
                if not exam_metadata.get('title'):
                    exam_metadata['title'] = f"Generated Exam - {topic}"
                if not exam_metadata.get('topic'):
                    exam_metadata['topic'] = topic
                if not exam_metadata.get('difficulty'):
                    exam_metadata['difficulty'] = 'standard'
                if not exam_metadata.get('total_marks'):
                    exam_metadata['total_marks'] = 100

                # Save to Supabase
                exam_id = vector_store.save_generated_exam(exam_paper)
                logger.info(f"✅ Exam saved to Supabase database (ID: {exam_id})")

                # Verify the save was successful
                if hasattr(vector_store, 'verify_exam_saved'):
                    verification_success = vector_store.verify_exam_saved(exam_id)
                    if verification_success:
                        logger.info(f"✅ Verified exam save successful (ID: {exam_id})")
                    else:
                        logger.warning(f"⚠️ Could not verify exam save (ID: {exam_id})")

                # Update generation stats to include database save
                if 'generation_stats' in exam_paper:
                    exam_paper['generation_stats']['saved_to_database'] = True
                    exam_paper['generation_stats']['database_id'] = exam_id

            except Exception as e:
                logger.error(f"❌ Failed to save exam to Supabase: {e}")
                logger.warning("⚠️ Exam generated locally but not saved to database")
                logger.info("🔍 Check your Supabase connection and database schema")

        # Save generated exam in multiple formats locally
        output_dir = Path("data/output/generated_exams")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Saving exam files to: {output_dir}")

        logger.info("💾 Saving exam in multiple formats")
        saved_files = structure_gen.save_multi_format_exam(
            exam_paper,
            output_dir,
            formats=format_list
        )

        # Log saved files
        logger.info(f"✅ Generated structured exam paper with {'templates' if template_only else 'AI components'}")
        logger.info("📝 Generated files:")
        for file_path in saved_files:
            file_type = "Question Paper" if "questions" in file_path else \
                       "Model Answers" if "answers" in file_path else \
                       "Marking Schemes" if "schemes" in file_path else "Complete Exam"
            logger.info(f"   {file_type}: {file_path}")

        # Calculate final statistics
        duration = time.time() - start_time
        stats = exam_paper.get('generation_stats', {})
        
        results = {
            "questions_generated": stats.get('questions_generated', 0),
            "total_marks": stats.get('total_marks', 0),
            "content_sources_used": stats.get('content_sources_used', 0),
            "files_generated": len(saved_files),
            "generation_mode": "template" if template_only else "ai_enhanced",
            "saved_to_database": stats.get('saved_to_database', False),
            "duration_seconds": round(duration, 2)
        }

        # Display comprehensive generation statistics
        logger.info(f"📊 EXAM GENERATION STATISTICS:")
        logger.info(f"  • Questions Generated: {results['questions_generated']}")
        logger.info(f"  • Total Marks: {results['total_marks']}")
        logger.info(f"  • Content Sources Used: {results['content_sources_used']}")
        logger.info(f"  • Files Generated: {results['files_generated']} ({', '.join(format_list)})")
        logger.info(f"  • Generation Mode: {results['generation_mode']}")
        logger.info(f"  • Duration: {results['duration_seconds']} seconds")

        # Display database save status
        if results['saved_to_database']:
            database_id = stats.get('database_id', 'Unknown')
            logger.info(f"  • Database Save: ✅ Saved (ID: {database_id})")
        else:
            logger.info(f"  • Database Save: ❌ Not saved (check Supabase connection)")

        log_pipeline_end("generate_structured_exam", success=True, duration=duration, results=results)

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        logger.error(f"❌ Structured exam generation failed after {duration:.2f} seconds: {error_msg}")
        
        if "quota" in error_msg.lower() or "429" in error_msg:
            logger.info("💡 Try using --template-only flag to generate exams without API calls")
        
        log_pipeline_end("generate_structured_exam", success=False, duration=duration, error=error_msg)
        raise

@cli.command('run-full-pipeline')  # Explicitly specify the command name
@create_operation_context("Full Pipeline Execution")
def run_full_pipeline():
    """Run the complete embedding-based exam generation pipeline with comprehensive logging"""
    
    log_pipeline_start("run_full_pipeline", {
        "mode": "complete_pipeline",
        "includes": ["text_processing", "embedding_generation", "exam_generation"]
    })
    
    start_time = time.time()
    
    try:
        logger.info("🚀 Starting COMPLETE embedding-based exam generation pipeline")
        logger.info("This will run: Text Processing → Embedding Generation → Exam Generation")
        
        ctx = click.get_current_context()

        # Step 1: Text Processing
        logger.info("📝 STEP 1/3: Processing text inputs with duplicate checking")
        try:
            ctx.invoke(process_texts, use_supabase=True, force_reprocess=False)
            logger.info("✅ Step 1 completed: Text processing successful")
        except Exception as e:
            logger.error(f"❌ Step 1 failed: Text processing error: {e}")
            raise

        # Step 2: Embedding Generation
        logger.info("🧠 STEP 2/3: Generating embeddings with duplicate checking")
        try:
            ctx.invoke(generate_embeddings, use_supabase=True, force_regenerate=False)
            logger.info("✅ Step 2 completed: Embedding generation successful")
        except Exception as e:
            logger.warning(f"⚠️ Step 2 had issues: Embedding generation error: {e}")
            
            # Check if we have any embeddings to continue
            embeddings_path = Path("data/output/processed/embeddings.json")
            if not embeddings_path.exists():
                logger.error("❌ No embeddings available. Cannot continue to exam generation.")
                raise
            else:
                logger.info("📄 Found existing embeddings file - continuing with available data")

        # Step 3: Exam Generation
        logger.info("📋 STEP 3/3: Generating structured exam paper")
        try:
            # Try AI-enhanced generation first
            logger.info("🧠 Attempting AI-enhanced exam generation")
            ctx.invoke(generate_structured_exam, 
                      formats='txt,md,pdf,json', 
                      quota_aware=True, 
                      use_supabase=True)
            logger.info("✅ Step 3 completed: AI-enhanced exam generation successful")
            
        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "429" in error_str:
                logger.warning("⚠️ API quota exhausted during exam generation")
                logger.info("💡 Falling back to template-only generation")
                try:
                    ctx.invoke(generate_structured_exam, 
                              formats='txt,md,pdf,json', 
                              template_only=True, 
                              use_supabase=True)
                    logger.info("✅ Step 3 completed: Template-based exam generation successful")
                except Exception as fallback_error:
                    logger.error(f"❌ Step 3 failed: Template fallback also failed: {fallback_error}")
                    raise
            else:
                logger.error(f"❌ Step 3 failed: Exam generation error: {e}")
                raise

        # Calculate final statistics
        duration = time.time() - start_time
        
        # Gather comprehensive statistics
        try:
            chunks_file = Path("data/output/processed/processed_chunks.json")
            embeddings_file = Path("data/output/processed/embeddings.json")
            exams_dir = Path("data/output/generated_exams")
            
            chunks_count = 0
            embeddings_count = 0
            exam_files_count = 0
            
            if chunks_file.exists():
                with open(chunks_file) as f:
                    chunks_count = len(json.load(f))
            
            if embeddings_file.exists():
                with open(embeddings_file) as f:
                    embeddings_count = len(json.load(f))
            
            if exams_dir.exists():
                exam_files_count = len(list(exams_dir.glob("*")))
            
            results = {
                "total_duration": round(duration, 2),
                "chunks_processed": chunks_count,
                "embeddings_generated": embeddings_count,
                "exam_files_created": exam_files_count,
                "pipeline_steps_completed": 3,
                "success": True
            }
            
        except Exception as stats_error:
            logger.warning(f"⚠️ Could not gather final statistics: {stats_error}")
            results = {
                "total_duration": round(duration, 2),
                "pipeline_steps_completed": 3,
                "success": True
            }

        # Log comprehensive completion
        logger.info("✅ COMPLETE PIPELINE EXECUTION SUCCESSFUL!")
        logger.info("🎉 All steps completed successfully")
        logger.info(f"📊 FINAL PIPELINE STATISTICS:")
        for key, value in results.items():
            logger.info(f"  • {key.replace('_', ' ').title()}: {value}")

        # Show final file locations
        logger.info("📁 GENERATED FILES LOCATION:")
        logger.info(f"  • Processed chunks: data/output/processed/processed_chunks.json")
        logger.info(f"  • Embeddings: data/output/processed/embeddings.json")
        logger.info(f"  • Exam files: data/output/generated_exams/")
        logger.info(f"  • Log files: data/output/logs/")

        log_pipeline_end("run_full_pipeline", success=True, duration=duration, results=results)

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        logger.error(f"❌ Full pipeline execution failed after {duration:.2f} seconds: {error_msg}")
        log_pipeline_end("run_full_pipeline", success=False, duration=duration, error=error_msg)
        raise

# === LOGGING MANAGEMENT COMMANDS ===

@cli.command()
@click.option('--tail', is_flag=True, help='Follow log output in real-time')
@click.option('--lines', default=50, help='Number of recent lines to show')
@click.option('--log-type', default='application', 
              type=click.Choice(['application', 'errors', 'api', 'pipeline', 'database', 'text', 'exam', 'performance']),
              help='Type of log to view')
def logs(tail, lines, log_type):
    """View pipeline logs from data/output/logs directory"""
    logger.info(f"📖 Viewing {log_type} logs")
    
    logs_dir = Path("data/output/logs")
    
    if not logs_dir.exists():
        click.echo("❌ No logs directory found at data/output/logs")
        click.echo("💡 Run any pipeline command first to create logs")
        return
    
    # Map log types to file patterns
    log_files_map = {
        'application': 'pipeline_application.log',
        'errors': 'pipeline_errors.log',
        'api': 'api_operations.log',
        'pipeline': 'pipeline_operations.log',
        'database': 'database_operations.log',
        'text': 'text_processing.log',
        'exam': 'exam_generation.log',
        'performance': 'performance.log'
    }
    
    log_file_pattern = log_files_map.get(log_type, 'pipeline_application.log')
    log_files = list(logs_dir.glob(log_file_pattern))
    
    # Also check for session logs if application type
    if log_type == 'application':
        session_logs = list(logs_dir.glob('session_*.log'))
        log_files.extend(session_logs)
    
    if not log_files:
        click.echo(f"❌ No {log_type} log files found")
        click.echo(f"Available log files:")
        for f in logs_dir.glob('*.log'):
            click.echo(f"  - {f.name}")
        return
    
    # Get most recent log file
    recent_log = max(log_files, key=lambda x: x.stat().st_mtime)
    
    if tail:
        click.echo(f"📝 Following log: {recent_log.name}")
        click.echo("Press Ctrl+C to stop")
        try:
            import subprocess
            subprocess.run(["tail", "-f", str(recent_log)])
        except KeyboardInterrupt:
            click.echo("\n👋 Stopped following log")
        except FileNotFoundError:
            # Fallback for systems without tail command
            click.echo("⚠️ 'tail' command not available. Showing recent lines instead.")
            with open(recent_log, 'r', encoding='utf-8') as f:
                lines_list = f.readlines()
                for line in lines_list[-lines:]:
                    click.echo(line.rstrip())
    else:
        click.echo(f"📝 Last {lines} lines from: {recent_log.name}")
        try:
            with open(recent_log, 'r', encoding='utf-8') as f:
                lines_list = f.readlines()
                for line in lines_list[-lines:]:
                    click.echo(line.rstrip())
        except Exception as e:
            click.echo(f"❌ Error reading log file: {e}")

@cli.command()
def log_status():
    """Show comprehensive logging status and log file information"""
    logger.info("📊 Generating logging status report")
    
    logs_dir = Path("data/output/logs")
    
    click.echo("📝 COMPREHENSIVE LOGGING STATUS REPORT")
    click.echo("=" * 60)
    
    if not logs_dir.exists():
        click.echo("❌ Logs directory does not exist at data/output/logs")
        click.echo("💡 Run any pipeline command to create the logs directory")
        return
    
    click.echo(f"📁 Logs Directory: {logs_dir.absolute()}")
    
    # List all log files with comprehensive details
    log_files = list(logs_dir.glob('*.log*'))
    
    if not log_files:
        click.echo("❌ No log files found")
        click.echo("💡 Run pipeline commands to generate logs")
        return
    
    click.echo(f"\n📋 Log Files Report ({len(log_files)} files):")
    
    # Group files by type
    log_types = {
        'Application': [f for f in log_files if 'application' in f.name],
        'Errors': [f for f in log_files if 'errors' in f.name],
        'API Operations': [f for f in log_files if 'api_operations' in f.name],
        'Pipeline Operations': [f for f in log_files if 'pipeline_operations' in f.name],
        'Database': [f for f in log_files if 'database' in f.name],
        'Text Processing': [f for f in log_files if 'text_processing' in f.name],
        'Exam Generation': [f for f in log_files if 'exam_generation' in f.name],
        'Performance': [f for f in log_files if 'performance' in f.name],
        'Sessions': [f for f in log_files if 'session_' in f.name],
        'Archives': [f for f in log_files if f.suffix == '.zip']
    }
    
    total_size = 0
    for log_type, files in log_types.items():
        if files:
            click.echo(f"\n  📂 {log_type}:")
            for log_file in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
                size = log_file.stat().st_size
                total_size += size
                
                # Format size
                if size > 1024 * 1024:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size} B"
                
                # Format timestamp
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                time_str = mtime.strftime("%Y-%m-%d %H:%M:%S")
                
                click.echo(f"    📄 {log_file.name:<35} {size_str:>8} {time_str}")
    
    # Total size
    if total_size > 1024 * 1024:
        total_size_str = f"{total_size / (1024 * 1024):.1f} MB"
    elif total_size > 1024:
        total_size_str = f"{total_size / 1024:.1f} KB"
    else:
        total_size_str = f"{total_size} B"
    
    click.echo(f"\n📊 Total Log Size: {total_size_str}")
    
    # Show recent activity
    if log_files:
        newest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        newest_time = datetime.fromtimestamp(newest_log.stat().st_mtime)
        click.echo(f"🕒 Most Recent Activity: {newest_time.strftime('%Y-%m-%d %H:%M:%S')} ({newest_log.name})")
    
    # Show log statistics
    try:
        log_stats = get_log_stats()
        if log_stats:
            click.echo("\n📈 Log File Statistics:")
            active_logs = [name for name, info in log_stats.items() if not name.endswith('.zip')]
            archived_logs = [name for name, info in log_stats.items() if name.endswith('.zip')]
            
            click.echo(f"  • Active log files: {len(active_logs)}")
            click.echo(f"  • Archived log files: {len(archived_logs)}")
            
            if active_logs:
                newest_active = max(active_logs, key=lambda x: log_stats[x]['modified'])
                click.echo(f"  • Newest active log: {newest_active}")
    except Exception as e:
        click.echo(f"⚠️ Could not generate log statistics: {e}")

@cli.command()
def clear_logs():
    """Clear all log files (WARNING: This will delete all logs)"""
    logs_dir = Path("data/output/logs")
    
    if not logs_dir.exists():
        click.echo("❌ No logs directory found")
        return
    
    log_files = list(logs_dir.glob('*.log*'))
    
    if not log_files:
        click.echo("❌ No log files found to clear")
        return
    
    # Confirm before deletion
    click.echo(f"⚠️  WARNING: This will delete {len(log_files)} log files")
    if click.confirm("Are you sure you want to clear all logs?"):
        try:
            deleted_count = 0
            for log_file in log_files:
                log_file.unlink()
                deleted_count += 1
            
            click.echo(f"✅ Deleted {deleted_count} log files")
            logger.info(f"🧹 Log files cleared by user command - {deleted_count} files deleted")
            
        except Exception as e:
            click.echo(f"❌ Error clearing logs: {e}")
            logger.error(f"❌ Failed to clear logs: {e}")
    else:
        click.echo("❌ Log clearing cancelled")

# Keep all your existing commands unchanged...
# (status, test_database, check_duplicates, test_supabase remain exactly the same)

@cli.command()
def status():
    """Show pipeline status including Supabase information and duplicate checking status"""
    click.echo("📊 Embedding-Based Pipeline Status Report with Duplicate Checking")
    click.echo("=" * 60)

    # Check local files
    chunks_file = Path("data/output/processed/processed_chunks.json")
    if chunks_file.exists():
        with open(chunks_file) as f:
            chunks = json.load(f)
        click.echo(f"📄 Processed chunks (local): {len(chunks)}")
    else:
        click.echo("📄 Processed chunks (local): Not found")

    embeddings_file = Path("data/output/processed/embeddings.json")
    if embeddings_file.exists():
        with open(embeddings_file) as f:
            embeddings = json.load(f)
        click.echo(f"🧠 Generated embeddings (local): {len(embeddings)}")
        if embeddings:
            click.echo(f"🧠 Embedding dimensions: {len(embeddings[0].get('embedding', []))}")
            click.echo(f"🧠 Embedding model: {embeddings[0].get('embedding_model', 'Unknown')}")
    else:
        click.echo("🧠 Generated embeddings (local): Not found")

    # Check Supabase status
    try:
        vector_store = VectorStore()
        stats = vector_store.get_database_stats()
        click.echo(f"🗄️ Supabase Documents: {stats['documents']}")
        click.echo(f"🗄️ Supabase Text Chunks: {stats['text_chunks']}")
        click.echo(f"🗄️ Supabase Embeddings: {stats['embeddings']}")
        click.echo(f"🗄️ Supabase Generated Exams: {stats['generated_exams']}")

        # Check for chunks without embeddings
        chunks_without_embeddings = vector_store.get_chunks_without_embeddings()
        click.echo(f"⚠️ Chunks missing embeddings: {len(chunks_without_embeddings)}")

        # Get recent exams from Supabase
        recent_exams = vector_store.get_generated_exams(limit=5)
        if recent_exams:
            click.echo("📋 Recent exam files:")
            for exam in recent_exams[:3]:
                created_at = exam.get('created_at', '')
                topic = exam.get('topic', 'Unknown')
                click.echo(f" - {exam['title']} (Topic: {topic}, Created: {created_at[:19]})")

    except Exception as e:
        click.echo(f"🗄️ Supabase Status: Error connecting ({e})")

    # Check generated exam files locally
    exams_dir = Path("data/output/generated_exams")
    if exams_dir.exists():
        exam_files = list(exams_dir.glob("*.json"))
        txt_files = list(exams_dir.glob("*.txt"))
        md_files = list(exams_dir.glob("*.md"))
        pdf_files = list(exams_dir.glob("*.pdf"))
        total_files = len(exam_files + txt_files + md_files + pdf_files)
        click.echo(f"📋 Generated exam files (local): {total_files}")
        
        if total_files > 0:
            click.echo("📋 Recent local exam files:")
            all_files = sorted(exam_files + txt_files + md_files + pdf_files,
                             key=lambda x: x.stat().st_mtime, reverse=True)
            for file_path in all_files[:5]:
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                click.echo(f" - {file_path.name} (created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
    else:
        click.echo("📋 Generated exams (local): Not found")

@cli.command()
def test_database():
    """Test database connection and insertion"""
    try:
        vector_store = VectorStore()
        # Test document insertion
        test_doc = Document(
            title="Test Document",
            content="This is a test document content for verifying database connectivity.",
            source_file="test.txt",
            paper_set="test_set",
            paper_number="1",
            metadata={"test": True}
        )

        doc_id = vector_store.insert_document(test_doc)
        logger.info(f"✅ Successfully inserted test document with ID: {doc_id}")

        # Test chunk insertion
        test_chunk = TextChunk(
            document_id=doc_id,
            chunk_text="This is a test chunk",
            chunk_index=0,
            chunk_size=19,
            overlap_size=0,
            metadata={"test": True}
        )

        chunk_ids = vector_store.insert_text_chunks([test_chunk])
        logger.info(f"✅ Successfully inserted test chunk with ID: {chunk_ids[0]}")

        # Test duplicate checking
        existing_doc = vector_store.document_exists_by_source_file("test.txt")
        if existing_doc:
            logger.info(f"✅ Duplicate checking works: Found existing document ID {existing_doc['id']}")
        else:
            logger.warning("⚠️ Duplicate checking failed: Should have found existing document")

        # Check counts
        stats = vector_store.get_database_stats()
        logger.info(f"📊 Database stats: {stats}")

    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")

@cli.command()
@click.option('--source-file', help='Check specific source file for duplicates')
def check_duplicates(source_file):
    """Check for duplicate documents in the database"""
    try:
        vector_store = VectorStore()

        if source_file:
            # Check specific file
            existing_doc = vector_store.document_exists_by_source_file(source_file)
            if existing_doc:
                click.echo(f"✅ Document exists: {source_file}")
                click.echo(f" Document ID: {existing_doc['id']}")
                click.echo(f" Title: {existing_doc['title']}")
                click.echo(f" Created: {existing_doc.get('created_at', 'Unknown')}")

                # Check chunks
                chunks = vector_store.get_chunks_by_source_file(source_file)
                click.echo(f" Chunks: {len(chunks)}")

                # Check embeddings
                chunks_with_embeddings = sum(1 for chunk in chunks
                                           if vector_store.embedding_exists_for_chunk(chunk['id']))
                click.echo(f" Embeddings: {chunks_with_embeddings}/{len(chunks)}")

            else:
                click.echo(f"❌ Document does not exist: {source_file}")

        else:
            # Show all documents
            all_docs = vector_store.get_all_documents()
            click.echo(f"📄 Total documents in database: {len(all_docs)}")

            if all_docs:
                click.echo("\n📋 Documents in database:")
                for doc in all_docs[:10]:  # Show first 10
                    chunks = vector_store.get_chunks_by_document(doc['id'])
                    chunks_with_embeddings = sum(1 for chunk in chunks
                                                if vector_store.embedding_exists_for_chunk(chunk['id']))
                    click.echo(f" - {doc['source_file']} (ID: {doc['id']}, Chunks: {len(chunks)}, Embeddings: {chunks_with_embeddings})")

                if len(all_docs) > 10:
                    click.echo(f" ... and {len(all_docs) - 10} more")

    except Exception as e:
        logger.error(f"❌ Failed to check duplicates: {e}")

@cli.command()
def test_supabase():
    """Test Supabase connection and schema"""
    try:
        vector_store = VectorStore()

        # Test connection
        logger.info("🔄 Testing Supabase connection...")
        stats = vector_store.get_database_stats()
        logger.info(f"✅ Connection successful: {stats}")

        # Test schema validation
        schema_valid = vector_store.validate_database_schema()
        logger.info(f"📊 Schema validation: {schema_valid}")

        # Test exam table specifically
        try:
            test_exam = {
                'exam_metadata': {'title': 'Test Exam', 'topic': 'Test', 'total_marks': 100},
                'generation_stats': {'questions_generated': 1},
                'questions': {'Q1': {'question': 'Test question'}}
            }

            exam_id = vector_store.save_generated_exam(test_exam)
            logger.info(f"✅ Test exam save successful: ID {exam_id}")

            # Clean up test data
            vector_store.client.client.table('generated_exams').delete().eq('id', exam_id).execute()
            logger.info("🧹 Test data cleaned up")

        except Exception as save_error:
            logger.error(f"❌ Exam save test failed: {save_error}")

    except Exception as e:
        logger.error(f"❌ Supabase test failed: {e}")

if __name__ == '__main__':
    # Final startup log
    logger.info("🎯 Gemini Embedding Model Pipeline CLI ready with comprehensive logging")
    logger.info("📋 Available commands: process-texts, generate-embeddings, generate-structured-exam, run-full-pipeline")
    logger.info("📋 Logging commands: logs, log-status, clear-logs")
    logger.info("📋 Utility commands: status, test-database, check-duplicates, test-supabase")
    logger.info(f"📁 All operations logged to: {logs_dir}")
    
    cli()
#!/usr/bin/env python3

"""
Main pipeline controller for embedding-based exam generation system.
Enhanced with Supabase integration, quota-aware generation, and duplicate checking.
NOW WITH COMPREHENSIVE FILE LOGGING TO data/output/logs/
"""

import sys
import os
import click
from pathlib import Path
import json
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
import time

# Load environment variables from .env file
load_dotenv()

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ===== INITIALIZE COMPREHENSIVE LOGGING SYSTEM FIRST =====
# Import and setup logging BEFORE any other project imports
from src.core.utils.logging_config import (
    initialize_logging,
    log_system_information,
    log_pipeline_start,
    log_pipeline_end,
    create_operation_context,
    get_log_stats
)

# Initialize complete logging system
logs_dir = initialize_logging()

# NOW import loguru logger (after configuration)
from loguru import logger

# Log pipeline startup
logger.info("🚀 Starting gemini-embedding-model pipeline with comprehensive file logging")
logger.info(f"📁 All logs will be saved to: {logs_dir}")

# Import project modules (after logging setup)
from config.settings import BATCH_SIZE
from src.core.text.text_loader import TextLoader
from src.core.text.chunker import TextChunker
from src.core.embedding.embedding_generator import EmbeddingGenerator
from src.core.generation.structure_generator import StructureGenerator
from src.core.storage.vector_store import VectorStore, Document, TextChunk, Embedding

@click.group()
def cli():
    """Embedding-Based Exam Generation Pipeline CLI with Comprehensive File Logging"""
    logger.info("🎯 CLI initialized - all operations will be logged to files")

@cli.command()
@click.option('--input-dir', default='data/output/converted_markdown', help='Input directory (converted markdown files)')  # UPDATED
@click.option('--use-supabase', is_flag=True, default=True, help='Store data in Supabase')
@click.option('--force-reprocess', is_flag=True, default=False, help='Force reprocess existing files')
@create_operation_context("Text Processing Pipeline")
def process_texts(input_dir, use_supabase, force_reprocess):
    """Load and process markdown files converted from PDFs with duplicate checking and Supabase storage"""
    
    # Log pipeline start with parameters
    log_pipeline_start("process_texts", {
        "input_dir": input_dir,
        "use_supabase": use_supabase,
        "force_reprocess": force_reprocess,
        "input_type": "converted_markdown"  # NEW
    })
    
    start_time = time.time()
    
    try:
        # Ensure output directory exists
        output_dir = Path("data/output/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Output directory: {output_dir}")
        
        # Initialize components
        logger.info("🔧 Initializing text processing components for markdown files")
        loader = TextLoader()
        chunker = TextChunker()
        vector_store = VectorStore() if use_supabase else None
        logger.info("✅ Components initialized successfully")
        
        # Load documents using markdown processing
        logger.info(f"📂 Loading markdown documents from: {input_dir}")
        documents = loader.process_directory(Path(input_dir))  # This now handles markdown
        logger.info(f"📄 Found {len(documents)} documents to process")
        
        if not documents:
            logger.error(f"❌ No markdown documents found in {input_dir}")
            logger.info("💡 Make sure PDF files have been converted to markdown first")
            raise ValueError(f"No markdown documents found in {input_dir}")
        
        # Log content statistics
        stats = loader.get_content_statistics()
        logger.info(f"📊 Content Statistics: {stats}")
        
        # Continue with existing processing logic...
        # [Rest of the method continues with existing chunking and database operations]
        
        # Process documents into chunks with duplicate checking
        all_chunks = []
        supabase_results = []
        processed_count = 0
        skipped_count = 0
        
        for doc_index, doc in enumerate(documents, 1):
            source_file = doc.source_file
            logger.info(f"🔍 Processing document {doc_index}/{len(documents)}: {source_file}")
            
            # Check if document already exists in database
            if use_supabase and vector_store and not force_reprocess:
                logger.info(f"🔍 Checking for existing document: {source_file}")
                existing_doc = vector_store.document_exists_by_source_file(source_file)
                if existing_doc:
                    logger.info(f"⏭️ Document already exists (ID: {existing_doc['id']}): {source_file}, skipping processing")
                    skipped_count += 1
                    continue
            
            logger.info(f"🔄 Processing NEW document: {source_file}")
            processed_count += 1
            
            # Create chunks
            logger.info(f"✂️ Chunking document content ({len(doc.content)} characters)")
            chunks = chunker.chunk_text(doc.content)
            logger.info(f"📝 Created {len(chunks)} chunks for {doc.source_file}")
            
            if not chunks:
                logger.warning(f"⚠️ No chunks created for {source_file}")
                continue
            
            # Store in Supabase if enabled
            if use_supabase and vector_store:
                try:
                    logger.info("💾 Storing document and chunks in Supabase database")
                    # Create Document object for Supabase
                    supabase_doc = Document(
                        title=Path(doc.source_file).stem,
                        content=doc.content,
                        source_file=doc.source_file,
                        paper_set=doc.paper_set,
                        paper_number=doc.paper_number,
                        metadata=doc.metadata
                    )
                    
                    # Insert document to database
                    doc_id = vector_store.insert_document(supabase_doc)
                    logger.info(f"✅ Document stored in Supabase with ID: {doc_id}")
                    
                    # Create and insert chunks to database
                    chunk_objects = []
                    for i, chunk_text in enumerate(chunks):
                        chunk_objects.append(TextChunk(
                            document_id=doc_id,
                            chunk_text=chunk_text,
                            chunk_index=i,
                            chunk_size=len(chunk_text),
                            overlap_size=0,
                            metadata={"source_file": doc.source_file}
                        ))
                    
                    chunk_ids = vector_store.insert_text_chunks(chunk_objects)
                    logger.info(f"✅ Stored {len(chunk_ids)} chunks in Supabase database")
                    
                    supabase_results.append({
                        'document_id': doc_id,
                        'chunk_ids': chunk_ids,
                        'source_file': doc.source_file,
                        'status': 'new'
                    })
                
                except Exception as e:
                    logger.error(f"❌ Failed to store document in Supabase: {e}")
                    logger.warning("⚠️ Continuing with local processing only")
            
            # Create local chunks data for JSON storage
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    "id": f"{doc.paper_set}_{doc.paper_number}_{i}",
                    "chunk_text": chunk,
                    "chunk_index": i,
                    "chunk_size": len(chunk),
                    "source_file": doc.source_file,
                    "paper_set": doc.paper_set,
                    "paper_number": doc.paper_number,
                    "metadata": doc.metadata
                }
                all_chunks.append(chunk_data)
        
        # Save processed chunks locally
        chunks_path = output_dir / "processed_chunks.json"
        logger.info(f"💾 Saving {len(all_chunks)} chunks to: {chunks_path}")
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Chunks saved successfully to {chunks_path}")
        
        # Save processed documents locally
        docs_path = output_dir / "processed_documents.json"
        logger.info(f"💾 Saving documents metadata to: {docs_path}")
        loader.save_processed_documents(docs_path)
        logger.info(f"✅ Documents metadata saved to {docs_path}")
        
        # Calculate final statistics
        duration = time.time() - start_time
        results = {
            "processed_documents": processed_count,
            "skipped_documents": skipped_count,
            "total_documents": len(documents),
            "total_chunks": len(all_chunks),
            "duration_seconds": round(duration, 2),
            "chunks_per_second": round(len(all_chunks) / duration, 2) if duration > 0 else 0
        }
        
        # Log comprehensive completion statistics
        logger.info("✅ Text processing completed successfully")
        logger.info(f"📊 FINAL PROCESSING STATISTICS:")
        logger.info(f" • Total documents found: {len(documents)}")
        logger.info(f" • New documents processed: {processed_count}")
        logger.info(f" • Existing documents skipped: {skipped_count}")
        logger.info(f" • Total chunks created: {len(all_chunks)}")
        logger.info(f" • Processing duration: {duration:.2f} seconds")
        
        log_pipeline_end("process_texts", success=True, duration=duration, results=results)
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        logger.error(f"❌ Text processing failed after {duration:.2f} seconds: {error_msg}")
        log_pipeline_end("process_texts", success=False, duration=duration, error=error_msg)
        raise

# [Include all other CLI commands: generate_embeddings, generate_structured_exam, run_full_pipeline, logs, etc.]
# [The rest of the file continues with the existing commands...]
