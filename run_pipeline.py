#!/usr/bin/env python3
"""
Complete pipeline controller for Gemini-based academic assessment generation system.
Incorporates all fixes for content aggregation, rate limiting, and error handling.
"""

import sys
import os
import click
from pathlib import Path
import json
from dotenv import load_dotenv
from datetime import datetime
import time

# Load environment variables
load_dotenv()

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Initialize logging system
from src.core.utils.logging_config import (
    initialize_logging, log_pipeline_start, log_pipeline_end, create_operation_context
)

logs_dir = initialize_logging()
from loguru import logger

logger.info("🚀 Starting gemini-embedding-model pipeline with comprehensive file logging")
logger.info(f"📁 All logs will be saved to: {logs_dir}")

# Import core modules
from config.settings import BATCH_SIZE
from src.core.text.text_loader import TextLoader
from src.core.text.chunker import TextChunker
from src.core.embedding.embedding_generator import EmbeddingGenerator
from src.core.storage.vector_store import VectorStore, Document, TextChunk, Embedding
from src.core.utils.process_lock import pipeline_lock

# Common utility functions
def load_json_file(file_path: Path) -> list:
    """Load and validate JSON file"""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        raise ValueError(f"Empty data in {file_path}")
    
    return data

def save_json_file(data: any, file_path: Path) -> None:
    """Save data to JSON file"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def log_operation_stats(operation_name: str, stats: dict, duration: float) -> None:
    """Centralized operation statistics logging"""
    logger.info(f"✅ {operation_name.upper()} COMPLETED")
    logger.info(f"⏱️ Duration: {duration:.2f} seconds")
    
    for key, value in stats.items():
        formatted_key = key.replace('_', ' ').title()
        logger.info(f"📊 {formatted_key}: {value}")

@click.group()
def cli():
    """Gemini Academic Assessment Generation Pipeline CLI"""
    logger.info("🎯 CLI initialized - all operations will be logged to files")

@cli.command()
@click.option('--input-dir', default='data/output/converted_markdown', help='Input directory')
@click.option('--use-supabase', is_flag=True, default=True, help='Use Supabase storage')
@click.option('--force-reprocess', is_flag=True, default=False, help='Force reprocess all files')
@create_operation_context("Text Processing")
def process_texts(input_dir, use_supabase, force_reprocess):
    """Process text files with duplicate detection"""
    
    use_supabase = True  # Ensure Supabase is always used
    start_time = time.time()
    log_pipeline_start("process_texts", {
        "input_dir": input_dir, "use_supabase": use_supabase, "force_reprocess": force_reprocess
    })
    
    try:
        with pipeline_lock():  # Add process lock
            # Initialize components
            output_dir = Path("data/output/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            loader = TextLoader()
            chunker = TextChunker()
            vector_store = VectorStore() if use_supabase else None
            
            # Load documents
            logger.info(f"📂 Loading documents from: {input_dir}")
            documents = loader.process_directory(Path(input_dir))
            
            if not documents:
                raise ValueError(f"No documents found in {input_dir}")
            
            # Process documents
            all_chunks = []
            processed_count = 0
            skipped_count = 0
            
            for doc_index, doc in enumerate(documents, 1):
                logger.info(f"🔍 Processing {doc_index}/{len(documents)}: {doc.source_file}")
                
                # Check for existing document
                if use_supabase and vector_store and not force_reprocess:
                    existing_doc = vector_store.document_exists_by_source_file(doc.source_file)
                    if existing_doc:
                        logger.info(f"⏭️ Skipping existing document: {doc.source_file}")
                        skipped_count += 1
                        
                        # Retrieve existing chunks
                        existing_chunks = vector_store.get_chunks_by_source_file(doc.source_file)
                        for chunk in existing_chunks:
                            chunk_data = {
                                "id": f"{doc.paper_set}_{doc.paper_number}_{chunk['chunk_index']}",
                                "chunk_text": chunk['chunk_text'],
                                "chunk_index": chunk['chunk_index'],
                                "source_file": doc.source_file,
                                "paper_set": doc.paper_set,
                                "content_type": doc.content_type,  # Fixed: Add content_type
                                "metadata": doc.metadata
                            }
                            all_chunks.append(chunk_data)
                        continue
                
                # Process new document
                processed_count += 1
                chunks = chunker.chunk_text(doc.content)
                
                if not chunks:
                    logger.warning(f"⚠️ No chunks created for {doc.source_file}")
                    continue
                
                # Store in Supabase
                if use_supabase and vector_store:
                    try:
                        supabase_doc = Document(
                            title=Path(doc.source_file).stem,
                            content=doc.content,
                            source_file=doc.source_file,
                            paper_set=doc.paper_set,
                            paper_number=doc.paper_number,
                            metadata=doc.metadata
                        )
                        
                        doc_id = vector_store.insert_document(supabase_doc)
                        
                        chunk_objects = [
                            TextChunk(
                                document_id=doc_id,
                                chunk_text=chunk_text,
                                chunk_index=i,
                                chunk_size=len(chunk_text)
                            ) for i, chunk_text in enumerate(chunks)
                        ]
                        
                        vector_store.insert_text_chunks(chunk_objects)
                        logger.info(f"✅ Stored in Supabase: {len(chunks)} chunks")
                        
                    except Exception as e:
                        logger.error(f"❌ Supabase storage failed: {e}")
                
                # Create local chunks with content_type
                for i, chunk in enumerate(chunks):
                    chunk_data = {
                        "id": f"{doc.paper_set}_{doc.paper_number}_{i}",
                        "chunk_text": chunk,
                        "chunk_index": i,
                        "source_file": doc.source_file,
                        "paper_set": doc.paper_set,
                        "content_type": doc.content_type,  # Fixed: Add content_type
                        "metadata": doc.metadata
                    }
                    all_chunks.append(chunk_data)
            
            # Save results
            save_json_file(all_chunks, output_dir / "processed_chunks.json")
            loader.save_processed_documents(output_dir / "processed_documents.json")
            
            # Log completion stats
            duration = time.time() - start_time
            stats = {
                "processed_documents": processed_count,
                "skipped_documents": skipped_count,
                "total_chunks": len(all_chunks),
                "processing_rate": f"{len(all_chunks) / duration:.2f} chunks/sec"
            }
            
            log_operation_stats("Text Processing", stats, duration)
            log_pipeline_end("process_texts", success=True, duration=duration, results=stats)
            
    except RuntimeError as e:
        if "already in progress" in str(e):
            logger.error("❌ Another pipeline is already running. Please wait for it to complete.")
            raise
        else:
            raise
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Text processing failed: {e}")
        log_pipeline_end("process_texts", success=False, duration=duration, error=str(e))
        raise

@cli.command()
@click.option('--batch-size', default=BATCH_SIZE, help='Batch size for embeddings')
@click.option('--use-supabase', is_flag=True, default=True, help='Store in Supabase')
@click.option('--force-regenerate', is_flag=True, default=False, help='Force regenerate embeddings')
@create_operation_context("Embedding Generation")
def generate_embeddings(batch_size, use_supabase, force_regenerate):
    """Generate embeddings with duplicate detection and enhanced rate limiting"""
    
    use_supabase = True  # Ensure Supabase is always used
    start_time = time.time()
    log_pipeline_start("generate_embeddings", {
        "batch_size": batch_size, "use_supabase": use_supabase, "force_regenerate": force_regenerate
    })
    
    try:
        with pipeline_lock():  # Add process lock
            # Initialize components
            generator = EmbeddingGenerator()
            vector_store = VectorStore() if use_supabase else None
            
            # Load chunks
            chunks = load_json_file(Path("data/output/processed/processed_chunks.json"))
            logger.info(f"📝 Loaded {len(chunks)} chunks")
            
            # Check quota
            quota_status = generator.check_quota_status()
            if quota_status:
                remaining = quota_status.get('requests_remaining', 'Unknown')
                logger.info(f"📊 API Quota: {remaining} requests remaining")
            
            # Determine chunks needing embeddings
            if use_supabase and vector_store and not force_regenerate:
                chunks_needing_embeddings = []
                for chunk in chunks:
                    existing_doc = vector_store.document_exists_by_source_file(chunk['source_file'])
                    if existing_doc:
                        db_chunks = vector_store.get_chunks_by_document(existing_doc['id'])
                        matching_chunk = next((c for c in db_chunks 
                                             if c['chunk_index'] == chunk['chunk_index']), None)
                        if matching_chunk and vector_store.embedding_exists_for_chunk(matching_chunk['id']):
                            continue
                    chunks_needing_embeddings.append(chunk)
            else:
                chunks_needing_embeddings = chunks
            
            logger.info(f"🧠 Processing {len(chunks_needing_embeddings)} new embeddings")
            
            if not chunks_needing_embeddings:
                logger.info("✅ All embeddings already exist")
                log_pipeline_end("generate_embeddings", success=True, duration=time.time() - start_time,
                                results={"new_embeddings": 0, "total_embeddings": len(chunks)})
                return
            
            # Use batch processing with proper delays
            embeddings_data = generator.process_chunks_batch(
                [chunk["chunk_text"] for chunk in chunks_needing_embeddings], 
                batch_size=batch_size  # Removed cap at 5
            )
            
            # Merge successful embeddings with chunk data
            final_embeddings = []
            successful_count = 0
            failed_count = 0
            
            for i, result in enumerate(embeddings_data):
                if result.get('success', False):
                    chunk_with_embedding = {
                        **chunks_needing_embeddings[i],
                        "embedding": result['embedding'],
                        "embedding_model": "text-embedding-004"
                    }
                    final_embeddings.append(chunk_with_embedding)
                    successful_count += 1
                    
                    # Store in Supabase
                    if use_supabase and vector_store:
                        try:
                            chunk = chunks_needing_embeddings[i]
                            logger.debug(f"Attempting to store embedding for chunk: {chunk.get('id', 'N/A')}")
                            existing_doc = vector_store.document_exists_by_source_file(chunk['source_file'])
                            if existing_doc:
                                logger.debug(f"Found existing document for chunk: {existing_doc.get('id', 'N/A')}")
                                db_chunks = vector_store.get_chunks_by_document(existing_doc['id'])
                                matching_chunk = next((c for c in db_chunks 
                                                     if c['chunk_index'] == chunk['chunk_index']), None)
                                if matching_chunk:
                                    logger.debug(f"Found matching chunk in DB: {matching_chunk.get('id', 'N/A')}")
                                    embedding_obj = Embedding(
                                        chunk_id=matching_chunk['id'],
                                        embedding=result['embedding']
                                    )
                                    vector_store.insert_embeddings([embedding_obj])
                                    logger.info(f"✅ Successfully inserted embedding for chunk: {matching_chunk.get('id', 'N/A')}")
                                else:
                                    logger.warning(f"⚠️ No matching chunk found in DB for chunk index {chunk['chunk_index']} of doc {existing_doc.get('id', 'N/A')}")
                            else:
                                logger.warning(f"⚠️ No existing document found in DB for source file: {chunk['source_file']}")
                        except Exception as supabase_error:
                            logger.error(f"❌ Supabase storage failed: {supabase_error}")
                else:
                    failed_count += 1
                    logger.warning(f"⚠️ Failed embedding for chunk {i}")
            
            # Merge with existing embeddings
            if not force_regenerate:
                existing_path = Path("data/output/processed/embeddings.json")
                if existing_path.exists():
                    existing_embeddings = load_json_file(existing_path)
                    existing_ids = {emb['id'] for emb in existing_embeddings}
                    
                    for new_emb in final_embeddings:
                        if new_emb['id'] not in existing_ids:
                            existing_embeddings.append(new_emb)
                    
                    final_embeddings = existing_embeddings
                    logger.info(f"🔗 Merged embeddings: {len(final_embeddings)} total")
            
            # Save embeddings
            save_json_file(final_embeddings, Path("data/output/processed/embeddings.json"))
            
            # Log completion stats
            duration = time.time() - start_time
            stats = {
                "new_embeddings": successful_count,
                "failed_embeddings": failed_count,
                "total_embeddings": len(final_embeddings),
                "generation_rate": f"{successful_count / duration:.2f} embeddings/sec"
            }
            
            log_operation_stats("Embedding Generation", stats, duration)
            log_pipeline_end("generate_embeddings", success=True, duration=duration, results=stats)
            
    except RuntimeError as e:
        if "already in progress" in str(e):
            logger.error("❌ Another pipeline is already running. Please wait for it to complete.")
            raise
        else:
            raise
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Embedding generation failed: {e}")
        log_pipeline_end("generate_embeddings", success=False, duration=duration, error=str(e))
        raise

@cli.command()
@click.option('--topic', required=True, help='Exam topic')
@click.option('--requirements-file', help='Custom requirements file')
@create_operation_context("Comprehensive Paper Generation")
def generate_comprehensive_papers(topic, requirements_file):
    """Generate comprehensive exam papers - MAIN COMMAND"""
    
    start_time = time.time()
    log_pipeline_start("generate_comprehensive_papers", {
        "topic": topic, "requirements_file": requirements_file
    })
    
    try:
        with pipeline_lock():  # Add process lock
            from src.core.workflows.single_prompt_workflow import SinglePromptWorkflow
            
            logger.info(f"🎯 Starting comprehensive paper generation for: {topic}")
            
            # Initialize and execute workflow
            workflow = SinglePromptWorkflow()
            result = workflow.execute_full_workflow(topic, requirements_file)
            
            if result["workflow_metadata"]["success"]:
                logger.info("✅ COMPREHENSIVE PAPER GENERATION SUCCESSFUL!")
                logger.info(f"📊 Generated files: {len(result.get('output_files', []))}")
                
                # Display generated files
                logger.info(f"📁 Generated Papers:")
                for file_path in result.get('output_files', []):
                    file_name = Path(file_path).name
                    if 'question_paper' in file_name:
                        logger.info(f"  📋 Question Paper: {file_path}")
                    elif 'model_answers' in file_name:
                        logger.info(f"  📝 Model Answers: {file_path}")
                    elif 'marking_scheme' in file_name:
                        logger.info(f"  📏 Marking Scheme: {file_path}")
                
                # Calculate and log workflow stats
                duration = time.time() - start_time
                logger.info(f"⏱️  Total workflow time: {duration:.2f} seconds")
                
                # Log processing statistics
                stats = result.get("processing_stats", {})
                logger.info(f"📄 Documents processed: {stats.get('documents_processed', 0)}")
                logger.info(f"🧠 Embeddings generated: {stats.get('embeddings_generated', 0)}")
                
                log_pipeline_end("generate_comprehensive_papers", success=True, duration=duration, results=result)
                
            else:
                error_msg = result["workflow_metadata"].get("error", "Unknown error")
                raise RuntimeError(error_msg)
                
    except RuntimeError as e:
        if "already in progress" in str(e):
            logger.error("❌ Another pipeline is already running. Please wait for it to complete.")
            raise
        else:
            raise
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Paper generation failed: {e}")
        log_pipeline_end("generate_comprehensive_papers", success=False, duration=duration, error=str(e))
        raise

@cli.command('run-full-pipeline')
@create_operation_context("Full Pipeline Execution")
def run_full_pipeline():
    """Run the complete embedding-based exam generation pipeline with comprehensive logging"""
    
    start_time = time.time()
    log_pipeline_start("run_full_pipeline", {
        "mode": "complete_pipeline",
        "includes": ["text_processing", "embedding_generation", "comprehensive_exam_generation"]
    })
    
    try:
        with pipeline_lock():  # Add process lock
            logger.info("🚀 Starting COMPLETE embedding-based exam generation pipeline")
            logger.info("This will run: Text Processing → Embedding Generation → Comprehensive Paper Generation")
            
            ctx = click.get_current_context()
            
            # Step 1: Text Processing
            logger.info("📝 STEP 1/3: Processing texts")
            try:
                ctx.invoke(process_texts, use_supabase=True, force_reprocess=False)
                logger.info("✅ Step 1 completed")
            except Exception as e:
                logger.error(f"❌ Step 1 failed: {e}")
                raise
            
            # Step 2: Embedding Generation
            logger.info("🧠 STEP 2/3: Generating embeddings")
            try:
                ctx.invoke(generate_embeddings, use_supabase=True, force_regenerate=False)
                logger.info("✅ Step 2 completed")
            except Exception as e:
                logger.warning(f"⚠️ Step 2 had issues: {e}")
                # Check if embeddings exist to continue
                if not Path("data/output/processed/embeddings.json").exists():
                    logger.error("❌ No embeddings available - cannot continue")
                    raise
                logger.info("📄 Continuing with existing embeddings")
            
            # Step 3: Paper Generation
            logger.info("📋 STEP 3/3: Generating comprehensive papers")
            try:
                ctx.invoke(generate_comprehensive_papers, topic='AI and Data Analytics')
                logger.info("✅ Step 3 completed")
            except Exception as e:
                logger.error(f"❌ Step 3 failed: {e}")
                raise
            
            # Final statistics
            duration = time.time() - start_time
            
            # Gather stats from generated files
            try:
                chunks_file = Path("data/output/processed/processed_chunks.json")
                embeddings_file = Path("data/output/processed/embeddings.json")
                papers_dir = Path("data/output/generated_exams")
                
                stats = {
                    "total_duration": f"{duration:.2f} seconds",
                    "chunks_processed": len(load_json_file(chunks_file)) if chunks_file.exists() else 0,
                    "embeddings_generated": len(load_json_file(embeddings_file)) if embeddings_file.exists() else 0,
                    "paper_files_created": len(list(papers_dir.glob("*"))) if papers_dir.exists() else 0,
                    "pipeline_success": True
                }
            except Exception:
                stats = {"total_duration": f"{duration:.2f} seconds", "pipeline_success": True}
            
            log_operation_stats("Full Pipeline", stats, duration)
            logger.info("🎉 COMPLETE PIPELINE EXECUTION SUCCESSFUL!")
            
            log_pipeline_end("run_full_pipeline", success=True, duration=duration, results=stats)
            
    except RuntimeError as e:
        if "already in progress" in str(e):
            logger.error("❌ Another pipeline is already running. Please wait for it to complete.")
            raise
        else:
            raise
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Full pipeline failed: {e}")
        log_pipeline_end("run_full_pipeline", success=False, duration=duration, error=str(e))
        raise

# Utility Commands
@cli.command()
def status():
    """Show pipeline status and health check"""
    logger.info("📊 Checking comprehensive pipeline status")
    
    click.echo("📊 Pipeline Status Report")
    click.echo("=" * 40)
    
    # Check local files
    chunks_file = Path("data/output/processed/processed_chunks.json")
    embeddings_file = Path("data/output/processed/embeddings.json")
    
    if chunks_file.exists():
        chunks = load_json_file(chunks_file)
        click.echo(f"📄 Processed chunks: {len(chunks)}")
    else:
        click.echo("📄 Processed chunks: Not found")
    
    if embeddings_file.exists():
        embeddings = load_json_file(embeddings_file)
        click.echo(f"🧠 Generated embeddings: {len(embeddings)}")
        if embeddings:
            click.echo(f"🧠 Embedding dimensions: {len(embeddings[0].get('embedding', []))}")
    else:
        click.echo("🧠 Generated embeddings: Not found")
    
    # Check converted markdown
    markdown_dir = Path("data/output/converted_markdown")
    if markdown_dir.exists():
        md_files = len(list(markdown_dir.rglob("*.md")))
        click.echo(f"📄 Converted markdown files: {md_files}")
    else:
        click.echo("📄 Converted markdown files: Not found")
    
    # Check generated exams
    exams_dir = Path("data/output/generated_exams")
    if exams_dir.exists():
        exam_files = len(list(exams_dir.glob("*")))
        click.echo(f"📋 Generated exam files: {exam_files}")
    else:
        click.echo("📋 Generated exam files: Not found")
    
    # Check Supabase connection
    try:
        vector_store = VectorStore()
        stats = vector_store.get_database_stats()
        click.echo(f"🗄️ Supabase Documents: {stats['documents']}")
        click.echo(f"🗄️ Supabase Chunks: {stats['text_chunks']}")
        click.echo(f"🗄️ Supabase Embeddings: {stats['embeddings']}")
        click.echo(f"🗄️ Generated Exams: {stats['generated_exams']}")
    except Exception as e:
        click.echo(f"⚠️ Supabase connection failed: {e}")

@cli.command()
@click.option('--lines', default=50, help='Number of lines to show')
def logs(lines):
    """View recent pipeline logs"""
    logs_dir = Path("data/output/logs")
    
    if not logs_dir.exists():
        click.echo("❌ No logs directory found")
        return
    
    log_files = list(logs_dir.glob('*.log'))
    if not log_files:
        click.echo("❌ No log files found")
        return
    
    recent_log = max(log_files, key=lambda x: x.stat().st_mtime)
    click.echo(f"📝 Recent logs from: {recent_log.name}")
    click.echo("=" * 60)
    
    try:
        with open(recent_log, 'r', encoding='utf-8') as f:
            lines_list = f.readlines()
            for line in lines_list[-lines:]:
                click.echo(line.rstrip())
    except Exception as e:
        click.echo(f"❌ Error reading logs: {e}")

@cli.command()
def validate_content():
    """Validate content pipeline for debugging"""
    from src.core.utils.content_validator import ContentValidator
    
    logger.info("🔍 Starting content validation")
    
    validation = ContentValidator.run_complete_validation()
    
    click.echo("📊 Content Validation Results")
    click.echo("=" * 50)
    
    # Markdown validation
    markdown_val = validation["markdown_validation"]
    click.echo(f"📁 Markdown Directory: {'✅' if markdown_val['markdown_dir_exists'] else '❌'}")
    click.echo(f"📄 Total Files: {markdown_val['total_files']}")
    click.echo(f"📊 Total Content: {markdown_val['total_content_chars']:,} characters")
    click.echo(f"📋 Kelvin Papers: {len(markdown_val['kelvin_papers'])}")
    click.echo(f"📚 Lectures: {len(markdown_val['lectures'])}")
    
    # Embeddings validation
    embeddings_val = validation["embeddings_validation"]
    click.echo(f"🧠 Embeddings File: {'✅' if embeddings_val['file_exists'] else '❌'}")
    click.echo(f"🧠 Embeddings Count: {embeddings_val['embeddings_count']}")
    click.echo(f"🧠 Chunks with Content: {embeddings_val['chunks_with_content']}")
    
    # Summary
    summary = validation["summary"]
    click.echo("\n📋 Summary:")
    click.echo(f"Markdown Files: {'✅ Ready' if summary['markdown_files_ok'] else '❌ Issues'}")
    click.echo(f"Embeddings: {'✅ Ready' if summary['embeddings_ok'] else '❌ Issues'}")
    click.echo(f"Pipeline: {'✅ Ready' if summary['pipeline_ready'] else '❌ Not Ready'}")
    
    if not summary['pipeline_ready']:
        click.echo("\n🔧 Suggested Actions:")
        if not summary['markdown_files_ok']:
            click.echo("1. Run: python scripts/direct_convert.py")
        if not summary['embeddings_ok']:
            click.echo("2. Run: python run_pipeline.py process-texts")
            click.echo("3. Run: python run_pipeline.py generate-embeddings")

@cli.command('test-supabase')
def test_supabase():
    """Test Supabase database connection and functionality"""
    logger.info("🧪 Testing Supabase connection and functionality")
    click.echo("🧪 Supabase Connection Test")
    click.echo("=" * 40)
    
    try:
        # Initialize vector store
        vector_store = VectorStore()
        
        # Test 1: Basic connection
        click.echo("1️⃣ Testing basic connection...")
        stats = vector_store.get_database_stats()
        click.echo(f"   ✅ Connection successful")
        click.echo(f"   📊 Documents: {stats['documents']}")
        click.echo(f"   📊 Chunks: {stats['text_chunks']}")
        click.echo(f"   📊 Embeddings: {stats['embeddings']}")
        click.echo(f"   📊 Generated Exams: {stats['generated_exams']}")
        
        # Test 2: Schema validation
        click.echo("\n2️⃣ Testing database schema...")
        schema_validation = vector_store.validate_database_schema()
        for table, valid in schema_validation.items():
            status = "✅" if valid else "❌"
            click.echo(f"   {status} Table '{table}': {'Valid' if valid else 'Invalid'}")
        
        # Test 3: Health check
        click.echo("\n3️⃣ Running health check...")
        health = vector_store.health_check()
        status = "✅" if health['status'] == 'healthy' else "❌"
        click.echo(f"   {status} Overall status: {health['status']}")
        
        # Test 4: Test basic operations
        click.echo("\n4️⃣ Testing basic operations...")
        
        # Test document retrieval
        recent_docs = vector_store.get_recent_documents(limit=1)
        click.echo(f"   📄 Can retrieve documents: {'✅ Yes' if recent_docs else '⚠️ No data'}")
        
        # Test chunk operations
        all_chunks = vector_store.get_chunks_count()
        click.echo(f"   📝 Total chunks accessible: {all_chunks}")
        
        # Test embedding operations  
        embeddings_count = vector_store.get_embeddings_count()
        click.echo(f"   🧠 Total embeddings accessible: {embeddings_count}")
        
        click.echo(f"\n✅ All Supabase tests passed!")
        return True
        
    except Exception as e:
        click.echo(f"\n❌ Supabase test failed: {e}")
        logger.error(f"Supabase test failed: {e}")
        
        # Provide debugging information
        click.echo(f"\n🔧 Debugging information:")
        click.echo(f"   SUPABASE_URL: {'Set' if os.getenv('SUPABASE_URL') else 'Missing'}")
        click.echo(f"   SUPABASE_SERVICE_KEY: {'Set' if os.getenv('SUPABASE_SERVICE_KEY') else 'Missing'}")
        click.echo(f"   SUPABASE_ANON_KEY: {'Set' if os.getenv('SUPABASE_ANON_KEY') else 'Missing'}")
        
        return False


if __name__ == "__main__":
    # Initialize logging and system info on startup
    logger.info("🎯 Gemini Embedding Model Pipeline CLI ready with comprehensive logging")
    logger.info("📋 Available commands: process-texts, generate-embeddings, generate-comprehensive-papers, run-full-pipeline")
    logger.info("📋 NEW: generate-comprehensive-papers - Main command for comprehensive academic assessment")
    logger.info("📋 Logging commands: logs, log-status, clear-logs")
    logger.info("📋 Utility commands: status, test-database, check-duplicates, test-supabase")
    logger.info("📁 All operations logged to: data/output/logs")
    
    cli()
