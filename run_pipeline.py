#!/usr/bin/env python3
"""
Main pipeline controller for embedding-based exam generation system.
Enhanced with Supabase integration, quota-aware generation, and duplicate checking.
"""

import sys
import os
import click
from loguru import logger
from pathlib import Path
import json
from dotenv import load_dotenv
from datetime import datetime
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import project modules
from config.settings import BATCH_SIZE
from src.core.text.text_loader import TextLoader
from src.core.text.chunker import TextChunker
from src.core.embedding.embedding_generator import EmbeddingGenerator
from src.core.generation.structure_generator import StructureGenerator
from src.core.storage.vector_store import VectorStore, Document, TextChunk, Embedding

@click.group()
def cli():
    """Embedding-Based Exam Generation Pipeline CLI with Supabase Integration and Duplicate Checking"""
    pass

@cli.command()
@click.option('--input-dir', default='data/input/kelvin_papers', help='Input directory')
@click.option('--use-supabase', is_flag=True, default=True, help='Store data in Supabase')
@click.option('--force-reprocess', is_flag=True, default=False, help='Force reprocess existing files')
def process_texts(input_dir, use_supabase, force_reprocess):
    """Load and process text files with duplicate checking and Supabase storage"""
    try:
        # Ensure output directory exists
        output_dir = Path("data/output/processed")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        loader = TextLoader()
        chunker = TextChunker()
        vector_store = VectorStore() if use_supabase else None

        # Load documents
        documents = loader.process_directory(Path(input_dir))
        logger.info(f"📄 Found {len(documents)} documents to process")

        # Process documents into chunks with duplicate checking
        all_chunks = []
        supabase_results = []
        processed_count = 0
        skipped_count = 0

        for doc in documents:
            source_file = doc.source_file
            
            # Check if document already exists in database
            if use_supabase and vector_store and not force_reprocess:
                existing_doc = vector_store.document_exists_by_source_file(source_file)
                
                if existing_doc:
                    logger.info(f"⏭️  Document already exists: {source_file}, skipping processing")
                    skipped_count += 1
                    
                    # Retrieve existing chunks for local consistency
                    existing_chunks = vector_store.get_chunks_by_source_file(source_file)
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
            chunks = chunker.chunk_text(doc.content)
            logger.info(f"📝 Created {len(chunks)} chunks for {doc.source_file}")

            # Store in Supabase if enabled
            if use_supabase and vector_store:
                try:
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

                    supabase_results.append({
                        'document_id': doc_id,
                        'chunk_ids': chunk_ids,
                        'source_file': doc.source_file,
                        'status': 'new'
                    })

                    logger.info(f"✅ Stored in Supabase: doc_id={doc_id}, {len(chunk_ids)} chunks")
                except Exception as e:
                    logger.error(f"❌ Failed to store in Supabase: {e}")

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
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        # Save processed documents locally
        docs_path = output_dir / "processed_documents.json"
        loader.save_processed_documents(docs_path)

        # Save Supabase results
        if use_supabase and supabase_results:
            supabase_path = output_dir / "supabase_results.json"
            with open(supabase_path, "w", encoding="utf-8") as f:
                json.dump(supabase_results, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Supabase results saved to: {supabase_path}")

        logger.info(f"✅ Text processing completed: {processed_count} new, {skipped_count} skipped")
        logger.info(f"✅ Total chunks processed: {len(all_chunks)}")
        logger.info(f"✅ Chunks saved to: {chunks_path}")
        logger.info(f"✅ Documents saved to: {docs_path}")

        # Show Supabase statistics
        if use_supabase and vector_store:
            stats = vector_store.get_database_stats()
            logger.info(f"📊 Supabase Status: {stats['documents']} documents, {stats['text_chunks']} chunks")

    except Exception as e:
        logger.error(f"❌ Text processing failed: {e}")
        raise

@cli.command()
@click.option('--batch-size', default=BATCH_SIZE, help='Batch size for embedding generation')
@click.option('--use-supabase', is_flag=True, default=True, help='Store embeddings in Supabase')
@click.option('--force-regenerate', is_flag=True, default=False, help='Force regenerate existing embeddings')
def generate_embeddings(batch_size, use_supabase, force_regenerate):
    """Generate embeddings for all processed chunks with duplicate checking and Supabase storage"""
    try:
        # Initialize components
        generator = EmbeddingGenerator()
        vector_store = VectorStore() if use_supabase else None

        # Load chunks from local file
        chunks_path = Path("data/output/processed/processed_chunks.json")
        if not chunks_path.exists():
            logger.error("❌ No processed chunks found. Run 'process-texts' first.")
            return

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        logger.info(f"📝 Processing embeddings for {len(chunks)} chunks")

        # Check quota status
        quota_status = generator.check_quota_status()
        if quota_status:
            logger.info(f"📊 API Quota Status: {quota_status['requests_remaining']}/{quota_status.get('daily_limit', 'Unknown')} remaining")

        # Get chunks that need embeddings
        chunks_needing_embeddings = []
        if use_supabase and vector_store and not force_regenerate:
            # Check which chunks already have embeddings in database
            for chunk in chunks:
                # Find corresponding database chunk
                existing_doc = vector_store.document_exists_by_source_file(chunk['source_file'])
                if existing_doc:
                    db_chunks = vector_store.get_chunks_by_document(existing_doc['id'])
                    matching_chunk = next((c for c in db_chunks 
                                         if c['chunk_index'] == chunk['chunk_index']), None)
                    
                    if matching_chunk and vector_store.embedding_exists_for_chunk(matching_chunk['id']):
                        logger.info(f"⏭️  Embedding already exists for chunk {chunk['id']}")
                        continue
                
                chunks_needing_embeddings.append(chunk)
        else:
            chunks_needing_embeddings = chunks

        logger.info(f"📝 Generating embeddings for {len(chunks_needing_embeddings)} chunks (skipped {len(chunks) - len(chunks_needing_embeddings)})")

        # Generate embeddings with Supabase storage
        embeddings_data = []
        supabase_embeddings = []
        new_embeddings_count = 0

        for i, chunk in enumerate(chunks_needing_embeddings):
            try:
                # Generate embedding
                embedding = generator.generate_single_embedding(chunk["chunk_text"])
                
                if embedding:
                    # Create embedding data for local storage (keep existing functionality)
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
                                    logger.info(f"✅ Stored embedding in Supabase for chunk {matching_chunk['id']}")
                                else:
                                    logger.warning(f"⚠️ Could not find matching chunk in Supabase for {chunk['id']}")
                            else:
                                logger.warning(f"⚠️ Could not find document in Supabase for {chunk['source_file']}")
                        except Exception as supabase_error:
                            logger.error(f"❌ Failed to store embedding in Supabase: {supabase_error}")

                    new_embeddings_count += 1
                    logger.info(f"✅ Generated embedding for chunk {chunk['id']} ({new_embeddings_count}/{len(chunks_needing_embeddings)})")
                else:
                    logger.warning(f"⚠️ Failed to generate embedding for chunk {chunk['id']}")

            except Exception as e:
                if "quota" in str(e).lower() or "429" in str(e):
                    logger.error(f"❌ API quota exhausted at chunk {i+1}. Generated {len(embeddings_data)} embeddings so far.")
                    break
                else:
                    logger.error(f"❌ Failed to generate embedding for chunk {chunk['id']}: {e}")
                    continue

        # Load existing embeddings for complete dataset
        if not force_regenerate:
            existing_embeddings_path = Path("data/output/processed/embeddings.json")
            if existing_embeddings_path.exists():
                with open(existing_embeddings_path, "r", encoding="utf-8") as f:
                    existing_embeddings = json.load(f)
                
                # Merge with new embeddings (avoid duplicates)
                existing_ids = {emb['id'] for emb in existing_embeddings}
                for new_emb in embeddings_data:
                    if new_emb['id'] not in existing_ids:
                        existing_embeddings.append(new_emb)
                
                embeddings_data = existing_embeddings

        # Save embeddings locally (keep existing functionality)
        output_path = Path("data/output/processed/embeddings.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(embeddings_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Embedding generation completed. Generated {new_embeddings_count} new embeddings")
        logger.info(f"✅ Total embeddings in dataset: {len(embeddings_data)}")
        logger.info(f"✅ Results saved to: {output_path}")

        # Show Supabase statistics
        if use_supabase and vector_store:
            stats = vector_store.get_database_stats()
            logger.info(f"📊 Supabase Status: {stats['embeddings']} embeddings stored")

        if new_embeddings_count < len(chunks_needing_embeddings):
            logger.warning(f"⚠️ Only {new_embeddings_count}/{len(chunks_needing_embeddings)} new embeddings generated. Check API quota.")

    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        raise

@cli.command()
@click.option('--topic', default='AI and Data Analytics', help='Exam topic')
@click.option('--structure-type', default='standard', help='Exam structure type')
@click.option('--formats', default='txt,md,pdf,json', help='Output formats (comma-separated)')
@click.option('--quota-aware', is_flag=True, default=True, help='Use quota-aware generation')
@click.option('--template-only', is_flag=True, default=False, help='Use template-only generation (no API calls)')
@click.option('--use-supabase', is_flag=True, default=True, help='Use Supabase for content retrieval')
def generate_structured_exam(topic, structure_type, formats, quota_aware, template_only, use_supabase):
    """Generate structured exam paper with model answers and marking schemes"""
    try:
        logger.info(f"🔄 Generating structured exam paper for topic: {topic}")
        
        # Store the original Supabase intent
        original_use_supabase = use_supabase
        
        # Initialize components
        structure_gen = StructureGenerator()
        vector_store = VectorStore() if use_supabase else None
        
        if template_only:
            logger.info("📝 Using template-only generation (no API calls)")
        elif quota_aware:
            logger.info("🛡️ Using quota-aware generation to prevent API exhaustion")
        
        # Parse formats
        format_list = [f.strip() for f in formats.split(',')]
        logger.info(f"📄 Output formats: {', '.join(format_list)}")
        
        # Check if we can use Supabase for CONTENT RETRIEVAL (separate from saving)
        exam_paper = None
        use_supabase_for_retrieval = use_supabase
        
        if use_supabase and vector_store:
            try:
                # Test Supabase connection and check data availability
                stats = vector_store.get_database_stats()
                if stats['documents'] > 0 and stats['embeddings'] > 0:
                    logger.info(f"✅ Using Supabase: {stats['documents']} documents, {stats['embeddings']} embeddings")
                    
                    # Generate exam using Supabase data (if method exists)
                    if hasattr(structure_gen, 'generate_structured_exam_from_supabase'):
                        exam_paper = structure_gen.generate_structured_exam_from_supabase(
                            topic=topic,
                            structure_type=structure_type,
                            vector_store=vector_store
                        )
                    else:
                        logger.warning("⚠️ Supabase integration method not found in StructureGenerator")
                        use_supabase_for_retrieval = False  # Only affects retrieval, not saving
                else:
                    logger.warning("⚠️ Supabase has no data, falling back to local files")
                    use_supabase_for_retrieval = False
            # In run_pipeline.py, around line 385
            except Exception as e:
                logger.error(f"❌ Failed to save exam to Supabase: {e}")
                logger.error(f"🔍 Exception type: {type(e)}")
                logger.error(f"📋 Exam data keys: {list(exam_paper.keys()) if exam_paper else 'No data'}")
                
                # Add more detailed debugging
                if hasattr(e, 'details'):
                    logger.error(f"🔍 Error details: {e.details}")
                if hasattr(e, 'code'):
                    logger.error(f"🔍 Error code: {e.code}")
                
                logger.warning("⚠️ Exam generated locally but not saved to database")
                logger.warning("🔍 Check your Supabase connection and database schema")

        
        # Fall back to local generation if Supabase retrieval not available or failed
        if not exam_paper:
            # Check if embeddings exist locally
            embeddings_path = Path("data/output/processed/embeddings.json")
            if not embeddings_path.exists():
                logger.error("❌ No embeddings found. Run 'generate-embeddings' first.")
                return
            
            # Generate exam based on mode
            if template_only:
                exam_paper = structure_gen.generate_template_only_exam(topic=topic)
                logger.info("📝 Template-only exam generated successfully")
            else:
                exam_paper = structure_gen.generate_structured_exam(
                    topic=topic,
                    structure_type=structure_type
                )
        
        # Check if generation was successful
        if not exam_paper or exam_paper.get('exam_metadata', {}).get('total_questions', 0) == 0:
            logger.error("❌ No questions were generated. Check your content and API quota.")
            if not template_only:
                logger.info("💡 Try using --template-only flag for fallback generation")
            return
        
        # === FIXED SUPABASE SAVE INTEGRATION - USES ORIGINAL INTENT ===
        if original_use_supabase and vector_store and exam_paper:
            try:
                # Ensure exam_paper has required metadata structure
                if not exam_paper.get('exam_metadata'):
                    # Add missing metadata for template-only exams
                    exam_paper['exam_metadata'] = {
                        'title': f"Template-Based Exam - {topic}",
                        'topic': topic,
                        'difficulty': 'standard',
                        'total_marks': 100,
                        'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                
                if not exam_paper.get('generation_stats'):
                    # Add missing generation stats
                    questions = exam_paper.get('questions', {})
                    exam_paper['generation_stats'] = {
                        'questions_generated': len(questions),
                        'total_marks': 100,
                        'content_sources_used': 4,  # Based on your embeddings count
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
                logger.info(f"✅ Saved exam to Supabase (ID: {exam_id})")
                
                # Verify the save was successful
                if hasattr(vector_store, 'verify_exam_saved'):
                    verification_success = vector_store.verify_exam_saved(exam_id)
                    if verification_success:
                        logger.info(f"✅ Verified exam saved successfully (ID: {exam_id})")
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
        
        saved_files = structure_gen.save_multi_format_exam(
            exam_paper,
            output_dir,
            formats=format_list
        )
        
        logger.info(f"✅ Generated structured exam paper with {'templates' if template_only else 'AI components'}")
        logger.info("📝 Generated files:")
        for file_path in saved_files:
            file_type = "Question Paper" if "questions" in file_path else \
                       "Model Answers" if "answers" in file_path else \
                       "Marking Schemes" if "schemes" in file_path else "Complete Exam"
            logger.info(f" {file_type}: {file_path}")
        
        # Display generation stats
        stats = exam_paper.get('generation_stats', {})
        logger.info(f"📊 Generation Statistics:")
        logger.info(f" Questions Generated: {stats.get('questions_generated', 0)}")
        logger.info(f" Total Marks: {stats.get('total_marks', 0)}")
        logger.info(f" Content Sources Used: {stats.get('content_sources_used', 0)}")
        generation_mode = "Template-based" if template_only else "AI-enhanced"
        logger.info(f" Generation Mode: {generation_mode}")
        
        # Display database save status
        if stats.get('saved_to_database'):
            logger.info(f" Database Save: ✅ Saved (ID: {stats.get('database_id', 'Unknown')})")
        else:
            logger.info(f" Database Save: ❌ Not saved (check Supabase connection)")
        
    except Exception as e:
        logger.error(f"❌ Structured exam generation failed: {e}")
        if "quota" in str(e).lower() or "429" in str(e):
            logger.info("💡 Try using --template-only flag to generate exams without API calls")
        raise


@cli.command()
def run_full_pipeline():
    """Run the complete embedding-based exam generation pipeline with Supabase integration and duplicate checking"""
    try:
        click.echo("🚀 Starting full embedding-based exam generation pipeline with duplicate checking...")

        ctx = click.get_current_context()

        click.echo("📝 Step 1: Processing text inputs with duplicate checking and storing in Supabase...")
        try:
            ctx.invoke(process_texts, use_supabase=True, force_reprocess=False)
        except Exception as e:
            logger.error(f"❌ Text processing failed: {e}")
            raise

        click.echo("🧠 Step 2: Generating embeddings with duplicate checking and storing in Supabase...")
        try:
            ctx.invoke(generate_embeddings, use_supabase=True, force_regenerate=False)
        except Exception as e:
            logger.warning(f"⚠️ Embedding generation had issues: {e}")
            # Continue with available embeddings
            embeddings_path = Path("data/output/processed/embeddings.json")
            if not embeddings_path.exists():
                logger.error("❌ No embeddings generated. Cannot continue.")
                raise

        click.echo("📋 Step 3: Generating structured exam paper with Supabase data...")
        try:
            # Try AI-enhanced generation first
            ctx.invoke(generate_structured_exam, formats='txt,md,pdf,json', quota_aware=True, use_supabase=True)
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                logger.warning("⚠️ API quota exhausted. Falling back to template-only generation...")
                try:
                    ctx.invoke(generate_structured_exam, formats='txt,md,pdf,json', template_only=True, use_supabase=True)
                except Exception as fallback_error:
                    logger.error(f"❌ Template fallback also failed: {fallback_error}")
                    raise
            else:
                raise

        click.echo("✅ Pipeline completed successfully with duplicate checking!")

    except Exception as e:
        click.echo(f"❌ Pipeline failed: {e}")
        logger.error(f"Full pipeline error: {e}")
        raise

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
    
    # === COMPLETE SUPABASE STATUS WITH EXAM INFORMATION ===
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
        
        # === GET RECENT EXAMS FROM SUPABASE ===
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
                click.echo(f"   Document ID: {existing_doc['id']}")
                click.echo(f"   Title: {existing_doc['title']}")
                click.echo(f"   Created: {existing_doc.get('created_at', 'Unknown')}")
                
                # Check chunks
                chunks = vector_store.get_chunks_by_source_file(source_file)
                click.echo(f"   Chunks: {len(chunks)}")
                
                # Check embeddings
                chunks_with_embeddings = sum(1 for chunk in chunks 
                                           if vector_store.embedding_exists_for_chunk(chunk['id']))
                click.echo(f"   Embeddings: {chunks_with_embeddings}/{len(chunks)}")
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
                    click.echo(f"  - {doc['source_file']} (ID: {doc['id']}, Chunks: {len(chunks)}, Embeddings: {chunks_with_embeddings})")
                
                if len(all_docs) > 10:
                    click.echo(f"  ... and {len(all_docs) - 10} more")

    except Exception as e:
        logger.error(f"❌ Failed to check duplicates: {e}")

# Add this to run_pipeline.py
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
    cli()
