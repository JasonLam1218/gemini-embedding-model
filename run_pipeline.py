#!/usr/bin/env python3

"""
Main pipeline controller for embedding-based exam generation system.
Enhanced with quota-aware generation and robust error handling.
"""

import sys
import os
import click
from loguru import logger
from pathlib import Path
import json
from dotenv import load_dotenv
from datetime import datetime

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
from src.core.storage.vector_store import VectorStore


@click.group()
def cli():
    """Embedding-Based Exam Generation Pipeline CLI with Quota Management"""
    pass


@cli.command()
@click.option('--input-dir', default='data/input/kelvin_papers', help='Input directory')
def process_texts(input_dir):
    """Load and process text files into chunks"""
    try:
        # Ensure output directory exists
        output_dir = Path("data/output/processed")
        output_dir.mkdir(parents=True, exist_ok=True)

        loader = TextLoader()
        documents = loader.process_directory(Path(input_dir))

        # Process documents into chunks
        chunker = TextChunker()
        all_chunks = []

        for doc in documents:
            chunks = chunker.chunk_text(doc.content)
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

        # Save processed chunks
        chunks_path = output_dir / "processed_chunks.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        # Save processed documents
        docs_path = output_dir / "processed_documents.json"
        loader.save_processed_documents(docs_path)

        logger.info(f"✅ Text processing completed. Processed {len(documents)} documents into {len(all_chunks)} chunks")
        logger.info(f"✅ Chunks saved to: {chunks_path}")
        logger.info(f"✅ Documents saved to: {docs_path}")

    except Exception as e:
        logger.error(f"❌ Text processing failed: {e}")
        raise


@cli.command()
@click.option('--batch-size', default=BATCH_SIZE, help='Batch size for embedding generation')
def generate_embeddings(batch_size):
    """Generate embeddings for all processed chunks using Gemini API with quota awareness"""
    try:
        chunks_path = Path("data/output/processed/processed_chunks.json")
        if not chunks_path.exists():
            logger.error("❌ No processed chunks found. Run 'process-texts' first.")
            return

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        logger.info(f"📝 Generating embeddings for {len(chunks)} chunks with quota awareness")

        generator = EmbeddingGenerator()

        # Check quota status before starting
        quota_status = generator.check_quota_status() if hasattr(generator, 'check_quota_status') else None
        if quota_status:
            logger.info(f"📊 API Quota Status: {quota_status['requests_remaining']}/{quota_status['daily_limit']} remaining")

        # Generate embeddings with metadata and quota management
        embeddings_data = []
        for i, chunk in enumerate(chunks):
            try:
                embedding = generator.generate_single_embedding(chunk["chunk_text"])
                if embedding:
                    chunk_with_embedding = {
                        **chunk,
                        "embedding": embedding,
                        "embedding_model": "text-embedding-004"
                    }
                    embeddings_data.append(chunk_with_embedding)
                    logger.info(f"✅ Generated embedding for chunk {chunk['id']} ({i+1}/{len(chunks)})")
                else:
                    logger.warning(f"⚠️ Failed to generate embedding for chunk {chunk['id']}")

            except Exception as e:
                if "quota" in str(e).lower() or "429" in str(e):
                    logger.error(f"❌ API quota exhausted at chunk {i+1}. Generated {len(embeddings_data)} embeddings so far.")
                    break
                else:
                    logger.error(f"❌ Failed to generate embedding for chunk {chunk['id']}: {e}")
                    continue

        # Save embeddings
        output_path = Path("data/output/processed/embeddings.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(embeddings_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Embedding generation completed. Generated {len(embeddings_data)} embeddings")
        logger.info(f"✅ Results saved to: {output_path}")

        if len(embeddings_data) < len(chunks):
            logger.warning(f"⚠️ Only {len(embeddings_data)}/{len(chunks)} embeddings generated. Check API quota.")

    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        raise


@cli.command()
@click.option('--topic', default='AI and Data Analytics', help='Exam topic')
@click.option('--structure-type', default='standard', help='Exam structure type')
@click.option('--formats', default='txt', help='Output formats (comma-separated: txt,md,json)')
@click.option('--quota-aware', is_flag=True, default=True, help='Use quota-aware generation')
@click.option('--template-only', is_flag=True, default=False, help='Use template-only generation (no API calls)')
def generate_structured_exam(topic, structure_type, formats, quota_aware, template_only):
    """Generate structured exam paper with model answers and marking schemes"""
    try:
        # Check if embeddings exist
        embeddings_path = Path("data/output/processed/embeddings.json")
        if not embeddings_path.exists():
            logger.error("❌ No embeddings found. Run 'generate-embeddings' first.")
            return

        logger.info(f"🔄 Generating structured exam paper for topic: {topic}")
        
        if template_only:
            logger.info("📝 Using template-only generation (no API calls)")
        elif quota_aware:
            logger.info("🛡️ Using quota-aware generation to prevent API exhaustion")

        # Parse formats
        format_list = [f.strip() for f in formats.split(',')]
        logger.info(f"📄 Output formats: {', '.join(format_list)}")

        # Initialize structure generator
        structure_gen = StructureGenerator()

        # Generate exam based on mode
        if template_only:
            exam_paper = structure_gen.generate_template_only_exam(topic=topic)
        else:
            exam_paper = structure_gen.generate_structured_exam(
                topic=topic,
                structure_type=structure_type
            )

        # Check if generation was successful
        if exam_paper.get('exam_metadata', {}).get('total_questions', 0) == 0:
            logger.error("❌ No questions were generated. Check your content and API quota.")
            if not template_only:
                logger.info("💡 Try using --template-only flag for fallback generation")
            return

        # Save generated exam in multiple formats
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
            logger.info(f"   {file_type}: {file_path}")

        # Display generation stats
        stats = exam_paper.get('generation_stats', {})
        logger.info(f"📊 Generation Statistics:")
        logger.info(f"   Questions Generated: {stats.get('questions_generated', 0)}")
        logger.info(f"   Total Marks: {stats.get('total_marks', 0)}")
        logger.info(f"   Content Sources Used: {stats.get('content_sources_used', 0)}")
        
        generation_mode = "Template-based" if template_only else "AI-enhanced"
        logger.info(f"   Generation Mode: {generation_mode}")

    except Exception as e:
        logger.error(f"❌ Structured exam generation failed: {e}")
        if "quota" in str(e).lower() or "429" in str(e):
            logger.info("💡 Try using --template-only flag to generate exams without API calls")
        raise


@cli.command()
@click.option('--topic', default='AI and Data Analytics', help='Exam topic')
@click.option('--num-questions', default=10, help='Number of simple questions to generate')
@click.option('--difficulty', default='intermediate', help='Question difficulty level')
def generate_simple_exam(topic, num_questions, difficulty):
    """Generate simple exam using the basic exam generator (for testing/fallback)"""
    try:
        logger.info(f"🔄 Generating simple exam: {topic} ({num_questions} questions, {difficulty} level)")
        
        # Import the basic exam generator
        from src.core.generation.exam_generator import ExamGenerator
        
        exam_gen = ExamGenerator()
        exam = exam_gen.generate_exam(
            topic=topic,
            num_questions=num_questions,
            difficulty=difficulty
        )

        # Save simple exam
        output_dir = Path("data/output/generated_exams")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"simple_exam_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(exam, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Generated simple exam with {len(exam.get('questions', []))} questions")
        logger.info(f"✅ Saved to: {output_file}")

    except Exception as e:
        logger.error(f"❌ Simple exam generation failed: {e}")
        raise


@cli.command()
def run_full_pipeline():
    """Run the complete embedding-based exam generation pipeline with error handling"""
    try:
        click.echo("🚀 Starting full embedding-based exam generation pipeline...")
        ctx = click.get_current_context()

        click.echo("📝 Step 1: Processing text inputs...")
        try:
            ctx.invoke(process_texts)
        except Exception as e:
            logger.error(f"❌ Text processing failed: {e}")
            raise

        click.echo("🧠 Step 2: Generating embeddings using Gemini API...")
        try:
            ctx.invoke(generate_embeddings)
        except Exception as e:
            logger.warning(f"⚠️ Embedding generation had issues: {e}")
            # Continue with available embeddings
            embeddings_path = Path("data/output/processed/embeddings.json")
            if not embeddings_path.exists():
                logger.error("❌ No embeddings generated. Cannot continue.")
                raise

        click.echo("📋 Step 3: Generating structured exam paper...")
        try:
            # Try AI-enhanced generation first
            ctx.invoke(generate_structured_exam, formats='txt,json', quota_aware=True)
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                logger.warning("⚠️ API quota exhausted. Falling back to template-only generation...")
                try:
                    ctx.invoke(generate_structured_exam, formats='txt,json', template_only=True)
                except Exception as fallback_error:
                    logger.error(f"❌ Template fallback also failed: {fallback_error}")
                    raise
            else:
                raise

        click.echo("✅ Pipeline completed successfully!")

    except Exception as e:
        click.echo(f"❌ Pipeline failed: {e}")
        logger.error(f"Full pipeline error: {e}")
        raise


@cli.command()
def status():
    """Show pipeline status and quota information"""
    click.echo("📊 Embedding-Based Pipeline Status Report")
    click.echo("=" * 50)

    # Check processed chunks
    chunks_file = Path("data/output/processed/processed_chunks.json")
    if chunks_file.exists():
        with open(chunks_file) as f:
            chunks = json.load(f)
        click.echo(f"📄 Processed chunks: {len(chunks)}")
    else:
        click.echo("📄 Processed chunks: Not found")

    # Check embeddings
    embeddings_file = Path("data/output/processed/embeddings.json")
    if embeddings_file.exists():
        with open(embeddings_file) as f:
            embeddings = json.load(f)
        click.echo(f"🧠 Generated embeddings: {len(embeddings)}")
        if embeddings:
            click.echo(f"🧠 Embedding dimensions: {len(embeddings[0].get('embedding', []))}")
            click.echo(f"🧠 Embedding model: {embeddings[0].get('embedding_model', 'Unknown')}")
    else:
        click.echo("🧠 Generated embeddings: Not found")

    # Check generated exams
    exams_dir = Path("data/output/generated_exams")
    if exams_dir.exists():
        exam_files = list(exams_dir.glob("*.json"))
        txt_files = list(exams_dir.glob("*.txt"))
        click.echo(f"📋 Generated exam files: {len(exam_files + txt_files)}")
        if exam_files or txt_files:
            click.echo("📋 Recent exam files:")
            all_files = sorted(exam_files + txt_files, key=lambda x: x.stat().st_mtime, reverse=True)
            for file_path in all_files[:5]:  # Show 5 most recent
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                click.echo(f"   - {file_path.name} (created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
    else:
        click.echo("📋 Generated exams: Not found")

    # Check quota status if available
    try:
        from src.core.utils.quota_manager import APIQuotaManager
        quota_manager = APIQuotaManager()
        quota_status = quota_manager.get_quota_status()
        click.echo(f"🔄 API Quota Status:")
        click.echo(f"   Date: {quota_status['date']}")
        click.echo(f"   Requests made: {quota_status['requests_made']}/{quota_status['daily_limit']}")
        click.echo(f"   Requests remaining: {quota_status['requests_remaining']}")
    except ImportError:
        click.echo("🔄 API Quota Status: Not available (quota manager not found)")
    except Exception as e:
        click.echo(f"🔄 API Quota Status: Error reading status ({e})")


@cli.command()
def reset_quota():
    """Reset API quota counter (for testing purposes)"""
    try:
        from src.core.utils.quota_manager import APIQuotaManager
        quota_manager = APIQuotaManager()
        quota_manager.reset_quota_data()
        quota_status = quota_manager.get_quota_status()
        click.echo("✅ API quota counter reset successfully")
        click.echo(f"📊 New status: {quota_status['requests_remaining']}/{quota_status['daily_limit']} requests available")
    except ImportError:
        click.echo("❌ Quota manager not available")
    except Exception as e:
        click.echo(f"❌ Failed to reset quota: {e}")


@cli.command()
@click.option('--test-embedding', is_flag=True, help='Test embedding generation')
@click.option('--test-generation', is_flag=True, help='Test content generation')
def test_api():
    """Test API connections and functionality"""
    click.echo("🧪 Testing API connections...")
    
    try:
        from src.core.embedding.gemini_client import GeminiClient
        client = GeminiClient()
        
        if test_embedding:
            click.echo("📝 Testing embedding generation...")
            test_result = client.test_connection()
            if test_result:
                click.echo("✅ Embedding API test passed")
            else:
                click.echo("❌ Embedding API test failed")
        
        if test_generation:
            click.echo("🤖 Testing content generation...")
            try:
                response = client.generate_content("Test prompt: What is AI?", temperature=0.3, max_tokens=100)
                if response and len(response) > 10:
                    click.echo("✅ Content generation API test passed")
                    click.echo(f"📄 Sample response: {response[:100]}...")
                else:
                    click.echo("❌ Content generation API test failed - empty response")
            except Exception as gen_error:
                click.echo(f"❌ Content generation API test failed: {gen_error}")
        
        if not test_embedding and not test_generation:
            # Test both by default
            embedding_test = client.test_connection()
            click.echo(f"📝 Embedding API: {'✅ Working' if embedding_test else '❌ Failed'}")
            
            try:
                gen_response = client.generate_content("Test", max_tokens=50)
                gen_test = bool(gen_response and len(gen_response) > 5)
                click.echo(f"🤖 Generation API: {'✅ Working' if gen_test else '❌ Failed'}")
            except Exception as e:
                click.echo(f"🤖 Generation API: ❌ Failed ({e})")
        
    except Exception as e:
        click.echo(f"❌ API test failed: {e}")


if __name__ == '__main__':
    cli()
