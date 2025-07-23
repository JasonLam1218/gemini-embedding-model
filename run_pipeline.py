#!/usr/bin/env python3

"""
Main pipeline controller for embedding-based exam generation system.
"""

import sys
import os
import click
from loguru import logger
from pathlib import Path
import json
from dotenv import load_dotenv

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
    """Embedding-Based Exam Generation Pipeline CLI"""
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
def generate_embeddings():
    """Generate embeddings for all processed chunks using Gemini API"""
    try:
        chunks_path = Path("data/output/processed/processed_chunks.json")
        if not chunks_path.exists():
            logger.error("❌ No processed chunks found. Run 'process-texts' first.")
            return

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        logger.info(f"📝 Generating embeddings for {len(chunks)} chunks")

        generator = EmbeddingGenerator()

        # Generate embeddings with metadata
        embeddings_data = []
        for chunk in chunks:
            try:
                embedding = generator.generate_single_embedding(chunk["chunk_text"])
                if embedding:
                    chunk_with_embedding = {
                        **chunk,
                        "embedding": embedding,
                        "embedding_model": "text-embedding-004"
                    }
                    embeddings_data.append(chunk_with_embedding)
                    logger.info(f"✅ Generated embedding for chunk {chunk['id']}")
            except Exception as e:
                logger.error(f"❌ Failed to generate embedding for chunk {chunk['id']}: {e}")
                continue

        # Save embeddings
        output_path = Path("data/output/processed/embeddings.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(embeddings_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Embedding generation completed. Generated {len(embeddings_data)} embeddings")
        logger.info(f"✅ Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        raise

@cli.command()
@click.option('--topic', default='AI and Data Analytics', help='Exam topic')
@click.option('--structure-type', default='standard', help='Exam structure type')
@click.option('--formats', default='txt,md,pdf', help='Output formats (comma-separated: txt,md,pdf)')
def generate_structured_exam(topic, structure_type, formats):
    """Generate structured exam paper using embedding similarity"""
    try:
        # Check if embeddings exist
        embeddings_path = Path("data/output/processed/embeddings.json")
        if not embeddings_path.exists():
            logger.error("❌ No embeddings found. Run 'generate-embeddings' first.")
            return

        logger.info(f"🔄 Generating structured exam paper for topic: {topic}")
        
        # Parse output formats
        output_formats = [fmt.strip().lower() for fmt in formats.split(',')]
        valid_formats = ['txt', 'md', 'pdf']
        output_formats = [fmt for fmt in output_formats if fmt in valid_formats]
        
        if not output_formats:
            output_formats = ['txt']  # Default fallback
            
        logger.info(f"📄 Output formats: {', '.join(output_formats)}")

        # Initialize structure generator
        structure_gen = StructureGenerator()

        # Generate exam with 4 main questions and sub-parts
        exam_paper = structure_gen.generate_structured_exam(
            topic=topic,
            structure_type=structure_type,
            num_main_questions=4
        )

        # Save generated exam
        output_dir = Path("data/output/generated_exams")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"structured_exam_{timestamp}"
        base_path = output_dir / base_filename

        # Save in multiple formats
        saved_files = structure_gen.save_multi_format_exam(
            exam_paper, base_path, output_formats
        )

        logger.info(f"✅ Generated structured exam paper")
        
        # Log all saved files
        for format_type, file_path in saved_files.items():
            logger.info(f"✅ {format_type.upper()} saved to: {file_path}")

        # Also save JSON version for data persistence
        json_path = output_dir / f"{base_filename}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(exam_paper, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ JSON data saved to: {json_path}")

    except Exception as e:
        logger.error(f"❌ Structured exam generation failed: {e}")
        raise

@cli.command()
def run_full_pipeline():
    """Run the complete embedding-based exam generation pipeline"""
    try:
        click.echo("🚀 Starting full embedding-based exam generation pipeline...")
        ctx = click.get_current_context()

        click.echo("📝 Step 1: Processing text inputs...")
        ctx.invoke(process_texts)

        click.echo("🧠 Step 2: Generating embeddings using Gemini API...")
        ctx.invoke(generate_embeddings)

        click.echo("📋 Step 3: Generating structured exam paper in multiple formats...")
        ctx.invoke(generate_structured_exam, formats='txt,md,pdf')  # Generate all formats

        click.echo("✅ Pipeline completed successfully!")

    except Exception as e:
        click.echo(f"❌ Pipeline failed: {e}")
        raise

@cli.command()
def status():
    """Show pipeline status and embedding information"""
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
    else:
        click.echo("🧠 Generated embeddings: Not found")

    # Check generated exams
    exams_dir = Path("data/output/generated_exams")
    if exams_dir.exists():
        exam_files = list(exams_dir.glob("*.json"))
        click.echo(f"📋 Generated exams: {len(exam_files)}")
    else:
        click.echo("📋 Generated exams: Not found")

if __name__ == '__main__':
    cli()
