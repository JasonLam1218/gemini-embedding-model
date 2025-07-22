#!/usr/bin/env python3
"""
Main pipeline controller for exam generation system.
"""

import sys
import os
import click
from loguru import logger
from pathlib import Path
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import project modules
from config.settings import *
from src.core.text.text_loader import TextLoader
from src.core.text.chunker import TextChunker
from src.core.embedding.embedding_generator import EmbeddingGenerator
from src.core.storage.vector_store import VectorStore, Document, TextChunk, Embedding

@click.group()
def cli():
    """Exam Generation Pipeline CLI"""
    pass

@cli.command()
@click.option('--input-dir', default='data/input/kelvin_papers', help='Input directory')
@click.option('--output-dir', default='data/output/processed', help='Output directory')
def process_texts(input_dir, output_dir):
    """Load and process text files"""
    try:
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        loader = TextLoader()
        documents = loader.process_directory(Path(input_dir))
        
        # Save processed documents
        output_path = Path(output_dir) / "processed_documents.json"
        loader.save_processed_documents(output_path)
        
        logger.info(f"✅ Text processing completed. Processed {len(documents)} documents")
        logger.info(f"✅ Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Text processing failed: {e}")
        raise

@cli.command()
@click.option('--input-file', default='data/output/processed/processed_documents.json', help='Input processed documents file')
@click.option('--output-file', default='data/output/processed/embeddings.json', help='Output embeddings file')
@click.option('--batch-size', default=BATCH_SIZE, help='Batch size for processing')
def generate_embeddings(input_file, output_file, batch_size):
    """Generate embeddings for all processed texts"""
    try:
        # Load processed documents
        with open(input_file, "r", encoding="utf-8") as f:
            docs = json.load(f)
        
        logger.info(f"📝 Loading {len(docs)} processed documents")
        
        # Initialize components
        chunker = TextChunker()
        generator = EmbeddingGenerator()
        
        all_embeddings = []
        
        # Process each document
        for i, doc in enumerate(docs, 1):
            logger.info(f"🔄 Processing document {i}/{len(docs)}: {doc.get('title', 'Untitled')}")
            
            # Chunk text
            chunks = chunker.chunk_text(doc["content"])
            logger.info(f"  📄 Created {len(chunks)} chunks")
            
            # Generate embeddings in batches
            embeddings = generator.process_chunks(chunks)
            
            # Store results
            for chunk_idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                all_embeddings.append({
                    "id": f"{doc['paper_set']}_{doc['paper_number']}_{chunk_idx}",
                    "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                    "chunk_text": chunk,
                    "chunk_index": chunk_idx,
                    "chunk_size": len(chunk),
                    "document_metadata": doc["metadata"],
                    "source_file": doc["source_file"],
                    "paper_set": doc["paper_set"],
                    "paper_number": doc["paper_number"]
                })
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_embeddings, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Embedding generation completed. Generated {len(all_embeddings)} embeddings")
        logger.info(f"✅ Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        raise

@cli.command()
@click.option('--topic', help='Specific topic for exam generation')
@click.option('--num-questions', default=10, help='Number of questions')
@click.option('--difficulty', default=DEFAULT_DIFFICULTY, help='Question difficulty level')
@click.option('--question-types', default='multiple_choice,short_answer', help='Comma-separated question types')
def generate_exam(topic, num_questions, difficulty, question_types):
    """Generate exam questions"""
    try:
        from src.core.generation.exam_generator import ExamGenerator
        
        # Parse question types
        types_list = [qt.strip() for qt in question_types.split(',')]
        
        exam_gen = ExamGenerator()
        exam = exam_gen.generate_exam(
            topic=topic, 
            num_questions=num_questions,
            difficulty=difficulty,
            question_types=types_list
        )
        
        # Save generated exam
        output_dir = Path("data/output/generated_exams")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"exam_{timestamp}.json"
        
        exam_data = {
            "title": f"Generated Exam - {topic or 'General'}",
            "topic": topic,
            "difficulty": difficulty,
            "num_questions": num_questions,
            "question_types": types_list,
            "generated_at": timestamp,
            "questions": exam.questions if hasattr(exam, 'questions') else []
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(exam_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Generated exam with {num_questions} questions")
        logger.info(f"✅ Exam saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Exam generation failed: {e}")
        raise

@cli.command()
@click.option('--skip-processing', is_flag=True, help='Skip text processing if already done')
@click.option('--skip-embeddings', is_flag=True, help='Skip embedding generation if already done')
def run_full_pipeline(skip_processing, skip_embeddings):
    """Run the complete pipeline"""
    try:
        click.echo("🚀 Starting full exam generation pipeline...")
        ctx = click.get_current_context()
        
        if not skip_processing:
            click.echo("📝 Processing text inputs...")
            ctx.invoke(process_texts)
        else:
            click.echo("⏭️  Skipping text processing")
        
        if not skip_embeddings:
            click.echo("🧠 Generating embeddings...")
            ctx.invoke(generate_embeddings)
        else:
            click.echo("⏭️  Skipping embedding generation")
        
        click.echo("📋 Generating sample exam...")
        ctx.invoke(generate_exam, topic="machine learning", num_questions=5)
        
        click.echo("✅ Pipeline completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Pipeline failed: {e}")
        raise

@cli.command()
def status():
    """Show pipeline status and file information"""
    click.echo("📊 Pipeline Status Report")
    click.echo("=" * 50)
    
    # Check processed documents
    processed_file = Path("data/output/processed/processed_documents.json")
    if processed_file.exists():
        with open(processed_file) as f:
            docs = json.load(f)
        click.echo(f"📄 Processed documents: {len(docs)}")
    else:
        click.echo("📄 Processed documents: Not found")
    
    # Check embeddings
    embeddings_file = Path("data/output/processed/embeddings.json")
    if embeddings_file.exists():
        with open(embeddings_file) as f:
            embeddings = json.load(f)
        click.echo(f"🧠 Generated embeddings: {len(embeddings)}")
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
