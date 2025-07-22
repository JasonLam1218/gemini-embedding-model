#!/usr/bin/env python3
"""
Main pipeline controller for exam generation system.
"""

import click
from loguru import logger
from pathlib import Path
import json
from src.core.text.text_loader import TextLoader
from src.core.text.chunker import TextChunker
from src.core.embedding.embedding_generator import EmbeddingGenerator

@click.group()
def cli():
    """Exam Generation Pipeline CLI"""
    pass

@cli.command()
@click.option('--input-dir', default='data/input/kelvin_papers', help='Input directory')
def process_texts(input_dir):
    """Load and process text files"""
    loader = TextLoader()
    loader.process_directory(Path(input_dir))
    output_path = Path("data/output/processed/processed_documents.json")
    loader.save_processed_documents(output_path)
    logger.info("✅ Text processing completed")

@cli.command()
def generate_embeddings():
    """Generate embeddings for all processed texts"""
    processed_path = Path("data/output/processed/processed_documents.json")
    with open(processed_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    chunker = TextChunker()
    generator = EmbeddingGenerator()

    all_embeddings = []
    for doc in docs:
        chunks = chunker.chunk_text(doc["content"])
        embeddings = generator.process_chunks(chunks)
        for chunk, emb in zip(chunks, embeddings):
            all_embeddings.append({
                "embedding": emb,
                "chunk": chunk,
                "metadata": doc["metadata"],
                "source_file": doc["source_file"],
                "paper_set": doc["paper_set"],
                "paper_number": doc["paper_number"]
            })
    output_path = Path("data/output/processed/embeddings.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_embeddings, f, indent=2, ensure_ascii=False)
    logger.info("✅ Embedding generation completed")

@cli.command()
@click.option('--topic', help='Specific topic for exam generation')
@click.option('--num-questions', default=10, help='Number of questions')
def generate_exam(topic, num_questions):
    """Generate exam questions"""
    from src.core.generation.exam_generator import ExamGenerator
    exam_gen = ExamGenerator()
    exam = exam_gen.generate_exam(topic=topic, num_questions=num_questions)
    logger.info(f"✅ Generated exam with {len(exam.questions)} questions")

@cli.command()
def run_full_pipeline():
    """Run the complete pipeline"""
    click.echo("🚀 Starting full exam generation pipeline...")
    ctx = click.get_current_context()
    click.echo("📝 Processing text inputs...")
    ctx.invoke(process_texts)
    click.echo("🧠 Generating embeddings...")
    ctx.invoke(generate_embeddings)
    click.echo("📋 Generating sample exam...")
    ctx.invoke(generate_exam, num_questions=5)
    click.echo("✅ Pipeline completed successfully!")

if __name__ == '__main__':
    cli() 