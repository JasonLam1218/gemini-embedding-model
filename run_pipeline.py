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
from config.settings import BATCH_SIZE
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
    try:
        # Ensure output directory exists
        output_dir = Path("data/output/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        loader = TextLoader()
        documents = loader.process_directory(Path(input_dir))
        
        output_path = output_dir / "processed_documents.json"
        loader.save_processed_documents(output_path)
        
        logger.info(f"✅ Text processing completed. Processed {len(documents)} documents")
        logger.info(f"✅ Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Text processing failed: {e}")
        raise

@cli.command()
def generate_embeddings():
    """Generate embeddings for all processed texts with improved error handling"""
    try:
        processed_path = Path("data/output/processed/processed_documents.json")
        
        if not processed_path.exists():
            logger.error("❌ No processed documents found. Run 'process-texts' first.")
            return
        
        with open(processed_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        
        logger.info(f"📝 Loading {len(docs)} processed documents")
        
        chunker = TextChunker()
        generator = EmbeddingGenerator()
        all_embeddings = []
        
        for i, doc in enumerate(docs, 1):
            logger.info(f"🔄 Processing document {i}/{len(docs)}: {doc.get('title', 'Untitled')}")
            
            # Validate document content
            content = doc.get("content", "")
            if not content or len(content.strip()) < 50:
                logger.warning(f"⚠️ Skipping document {i}: insufficient content")
                continue
            
            chunks = chunker.chunk_text(content)
            logger.info(f"  📄 Created {len(chunks)} chunks")
            
            if not chunks:
                logger.warning(f"⚠️ No chunks created for document {i}")
                continue
            
            try:
                embeddings = generator.process_chunks(chunks)
                logger.info(f"  🧠 Generated {len(embeddings)} embeddings")
                
                # Only process successful embeddings
                for chunk_idx, (chunk, emb) in enumerate(zip(chunks[:len(embeddings)], embeddings)):
                    if emb is not None and len(emb) > 0:  # Valid embedding
                        all_embeddings.append({
                            "id": f"{doc['paper_set']}_{doc['paper_number']}_{chunk_idx}",
                            "embedding": emb,
                            "chunk": chunk,
                            "chunk_text": chunk,  # For compatibility
                            "chunk_index": chunk_idx,
                            "chunk_size": len(chunk),
                            "metadata": doc.get("metadata", {}),
                            "source_file": doc["source_file"],
                            "paper_set": doc["paper_set"],
                            "paper_number": doc["paper_number"]
                        })
                
            except Exception as e:
                logger.error(f"❌ Failed to process document {i}: {e}")
                continue  # Continue with next document
        
        if not all_embeddings:
            logger.error("❌ No embeddings generated successfully")
            return
        
        # Save embeddings
        output_path = Path("data/output/processed/embeddings.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_embeddings, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Embedding generation completed. Generated {len(all_embeddings)} valid embeddings")
        logger.info(f"✅ Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        raise

@cli.command()
@click.option('--topic', help='Specific topic for exam generation')
@click.option('--num-questions', default=10, help='Number of questions')
def generate_exam(topic, num_questions):
    """Generate exam questions"""
    try:
        from src.core.generation.exam_generator import ExamGenerator
        
        exam_gen = ExamGenerator()
        exam = exam_gen.generate_exam(topic=topic, num_questions=num_questions)
        
        # Save generated exam
        output_dir = Path("data/output/generated_exams")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"exam_{timestamp}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(exam, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Generated exam with {num_questions} questions")
        logger.info(f"✅ Exam saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Exam generation failed: {e}")
        raise

@cli.command()
def run_full_pipeline():
    """Run the complete pipeline"""
    try:
        click.echo("🚀 Starting full exam generation pipeline...")
        ctx = click.get_current_context()
        
        click.echo("📝 Processing text inputs...")
        ctx.invoke(process_texts)
        
        click.echo("🧠 Generating embeddings...")
        ctx.invoke(generate_embeddings)
        
        click.echo("📋 Generating sample exam...")
        ctx.invoke(generate_exam, num_questions=5)
        
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
