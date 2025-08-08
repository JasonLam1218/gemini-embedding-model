"""
Content validation utilities for the gemini-embedding-model pipeline.
Validates markdown files, embeddings, and pipeline readiness.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger

class ContentValidator:
    """Validates content pipeline components for debugging and health checks"""
    
    def __init__(self):
        self.base_dir = Path(".")
        
    def validate_markdown_files(self) -> Dict[str, Any]:
        """Validate converted markdown files"""
        markdown_dir = Path("data/output/converted_markdown")
        
        validation_result = {
            "markdown_dir_exists": markdown_dir.exists(),
            "total_files": 0,
            "total_content_chars": 0,
            "kelvin_papers": [],
            "lectures": []
        }
        
        if not markdown_dir.exists():
            logger.warning("⚠️ Markdown directory does not exist")
            return validation_result
            
        # Check kelvin_papers directory
        kelvin_dir = markdown_dir / "kelvin_papers"
        if kelvin_dir.exists():
            kelvin_files = list(kelvin_dir.glob("*.md"))
            validation_result["kelvin_papers"] = [f.name for f in kelvin_files]
            for f in kelvin_files:
                try:
                    content = f.read_text(encoding='utf-8')
                    validation_result["total_content_chars"] += len(content)
                    validation_result["total_files"] += 1
                except Exception as e:
                    logger.error(f"Error reading {f}: {e}")
        
        # Check lectures directory  
        lectures_dir = markdown_dir / "lectures"
        if lectures_dir.exists():
            lecture_files = list(lectures_dir.glob("*.md"))
            validation_result["lectures"] = [f.name for f in lecture_files]
            for f in lecture_files:
                try:
                    content = f.read_text(encoding='utf-8')
                    validation_result["total_content_chars"] += len(content)
                    validation_result["total_files"] += 1
                except Exception as e:
                    logger.error(f"Error reading {f}: {e}")
                    
        return validation_result
    
    def validate_embeddings(self) -> Dict[str, Any]:
        """Validate embeddings file"""
        embeddings_file = Path("data/output/processed/embeddings.json")
        
        validation_result = {
            "file_exists": embeddings_file.exists(),
            "embeddings_count": 0,
            "chunks_with_content": 0,
            "embedding_dimensions": 0,
            "valid_embeddings": 0
        }
        
        if not embeddings_file.exists():
            logger.warning("⚠️ Embeddings file does not exist")
            return validation_result
            
        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
                
            validation_result["embeddings_count"] = len(embeddings_data)
            
            for emb in embeddings_data:
                if emb.get("chunk_text") and emb.get("chunk_text").strip():
                    validation_result["chunks_with_content"] += 1
                    
                if emb.get("embedding"):
                    validation_result["valid_embeddings"] += 1
                    if validation_result["embedding_dimensions"] == 0:
                        validation_result["embedding_dimensions"] = len(emb["embedding"])
                        
        except Exception as e:
            logger.error(f"Error validating embeddings: {e}")
            
        return validation_result
    
    def validate_processed_chunks(self) -> Dict[str, Any]:
        """Validate processed chunks file"""
        chunks_file = Path("data/output/processed/processed_chunks.json")
        
        validation_result = {
            "file_exists": chunks_file.exists(),
            "total_chunks": 0,
            "chunks_with_content": 0,
            "content_types": {},
            "paper_sets": {}
        }
        
        if not chunks_file.exists():
            logger.warning("⚠️ Processed chunks file does not exist")
            return validation_result
            
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
                
            validation_result["total_chunks"] = len(chunks_data)
            
            for chunk in chunks_data:
                if chunk.get("chunk_text") and chunk.get("chunk_text").strip():
                    validation_result["chunks_with_content"] += 1
                    
                # Count content types
                content_type = chunk.get("content_type", "unknown")
                validation_result["content_types"][content_type] = validation_result["content_types"].get(content_type, 0) + 1
                
                # Count paper sets
                paper_set = chunk.get("paper_set", "unknown")
                validation_result["paper_sets"][paper_set] = validation_result["paper_sets"].get(paper_set, 0) + 1
                
        except Exception as e:
            logger.error(f"Error validating chunks: {e}")
            
        return validation_result
    
    def validate_generated_exams(self) -> Dict[str, Any]:
        """Validate generated exam files"""
        exams_dir = Path("data/output/generated_exams")
        
        validation_result = {
            "exams_dir_exists": exams_dir.exists(),
            "total_files": 0,
            "pdf_files": 0,
            "txt_files": 0,
            "md_files": 0,
            "json_files": 0,
            "file_list": []
        }
        
        if not exams_dir.exists():
            logger.warning("⚠️ Generated exams directory does not exist")
            return validation_result
            
        for file_path in exams_dir.iterdir():
            if file_path.is_file():
                validation_result["total_files"] += 1
                validation_result["file_list"].append(file_path.name)
                
                if file_path.suffix.lower() == '.pdf':
                    validation_result["pdf_files"] += 1
                elif file_path.suffix.lower() == '.txt':
                    validation_result["txt_files"] += 1
                elif file_path.suffix.lower() == '.md':
                    validation_result["md_files"] += 1
                elif file_path.suffix.lower() == '.json':
                    validation_result["json_files"] += 1
                    
        return validation_result
    
    @classmethod
    def run_complete_validation(cls) -> Dict[str, Any]:
        """Run complete validation of the entire pipeline"""
        validator = cls()
        
        logger.info("🔍 Starting complete content validation")
        
        # Run all validations
        markdown_validation = validator.validate_markdown_files()
        embeddings_validation = validator.validate_embeddings()
        chunks_validation = validator.validate_processed_chunks()
        exams_validation = validator.validate_generated_exams()
        
        # Determine pipeline readiness
        markdown_files_ok = (markdown_validation["markdown_dir_exists"] and 
                           markdown_validation["total_files"] > 0)
        
        embeddings_ok = (embeddings_validation["file_exists"] and 
                        embeddings_validation["valid_embeddings"] > 0)
        
        chunks_ok = (chunks_validation["file_exists"] and 
                    chunks_validation["chunks_with_content"] > 0)
        
        pipeline_ready = markdown_files_ok and embeddings_ok and chunks_ok
        
        # Create summary
        summary = {
            "markdown_files_ok": markdown_files_ok,
            "embeddings_ok": embeddings_ok,
            "chunks_ok": chunks_ok,
            "pipeline_ready": pipeline_ready,
            "total_markdown_files": markdown_validation["total_files"],
            "total_embeddings": embeddings_validation["embeddings_count"],
            "total_chunks": chunks_validation["total_chunks"],
            "total_exam_files": exams_validation["total_files"]
        }
        
        return {
            "validation_timestamp": str(Path().resolve()),
            "markdown_validation": markdown_validation,
            "embeddings_validation": embeddings_validation,
            "chunks_validation": chunks_validation,
            "exams_validation": exams_validation,
            "summary": summary
        }
    
    def check_file_integrity(self, file_path: Path) -> bool:
        """Check if a JSON file can be loaded without errors"""
        try:
            if not file_path.exists():
                return False
                
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
            return True
            
        except Exception as e:
            logger.error(f"File integrity check failed for {file_path}: {e}")
            return False
    
    def get_pipeline_health_score(self) -> float:
        """Calculate overall pipeline health score (0-1)"""
        try:
            validation = self.run_complete_validation()
            summary = validation["summary"]
            
            # Weight different components
            weights = {
                "markdown_files_ok": 0.3,
                "chunks_ok": 0.3, 
                "embeddings_ok": 0.4
            }
            
            score = sum(weights[key] * (1 if summary[key] else 0) 
                       for key in weights.keys())
            
            return round(score, 2)
            
        except Exception as e:
            logger.error(f"Health score calculation failed: {e}")
            return 0.0
