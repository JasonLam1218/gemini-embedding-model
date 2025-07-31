"""Content classification for different types of academic materials"""

from pathlib import Path
from typing import Dict, Any

def classify_content_type(file_path: Path) -> str:
    """Classify if content is exam paper, model answers, or lecture notes"""
    file_name = file_path.name.lower()
    parent_dir = file_path.parent.name.lower()
    
    if "kelvin_papers" in parent_dir:
        if "ms" in file_name or "model" in file_name:
            return "model_answers"
        elif "exam" in file_name or "paper" in file_name:
            return "exam_questions"
        else:
            return "sample_paper"
    elif "lectures" in parent_dir:
        return "lecture_notes"
    else:
        return "unknown"
