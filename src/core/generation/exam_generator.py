from src.core.generation.rag_pipeline import RAGPipeline
from loguru import logger
from typing import List

class Exam:
    def __init__(self, questions: List[str]):
        self.questions = questions

class ExamGenerator:
    def __init__(self):
        self.rag_pipeline = RAGPipeline()

    def generate_exam(self, topic: str, num_questions: int = 10) -> Exam:
        logger.info(f"Generating exam for topic: {topic}, num_questions: {num_questions}")
        # For demonstration, generate dummy questions using RAGPipeline
        questions = []
        for i in range(num_questions):
            content = self.rag_pipeline.generate_content(query=topic)
            questions.append(f"Q{i+1}: {content}")
        return Exam(questions)
