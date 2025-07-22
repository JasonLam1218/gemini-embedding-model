import requests
from typing import List
from loguru import logger
from config.settings import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL

class GeminiClient:
    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = GEMINI_EMBEDDING_MODEL):
        self.api_key = api_key
        self.model = model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent?key={self.api_key}"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        headers = {"Content-Type": "application/json"}
        data = {
            "requests": [
                {"content": text, "model": self.model}
                for text in texts
            ]
        }
        try:
            response = requests.post(self.base_url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            results = response.json()
            return [r["embedding"]["values"] for r in results["responses"]]
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
