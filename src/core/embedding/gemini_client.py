import google.generativeai as genai
import numpy as np
from typing import List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
import os
import time

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client with enhanced capabilities"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        genai.configure(api_key=self.api_key)
        
        # Initialize models
        self.embedding_model = "models/text-embedding-004"
        self.generation_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Configuration
        self.max_content_length = 30000
        self.max_retries = 3
        
        logger.info("✅ Enhanced Gemini client initialized with embedding and generation models")

    def _validate_and_truncate_content(self, text: str) -> str:
        """Validate and truncate content if necessary"""
        if not text or not text.strip():
            raise ValueError("Empty or invalid text content")

        text = text.strip()

        # Truncate if too long, preserving complete sentences
        if len(text) > self.max_content_length:
            truncated = text[:self.max_content_length]
            # Find last complete sentence
            last_period = truncated.rfind('.')
            if last_period > self.max_content_length * 0.8:
                truncated = truncated[:last_period + 1]
            
            logger.warning(f"⚠️ Content truncated from {len(text)} to {len(truncated)} characters")
            return truncated

        return text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed_text(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """Generate embedding for a single text using Gemini API"""
        try:
            # Validate and clean content
            clean_text = self._validate_and_truncate_content(text)

            result = genai.embed_content(
                model=self.embedding_model,
                content=clean_text,
                task_type=task_type
            )

            embedding = result['embedding']

            # Validate embedding result
            if not embedding or len(embedding) == 0:
                raise ValueError("Empty embedding returned from API")

            logger.debug(f"✅ Generated embedding with {len(embedding)} dimensions")
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def embed_texts(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """Generate embeddings for multiple texts with robust error handling"""
        if not texts:
            logger.warning("Empty text list provided")
            return []

        embeddings = []
        successful_count = 0
        failed_count = 0

        for i, text in enumerate(texts, 1):
            try:
                logger.info(f"🔄 Generating embedding {i}/{len(texts)}")
                
                # Add small delay between requests to avoid rate limiting
                if i > 1:
                    time.sleep(1)

                embedding = self.embed_text(text, task_type)
                
                if embedding and isinstance(embedding, list) and len(embedding) > 0:
                    embeddings.append(embedding)
                    successful_count += 1
                    logger.debug(f"✅ Successfully generated embedding {i} ({len(embedding)} dimensions)")
                else:
                    embeddings.append([])  # Empty list instead of None
                    failed_count += 1
                    logger.error(f"❌ Empty embedding returned for text {i}")

            except Exception as e:
                failed_count += 1
                logger.error(f"❌ Failed to embed text {i}/{len(texts)}: {e}")
                embeddings.append([])  # Empty list instead of None

        logger.info(f"📊 Embedding Results: {successful_count} successful, {failed_count} failed")
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding optimized for query/search"""
        return self.embed_text(query, task_type="RETRIEVAL_QUERY")

    def _get_generation_config(self, temperature: float = 0.7, 
                              max_tokens: int = 1000) -> genai.types.GenerationConfig:
        """Get generation configuration"""
        return genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=0.8,
            top_k=40
        )

    def generate_content(self, prompt: str, temperature: float = 0.7,
                        max_tokens: int = 1000) -> str:
        """Generate content using Gemini generative model with detailed logging"""
        
        # Log prompt details BEFORE sending
        logger.info("🔍 GEMINI API REQUEST DETAILS:")
        logger.info(f"📏 Prompt length: {len(prompt)} characters")
        logger.info(f"📊 Estimated tokens: {len(prompt.split()) * 1.3:.0f}")
        logger.info(f"🌡️ Temperature: {temperature}")
        logger.info(f"🔢 Max tokens: {max_tokens}")
        
        # Log first and last parts of prompt for verification
        logger.info("📝 PROMPT PREVIEW (First 500 chars):")
        logger.info(f"'{prompt[:500]}...'")
        
        logger.info("📝 PROMPT PREVIEW (Last 500 chars):")
        logger.info(f"'...{prompt[-500:]}'")
        
        # Count content sections
        content_sections = prompt.count("===")
        logger.info(f"📁 Content sections detected: {content_sections}")
        
        try:
            # Make API call
            logger.info("🚀 Sending request to Gemini 2.5 Flash...")
            response = self.generation_model.generate_content(
                prompt,
                generation_config=self._get_generation_config(temperature, max_tokens)
            )
            
            # Log response details
            logger.info("✅ GEMINI API RESPONSE RECEIVED:")
            logger.info(f"📏 Response length: {len(response.text)} characters")
            logger.info(f"📝 Response preview: '{response.text[:200]}...'")
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"❌ Gemini API request failed: {e}")
            logger.error(f"💡 Failed prompt length was: {len(prompt)} characters")
            raise


    def generate_structured_content(self, prompt: str, 
                                   structure_template: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured content following a specific template"""
        try:
            # Add structure instructions to prompt
            structure_prompt = f"""
{prompt}

Please format your response according to this structure:
{structure_template}

Provide a JSON-like structured response that matches this template.
"""
            
            response = self.generate_content(structure_prompt, temperature=0.3)
            
            # Try to parse as structured data (basic implementation)
            # In a production system, you might want more sophisticated parsing
            
            return {
                "generated_content": response,
                "structure_template": structure_template,
                "success": True
            }

        except Exception as e:
            logger.error(f"❌ Structured content generation failed: {e}")
            return {
                "generated_content": "",
                "structure_template": structure_template,
                "success": False,
                "error": str(e)
            }

    def batch_embed_with_metadata(self, texts_with_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch embed texts while preserving metadata"""
        results = []
        
        for item in texts_with_metadata:
            text = item.get('text', '')
            metadata = item.get('metadata', {})
            
            try:
                embedding = self.embed_text(text)
                result_item = {
                    **metadata,
                    'text': text,
                    'embedding': embedding,
                    'embedding_success': True
                }
                results.append(result_item)
                
            except Exception as e:
                logger.error(f"❌ Failed to embed text with metadata: {e}")
                result_item = {
                    **metadata,
                    'text': text,
                    'embedding': [],
                    'embedding_success': False,
                    'error': str(e)
                }
                results.append(result_item)

        return results

    def calculate_token_count(self, text: str) -> int:
        """Estimate token count for text (basic implementation)"""
        # This is a rough estimate - actual tokenization may differ
        return len(text.split()) * 1.3  # Approximate tokens per word

    def test_connection(self) -> bool:
        """Test connection to Gemini API"""
        try:
            test_text = "This is a test."
            embedding = self.embed_text(test_text)
            
            if embedding and len(embedding) > 0:
                logger.info("✅ Gemini API connection test successful")
                return True
            else:
                logger.error("❌ Gemini API connection test failed - no embedding returned")
                return False
                
        except Exception as e:
            logger.error(f"❌ Gemini API connection test failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        try:
            # List available models
            models = list(genai.list_models())
            
            embedding_models = [m for m in models if 'embedding' in m.name]
            generation_models = [m for m in models if 'gemini' in m.name]
            
            return {
                "embedding_models": [m.name for m in embedding_models],
                "generation_models": [m.name for m in generation_models],
                "current_embedding_model": self.embedding_model,
                "current_generation_model": "gemini-2.5-flash"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get model info: {e}")
            return {}
