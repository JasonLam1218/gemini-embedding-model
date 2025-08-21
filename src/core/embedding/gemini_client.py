import google.generativeai as genai
import numpy as np
import json # ADD THIS IMPORT
from typing import List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger
import os
import time
import signal
from .rate_limiter import gemini_rate_limiter

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client with enhanced capabilities"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=self.api_key)
        
        # Initialize models
        self.embedding_model = "models/gemini-embedding-001"
        self.generation_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Configuration: Set a concrete maximum content length for safety
        self.max_content_length = 30000 
        self.max_retries = 2
        
        logger.info("✅ Enhanced Gemini client initialized with embedding and generation models")

    def _validate_and_truncate_content(self, text: str) -> str:
        """Validate and truncate content if necessary."""
        if not text or not text.strip():
            logger.warning("⚠️ Empty or whitespace-only text provided for embedding. Returning empty string.")
            return ""

        text = text.strip()
        
        if len(text) > self.max_content_length:
            truncated = text[:self.max_content_length]
            last_period = truncated.rfind('.')
            if last_period > self.max_content_length * 0.8:
                truncated = truncated[:last_period + 1]
            logger.warning(f"⚠️ Content truncated from {len(text)} to {len(truncated)} characters to fit API limits.")
            return truncated
        
        return text

    @gemini_rate_limiter
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=5),
        retry=retry_if_exception_type((
            Exception,
        ))
    )
    def embed_texts_batch(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """Generate embeddings for a list of texts using Gemini API in a single call."""
        if not texts:
            logger.warning("No texts provided for batch embedding.")
            return []

        processed_texts_for_api = []
        original_indices_map = []
        for i, text in enumerate(texts):
            clean_text = self._validate_and_truncate_content(text)
            if clean_text:
                processed_texts_for_api.append(clean_text)
                original_indices_map.append(i)
            else:
                logger.warning(f"Skipping text at original index {i} due to empty content after validation: '{text[:50]}...'")

        # Initialize the result list with empty lists to maintain original order and handle failures
        all_embeddings = [[] for _ in texts] 

        if not processed_texts_for_api:
            logger.warning("All texts were empty or invalid after preprocessing. Returning a list of empty embeddings.")
            return all_embeddings


        try:
            logger.debug(f"Sending {len(processed_texts_for_api)} texts for batch embedding. Sample: '{processed_texts_for_api[0][:100]}...'")
            result = genai.embed_content(
                model=self.embedding_model,
                content=processed_texts_for_api,
                task_type=task_type
            )
            logger.debug(f"Raw embed_content result: {result}")
            
            api_returned_embeddings_raw = []
            if 'embeddings' in result and isinstance(result['embeddings'], list):
                # This is the expected structure for batch results
                api_returned_embeddings_raw = [item['embedding'] for item in result['embeddings']]
            elif 'embedding' in result: # Handles singular 'embedding' key
                # Check if it's already a list of lists, indicating it's a wrapped single embedding
                # for a potentially multi-item batch request (unexpected API behavior)
                if isinstance(result['embedding'], list) and len(result['embedding']) == 1 and isinstance(result['embedding'][0], list):
                    # If it's a list containing a single list (e.g., [[float, float]]), take the inner list
                    api_returned_embeddings_raw = [result['embedding'][0]]
                    logger.debug(f"Flattened single nested embedding returned by API for batch. Original shape: {len(result['embedding'])}x{len(result['embedding'][0]) if result['embedding'][0] else 0}")
                elif isinstance(result['embedding'], list):
                    # If it's a flat list (e.g., [float, float]), assume it's the single embedding for the whole batch
                    api_returned_embeddings_raw = [result['embedding']]
                    logger.debug(f"Treated single flat embedding returned by API for batch. Embedding length: {len(result['embedding'])}")
                elif isinstance(result['embedding'], str):
                    try:
                        # Attempt to load from JSON string if it's a string representation of an embedding
                        parsed_emb = json.loads(result['embedding'])
                        if isinstance(parsed_emb, list) and len(parsed_emb) == 1 and isinstance(parsed_emb[0], list):
                             api_returned_embeddings_raw = [parsed_emb[0]]
                             logger.debug(f"JSON-loaded and flattened nested single embedding string.")
                        elif isinstance(parsed_emb, list):
                            api_returned_embeddings_raw = [parsed_emb]
                            logger.debug(f"JSON-loaded flat embedding string.")
                        else:
                            logger.warning(f"🚨 JSON-loaded embedding string for batch is unexpected type '{type(parsed_emb)}'. Setting to empty list for this.")
                    except json.JSONDecodeError:
                        logger.warning(f"🚨 Raw embedding data for batch is a non-JSON string: '{result['embedding'][:50]}...'. Setting to empty list.")
                else:
                    logger.warning(f"🚨 Raw embedding data for batch is unexpected type '{type(result['embedding'])}'. Setting to empty list.")
            else:
                logger.error(f"Unexpected API response structure for embeddings. Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
                raise ValueError("Neither 'embedding' nor 'embeddings' found in the API response or invalid structure.")
            
            if len(api_returned_embeddings_raw) != len(processed_texts_for_api):
                logger.warning(f"Mismatch in returned embeddings count ({len(api_returned_embeddings_raw)}) and requested texts count ({len(processed_texts_for_api)}) for batch. This might indicate partial API failure or unexpected behavior.")
            
            # Process and validate each raw embedding, attempting JSON deserialization if needed
            for i, raw_emb_data in enumerate(api_returned_embeddings_raw):
                if i < len(original_indices_map): # Ensure index is valid for mapping
                    original_index = original_indices_map[i]
                    processed_emb = []
                    
                    if isinstance(raw_emb_data, str):
                        try:
                            # Attempt to load from JSON string if it's a string
                            processed_emb = json.loads(raw_emb_data)
                            logger.debug(f"Successfully JSON-loaded embedding string for original index {original_index}.")
                        except json.JSONDecodeError:
                            logger.warning(f"🚨 Raw embedding data for original index {original_index} is a non-JSON string: '{raw_emb_data[:50]}...'. Setting to empty list.")
                            processed_emb = []
                    elif isinstance(raw_emb_data, list):
                        processed_emb = raw_emb_data
                    elif isinstance(raw_emb_data, np.ndarray):
                        processed_emb = raw_emb_data.flatten().tolist()
                    else:
                        logger.warning(f"🚨 Raw embedding data for original index {original_index} is unexpected type '{type(raw_emb_data)}'. Setting to empty list.")
                        processed_emb = []

                    # Final validation of the processed embedding list
                    if isinstance(processed_emb, list) and all(isinstance(x, (float, int)) for x in processed_emb):
                        all_embeddings[original_index] = processed_emb
                    else:
                        logger.warning(f"🚨 Processed embedding for original index {original_index} is malformed (not list of floats). Setting to empty list. Data: {str(processed_emb)[:100]}...")
                        all_embeddings[original_index] = []
                else:
                    logger.warning(f"Extraneous embedding returned by API at index {i} that doesn't map to an original request. Ignoring.")

            if not all_embeddings or any(not e for e in all_embeddings):
                logger.warning("Batch embedding generation completed, but some texts resulted in empty/invalid embeddings. Review logs for details.")
            
            logger.debug(f"✅ Generated {len(all_embeddings)} embeddings in batch with proper mapping.")
            return all_embeddings
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise

    def embed_text(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """Generate embedding for a single text using Gemini API"""
        try:
            clean_text = self._validate_and_truncate_content(text)
            if not clean_text:
                logger.warning("Attempted to embed an empty string after validation in embed_text. Returning empty list.")
                return []

            result = genai.embed_content(
                model=self.embedding_model,
                content=clean_text,
                task_type=task_type
            )
            
            embedding_data = result['embedding']
            
            # Ensure embedding_data is a list of floats, handle string case for robustness
            processed_embedding = []
            if isinstance(embedding_data, str):
                try:
                    processed_embedding = json.loads(embedding_data)
                    logger.debug("Successfully JSON-loaded single embedding string.")
                except json.JSONDecodeError:
                    logger.warning(f"🚨 Single embedding data is a non-JSON string: '{embedding_data[:50]}...'. Setting to empty list.")
                    processed_embedding = []
            elif isinstance(embedding_data, list):
                processed_embedding = embedding_data
            elif isinstance(embedding_data, np.ndarray):
                processed_embedding = embedding_data.flatten().tolist()
            else:
                logger.warning(f"🚨 Single embedding data is unexpected type '{type(embedding_data)}'. Setting to empty list.")
                processed_embedding = []

            if not processed_embedding or len(processed_embedding) == 0 or not all(isinstance(x, (float, int)) for x in processed_embedding):
                raise ValueError("Empty or invalid embedding returned from API for single text after processing.")
            
            logger.debug(f"✅ Generated embedding with {len(processed_embedding)} dimensions")
            return processed_embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def generate_content(self, prompt: str, temperature: float = 0.7,
                        max_tokens: int = 1000, timeout: int = 180) -> str:
        """Generate content using Gemini generative model with timeout"""
        
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
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Request timed out after {timeout} seconds")
        
        try:
            # Set timeout
            # signal.signal(signal.SIGALRM, timeout_handler) # Temporarily commenting out signal for potential async issues
            # signal.alarm(timeout) # Temporarily commenting out signal for potential async issues
            
            # Make API call
            logger.info("🚀 Sending request to Gemini 2.5 Flash...")
            response = self.generation_model.generate_content(
                prompt,
                generation_config=self._get_generation_config(temperature, max_tokens)
            )
            
            # signal.alarm(0)  # Cancel alarm # Temporarily commenting out signal for potential async issues
            
            # Log response details
            logger.info("✅ GEMINI API RESPONSE RECEIVED:")
            logger.info(f"📏 Response length: {len(response.text)} characters")
            
            return response.text.strip()
            
        except TimeoutError as e:
            logger.error(f"⏰ Request timed out: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Gemini API request failed: {e}")
            logger.error(f"💡 Failed prompt length was: {len(prompt)} characters")
            raise
        finally:
            # signal.alarm(0)  # Ensure alarm is cancelled # Temporarily commenting out signal for potential async issues
            pass # Keep pass if signal is commented out

    def _get_generation_config(self, temperature: float = 0.7,
                              max_tokens: int = 1000) -> genai.types.GenerationConfig:
        """Get generation configuration"""
        return genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=0.8,
            top_k=40
        )