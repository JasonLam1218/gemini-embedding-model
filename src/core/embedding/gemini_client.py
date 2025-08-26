import google.generativeai as genai
import numpy as np
import json
from typing import List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger
import os
import time
import signal
from .rate_limiter import gemini_rate_limiter
from config.settings import EMBEDDING_DIMENSIONS # Import EMBEDDING_DIMENSIONS

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

    def _flatten_embedding_data(self, item: Any, acc_list: List[float]):
        """
        Recursively flattens an item (which might be a nested list) into a single list of floats.
        This helper ensures no nested lists are present in the final embedding representation.
        """
        if isinstance(item, (list, tuple)):
            for sub_item in item:
                self._flatten_embedding_data(sub_item, acc_list)
        elif isinstance(item, (float, int)):
            acc_list.append(float(item))
        else:
            # Log a warning if unexpected type is encountered, but don't add it
            logger.warning(f"🚨 _flatten_embedding_data: Encountered unexpected embedding element type '{type(item)}'. Skipping this element.")

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
            
            parsed_embeddings_from_response: List[List[float]] = [] 

            if 'embeddings' in result and isinstance(result['embeddings'], list):
                # Standard and expected structure for batch results
                for item in result['embeddings']:
                    if 'embedding' in item and isinstance(item['embedding'], list):
                        # Explicitly flatten each individual embedding item
                        temp_embedding_flat = []
                        self._flatten_embedding_data(item['embedding'], temp_embedding_flat)
                        parsed_embeddings_from_response.append(temp_embedding_flat)
                    else:
                        logger.warning(f"🚨 Item in 'embeddings' list is malformed or missing 'embedding' key. Item: {item}. Appending empty list.")
                        parsed_embeddings_from_response.append([])
            elif 'embedding' in result:
                # This path is for non-standard API responses where a singular 'embedding' key is returned for a batch.
                logger.warning(f"🚨 Received singular 'embedding' key for a batch request (expected 'embeddings'). This is unexpected for `genai.embed_content` with List[str]. Attempting to parse.")
                
                raw_emb_data = result['embedding']
                
                # Flatten the entire raw_emb_data first if it's potentially nested before trying to split/parse
                temp_flat_raw_emb_data = []
                self._flatten_embedding_data(raw_emb_data, temp_flat_raw_emb_data)
                raw_emb_data = temp_flat_raw_emb_data # Use the flattened version for subsequent checks

                if isinstance(raw_emb_data, list) and all(isinstance(x, (float, int)) for x in raw_emb_data):
                    # It's a flat list of numbers. Now, try to split it into expected dimensions.
                    if len(raw_emb_data) % EMBEDDING_DIMENSIONS == 0 and len(raw_emb_data) > 0:
                        num_found_embeddings = len(raw_emb_data) // EMBEDDING_DIMENSIONS
                        
                        # --- START OF MODIFICATION ---
                        # Always split into `num_found_embeddings` chunks first.
                        # Then, take only the number of embeddings requested, padding if necessary.
                        split_embeddings = []
                        for j in range(num_found_embeddings):
                            start_idx = j * EMBEDDING_DIMENSIONS
                            end_idx = start_idx + EMBEDDING_DIMENSIONS
                            split_embeddings.append(raw_emb_data[start_idx:end_idx])

                        # Take up to the number of original texts requested
                        for j in range(len(processed_texts_for_api)):
                            if j < len(split_embeddings):
                                parsed_embeddings_from_response.append(split_embeddings[j])
                            else:
                                # This scenario would mean API returned fewer than it could split into
                                logger.warning(f"🚨 API anomaly: Fewer embeddings found after splitting ({len(split_embeddings)}) than original texts requested ({len(processed_texts_for_api)}). Padding with empty embeddings.")
                                parsed_embeddings_from_response.append([])
                        
                        if num_found_embeddings > len(processed_texts_for_api):
                            logger.warning(f"🚨 API anomaly: Concatenated embedding split into {num_found_embeddings} embeddings, but only {len(processed_texts_for_api)} texts were requested. Taking the first {len(processed_texts_for_api)} embeddings.")
                        # --- END OF MODIFICATION ---
                    elif len(raw_emb_data) == EMBEDDING_DIMENSIONS:
                        logger.warning(f"🚨 Received a single correctly-sized embedding for a batch request. This is highly unusual for a batch. Appending as one embedding.")
                        parsed_embeddings_from_response.append(raw_emb_data)
                    else:
                        logger.warning(f"🚨 Singular 'embedding' is a flat list of unexpected total length ({len(raw_emb_data)}) that is not a multiple of {EMBEDDING_DIMENSIONS}. Cannot safely split for a batch request. Appending empty lists for all requested texts.")
                        parsed_embeddings_from_response.extend([[] for _ in processed_texts_for_api])
                        
                elif isinstance(raw_emb_data, list) and all(isinstance(x, list) for x in raw_emb_data):
                    # This case handles when the singular 'embedding' key might contain a list of multiple embeddings directly
                    # Flatten each inner list before appending
                    for emb_list in raw_emb_data:
                        temp_embedding_flat = []
                        self._flatten_embedding_data(emb_list, temp_embedding_flat)
                        parsed_embeddings_from_response.append(temp_embedding_flat)
                    logger.debug(f"Parsed singular 'embedding' as a list of multiple lists. Count: {len(raw_emb_data)}")
                
                elif isinstance(raw_emb_data, str): # Attempt to parse JSON string
                    try:
                        parsed_emb = json.loads(raw_emb_data)
                        # Flatten the entire parsed JSON before further processing
                        temp_flat_json_emb = []
                        self._flatten_embedding_data(parsed_emb, temp_flat_json_emb)
                        parsed_emb = temp_flat_json_emb # Use the flattened version
                            
                        if isinstance(parsed_emb, list) and all(isinstance(x, (float, int)) for x in parsed_emb):
                            if len(parsed_emb) % EMBEDDING_DIMENSIONS == 0 and len(parsed_emb) > 0:
                                num_found_embeddings = len(parsed_emb) // EMBEDDING_DIMENSIONS
                                # --- START OF MODIFICATION ---
                                # Apply the same splitting and taking logic as above
                                split_embeddings = []
                                for j in range(num_found_embeddings):
                                    start_idx = j * EMBEDDING_DIMENSIONS
                                    end_idx = start_idx + EMBEDDING_DIMENSIONS
                                    split_embeddings.append(parsed_emb[start_idx:end_idx])
                                
                                for j in range(len(processed_texts_for_api)):
                                    if j < len(split_embeddings):
                                        parsed_embeddings_from_response.append(split_embeddings[j])
                                    else:
                                        logger.warning(f"🚨 API anomaly: Fewer JSON-parsed embeddings found after splitting ({len(split_embeddings)}) than original texts requested ({len(processed_texts_for_api)}). Padding with empty embeddings.")
                                        parsed_embeddings_from_response.append([])

                                if num_found_embeddings > len(processed_texts_for_api):
                                    logger.warning(f"🚨 API anomaly: JSON-parsed concatenated embedding split into {num_found_embeddings} embeddings, but only {len(processed_texts_for_api)} texts were requested. Taking the first {len(processed_texts_for_api)} embeddings.")
                                # --- END OF MODIFICATION ---
                            elif len(parsed_emb) == EMBEDDING_DIMENSIONS:
                                logger.warning(f"🚨 JSON-loaded a single correctly-sized embedding for a batch request. This is unusual. Appending as one embedding.")
                                parsed_embeddings_from_response.append(parsed_emb)
                            else:
                                logger.warning(f"🚨 JSON-loaded singular 'embedding' as a single flat list of unexpected length ({len(parsed_emb)}) that is not a multiple of {EMBEDDING_DIMENSIONS}. Cannot safely split for a batch request. Appending empty lists for all requested texts.")
                                parsed_embeddings_from_response.extend([[] for _ in processed_texts_for_api])
                        elif isinstance(parsed_emb, list) and all(isinstance(x, list) for x in parsed_emb):
                            parsed_embeddings_from_response.extend(parsed_emb)
                            logger.debug(f"JSON-loaded singular 'embedding' as a list of multiple lists. Count: {len(parsed_emb)}")
                        else:
                            logger.warning(f"🚨 JSON-loaded singular 'embedding' string is unexpected type '{type(parsed_emb)}'. Appending empty list.")
                            parsed_embeddings_from_response.extend([[] for _ in processed_texts_for_api])
                    except json.JSONDecodeError:
                        logger.warning(f"🚨 Raw embedding data for batch is a non-JSON string: '{raw_emb_data[:50]}...'. Appending empty list.")
                        parsed_embeddings_from_response.extend([[] for _ in processed_texts_for_api])
                else:
                    logger.warning(f"🚨 Raw embedding data for singular 'embedding' is unexpected type '{type(raw_emb_data)}'. Appending empty list for all requested texts.")
                    parsed_embeddings_from_response.extend([[] for _ in processed_texts_for_api])
            
            # --- IMPORTANT: Now, map the parsed embeddings back to the original `all_embeddings` list. ---
            # This ensures that even if the API response was malformed, we maintain the correct length
            # and position of embeddings relative to the original 'texts' input.
            
            # First, ensure parsed_embeddings_from_response has the same count as processed_texts_for_api
            # This is crucial for correct mapping and to handle API anomalies where more/fewer embeddings
            # were detected than requested. We prioritize the requested count.
            if len(parsed_embeddings_from_response) != len(processed_texts_for_api):
                logger.warning(f"Mismatch in count of parsed embeddings ({len(parsed_embeddings_from_response)}) and processed texts ({len(processed_texts_for_api)}). Adjusting for mapping.")
                # If more embeddings were parsed than texts, truncate.
                if len(parsed_embeddings_from_response) > len(processed_texts_for_api):
                    parsed_embeddings_from_response = parsed_embeddings_from_response[:len(processed_texts_for_api)]
                # If fewer, pad with empty lists.
                else:
                    parsed_embeddings_from_response.extend([[] for _ in range(len(processed_texts_for_api) - len(parsed_embeddings_from_response))])

            for i in range(len(processed_texts_for_api)): 
                original_index = original_indices_map[i]
                current_parsed_emb = parsed_embeddings_from_response[i]
                
                # Final validation before assigning to the main results list
                if isinstance(current_parsed_emb, list) and all(isinstance(x, (float, int)) for x in current_parsed_emb):
                    if len(current_parsed_emb) == EMBEDDING_DIMENSIONS: 
                        all_embeddings[original_index] = current_parsed_emb
                    else:
                        logger.warning(f"🚨 Final check: Embedding for original index {original_index} has unexpected dimensions ({len(current_parsed_emb)}). Expected {EMBEDDING_DIMENSIONS}. Setting to empty list. Data preview: {str(current_parsed_emb)[:100]}...")
                        all_embeddings[original_index] = []
                else:
                    logger.warning(f"🚨 Final check: Parsed embedding for original index {original_index} is malformed (not list of floats). Setting to empty list. Data preview: {str(current_parsed_emb)[:100]}...")
                    all_embeddings[original_index] = []
            
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
            
            # Explicitly flatten any list directly received from the API response
            self._flatten_embedding_data(embedding_data, processed_embedding)

            if not processed_embedding or len(processed_embedding) == 0 or not all(isinstance(x, (float, int)) for x in processed_embedding):
                raise ValueError("Empty or invalid embedding returned from API for single text after processing.")
            
            if len(processed_embedding) != EMBEDDING_DIMENSIONS: # <--- CRUCIAL DIMENSION CHECK
                logger.error(f"🚨 Single embedding has unexpected dimensions ({len(processed_embedding)}). Expected {EMBEDDING_DIMENSIONS}. Returning empty list.")
                return []

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