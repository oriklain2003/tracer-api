"""
AI Classification Module for Flight Anomalies

Provides asynchronous AI-powered classification of detected anomaly flights using Google Gemini.
Generates concise 3-6 word root cause summaries and stores results in PostgreSQL.
"""
from __future__ import annotations

import logging
import time
import base64
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class AIClassifier:
    """
    Asynchronous AI classification engine for anomaly flights.
    
    Uses Google Gemini to analyze flight anomalies and generate concise root cause summaries.
    Runs classifications in background threads to avoid blocking the monitor.
    """
    
    SYSTEM_INSTRUCTION = """As an expert aviation data analyst, your core mission is to perform a surgical inference of the root cause by correlating the detected flight anomaly with the provided environmental context. You must move beyond simple observation to determine exactly why the anomaly was a logical necessity or a specific response to the surrounding conditions, ensuring that the environmental data justifies the flight behavior. It is critical that your final output is restricted to a professional summary of exactly three to six words, providing only the ultimate root cause without any introductory phrases, filler text, or repetition of the input. if the callsign is SHAHD it dosent mean that its a uav it can be any other platform"""
    
    def __init__(self, gemini_api_key: str, schema: str = 'live', max_workers: int = 2):
        """
        Initialize the AI classifier.
        
        Args:
            gemini_api_key: Google Gemini API key
            schema: PostgreSQL schema name (default: 'live')
            max_workers: Maximum number of concurrent classification threads (default: 2)
        """
        self.schema = schema
        self.gemini_client = None
        self.types = None
        
        # Initialize Gemini client
        try:
            from google import genai
            from google.genai import types
            
            self.gemini_client = genai.Client(
                api_key=gemini_api_key,
                http_options={'api_version': 'v1alpha'}
            )
            self.types = types
            logger.info("Gemini AI client initialized successfully")
            
        except ImportError:
            logger.error("google-genai package not installed - AI classification disabled")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
        
        # Thread pool for async execution
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="AIClassifier")
        logger.info(f"AI Classifier thread pool initialized with {max_workers} workers")
        
        # Ensure table exists
        try:
            from service.pg_provider import create_ai_classifications_table
            create_ai_classifications_table(schema)
        except Exception as e:
            logger.warning(f"Could not create ai_classifications table: {e}")
    
    def classify_async(
        self, 
        flight_id: str, 
        flight_data: List[Dict], 
        anomaly_report: Dict, 
        metadata: Dict
    ) -> None:
        """
        Trigger asynchronous classification (non-blocking).
        
        Args:
            flight_id: Flight identifier
            flight_data: List of track point dictionaries
            anomaly_report: Anomaly report from pipeline
            metadata: Flight metadata dictionary
        """
        # Submit to thread pool
        future = self.executor.submit(
            self._classify_sync,
            flight_id,
            flight_data,
            anomaly_report,
            metadata
        )
        
        # Attach completion callback
        future.add_done_callback(self._handle_completion)
        
        logger.debug(f"Async classification queued for flight {flight_id}")
    
    def _classify_sync(
        self,
        flight_id: str,
        flight_data: List[Dict],
        anomaly_report: Dict,
        metadata: Dict
    ) -> Dict[str, Any]:
        """
        Synchronous classification logic (runs in background thread).
        
        Args:
            flight_id: Flight identifier
            flight_data: List of track point dictionaries
            anomaly_report: Anomaly report from pipeline
            metadata: Flight metadata dictionary
        
        Returns:
            Dict containing classification results
        """
        start_time = time.time()
        result = {
            'flight_id': flight_id,
            'classification_text': None,
            'full_response': None,
            'error_message': None,
            'processing_time_sec': 0.0,
            'gemini_model': 'gemini-3-flash-preview'
        }
        
        try:
            logger.info(f"Starting AI classification for flight {flight_id}")
            
            # Step 1: Build context
            context_text = self._build_context(anomaly_report, metadata, flight_data)
            
            # Step 2: Generate map image
            image_bytes = self._generate_map_image(flight_data)
            
            # Step 3: Call Gemini API
            classification_text = self._call_gemini_api(context_text, image_bytes)
            
            # Step 4: Store results
            result['classification_text'] = classification_text
            result['full_response'] = classification_text
            result['processing_time_sec'] = time.time() - start_time
            
            # Step 5: Save to database
            self._save_to_database(result)
            
            logger.info(f"Classification completed for {flight_id}: '{classification_text}' ({result['processing_time_sec']:.2f}s)")
            
        except Exception as e:
            logger.error(f"Classification failed for {flight_id}: {e}", exc_info=True)
            result['error_message'] = str(e)
            result['processing_time_sec'] = time.time() - start_time
            
            # Still try to save error to database
            try:
                self._save_to_database(result)
            except Exception as save_err:
                logger.error(f"Failed to save error result: {save_err}")
        
        return result
    
    def _build_context(
        self,
        anomaly_report: Dict,
        metadata: Dict,
        flight_data: List[Dict]
    ) -> str:
        """
        Build context text for Gemini API.
        
        Args:
            anomaly_report: Anomaly report from pipeline
            metadata: Flight metadata dictionary
            flight_data: List of track point dictionaries
        
        Returns:
            str: Formatted context text
        """
        try:
            from ai_helpers import build_anomaly_context
            context = build_anomaly_context(anomaly_report, metadata, flight_data)
            
            # Add task instruction
            context += "\n\n=== TASK ==="
            context += "\nA map visualization of the flight path is attached. Use it to analyze the flight pattern visually."
            context += "\nBased on the flight data, map, and analysis above, provide a concise 3-6 word summary of the root cause."
            context += "\nOutput ONLY the root cause summary without any introductory text or explanation."
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to build context: {e}")
            # Fallback to basic context
            return f"Flight ID: {metadata.get('flight_id', 'Unknown')}\nCallsign: {metadata.get('callsign', 'Unknown')}\nAnomaly detected."
    
    def _generate_map_image(self, flight_data: List[Dict]) -> Optional[bytes]:
        """
        Generate map image for the flight.
        
        Args:
            flight_data: List of track point dictionaries
        
        Returns:
            bytes: PNG image bytes, or None if generation fails
        """
        try:
            from ai_helpers import generate_flight_map
            image_bytes = generate_flight_map(flight_data, width=800, height=600)
            
            if image_bytes:
                logger.debug(f"Generated map image ({len(image_bytes)} bytes)")
            else:
                logger.warning("Map image generation returned None")
            
            return image_bytes
            
        except Exception as e:
            logger.warning(f"Failed to generate map image: {e}")
            return None
    
    def _call_gemini_api(self, context_text: str, image_bytes: Optional[bytes]) -> str:
        """
        Call Gemini API for classification.
        
        Args:
            context_text: Formatted context text
            image_bytes: Optional PNG image bytes
        
        Returns:
            str: Classification text (3-6 words)
        """
        try:
            # Build request parts
            parts = [self.types.Part(text=context_text)]
            
            # Add image if available
            if image_bytes:
                parts.append(
                    self.types.Part(
                        inline_data=self.types.Blob(
                            mime_type="image/png",
                            data=image_bytes
                        )
                    )
                )
                logger.debug("Added map image to Gemini request")
            
            # Configure request
            config = self.types.GenerateContentConfig(
                system_instruction=self.SYSTEM_INSTRUCTION,
                tools=[
                    self.types.Tool(google_search=self.types.GoogleSearch()),
                    {'code_execution': {}}
                ]
            )
            
            content = self.types.Content(parts=parts)
            
            # Call API
            logger.debug("Calling Gemini API...")
            api_start = time.time()
            
            response = self.gemini_client.models.generate_content(
                model="gemini-3-flash-preview",
                config=config,
                contents=[content]
            )
            
            api_time = time.time() - api_start
            logger.debug(f"Gemini API responded in {api_time:.2f}s")
            
            # Extract text from response
            classification_text = self._extract_gemini_text(response)
            
            if not classification_text:
                raise ValueError("Empty response from Gemini API")
            
            return classification_text.strip()
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise
    
    def _extract_gemini_text(self, response) -> str:
        """
        Extract text from Gemini response.
        
        Args:
            response: Gemini API response object
        
        Returns:
            str: Extracted text
        """
        try:
            output = ""
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    output += part.text
            return output.strip()
        except Exception as e:
            logger.error(f"Failed to extract text from Gemini response: {e}")
            return ""
    
    def _save_to_database(self, result: Dict[str, Any]) -> None:
        """
        Save classification result to PostgreSQL.
        
        Args:
            result: Classification result dictionary
        """
        try:
            from service.pg_provider import save_ai_classification
            
            success = save_ai_classification(
                flight_id=result['flight_id'],
                classification=result,
                schema=self.schema
            )
            
            if success:
                logger.debug(f"Saved classification result for {result['flight_id']}")
            else:
                logger.error(f"Failed to save classification result for {result['flight_id']}")
                
        except Exception as e:
            logger.error(f"Database save error: {e}")
            raise
    
    def _handle_completion(self, future):
        """
        Callback for completed classification tasks.
        
        Args:
            future: Completed Future object
        """
        try:
            result = future.result()
            
            if result.get('error_message'):
                logger.error(f"Classification task failed: {result['error_message']}")
            else:
                flight_id = result.get('flight_id', 'Unknown')
                classification = result.get('classification_text', 'N/A')
                logger.info(f"✅ Classification task completed: {flight_id} -> '{classification}'")
                
        except Exception as e:
            logger.error(f"Error in completion callback: {e}", exc_info=True)
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown the thread pool executor.
        
        Args:
            wait: Whether to wait for pending tasks to complete
        """
        logger.info(f"Shutting down AI Classifier (wait={wait})...")
        self.executor.shutdown(wait=wait)
        logger.info("AI Classifier shutdown complete")
