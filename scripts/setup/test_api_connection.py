#!/usr/bin/env python3
"""
Test API connections for Gemini AI and Supabase
"""

import sys
import os

# Add project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, os.path.abspath(project_root))

from config.settings import (
    SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_ANON_KEY,
    GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL, VECTOR_DIMENSIONS
)
import google.generativeai as genai
from supabase import create_client, Client
from loguru import logger

def test_supabase_connection():
    """Test Supabase connection and basic operations"""
    try:
        # Test with service key (admin operations)
        client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        
        # Test basic connection
        response = client.table("documents").select("id").limit(1).execute()
        logger.info("✅ Supabase connection successful (service key)")
        
        # Test with anon key if available
        if SUPABASE_ANON_KEY:
            anon_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            anon_response = anon_client.table("documents").select("id").limit(1).execute()
            logger.info("✅ Supabase anon key connection successful")
        
        # Test database functions if they exist
        try:
            stats = client.rpc('get_database_stats').execute()
            logger.info("✅ Database functions are working")
            logger.info("📊 Database Statistics:")
            for stat in stats.data:
                logger.info(f"  {stat['table_name']}: {stat['row_count']} rows")
        except Exception as e:
            logger.info("ℹ️  Database functions not available yet (run initialize_database.py first)")
        
        return True
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {e}")
        return False

def test_gemini_connection():
    """Test Gemini API connection and embedding generation"""
    try:
        # Configure API
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("✅ Gemini API configured successfully")
        
        # Test embedding generation
        test_content = "This is a comprehensive test sentence for embedding generation using the Gemini API."
        
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=test_content,
            task_type="RETRIEVAL_DOCUMENT"
        )
        
        embedding = result['embedding']
        
        # Validate embedding
        if not isinstance(embedding, list):
            raise ValueError("Embedding is not a list")
        
        if len(embedding) != VECTOR_DIMENSIONS:
            logger.warning(f"⚠️  Expected {VECTOR_DIMENSIONS} dimensions, got {len(embedding)}")
        
        logger.info("✅ Gemini API connection successful")
        logger.info(f"✅ Embedding dimensions: {len(embedding)}")
        logger.info(f"✅ Sample values: {embedding[:5]}")
        logger.info(f"✅ Embedding range: [{min(embedding):.6f}, {max(embedding):.6f}]")
        
        # Test batch processing capability
        batch_content = [
            "Machine learning algorithms process data.",
            "Neural networks learn complex patterns.",
            "Data preprocessing improves model accuracy."
        ]
        
        batch_results = []
        for content in batch_content:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=content,
                task_type="RETRIEVAL_DOCUMENT"
            )
            batch_results.append(result['embedding'])
        
        logger.info(f"✅ Batch processing successful: {len(batch_results)} embeddings")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Gemini API connection failed: {e}")
        return False

def test_environment_configuration():
    """Test environment configuration and settings"""
    try:
        logger.info("🔍 Testing environment configuration...")
        
        # Check required environment variables
        required_vars = {
            'GEMINI_API_KEY': GEMINI_API_KEY,
            'SUPABASE_URL': SUPABASE_URL,
            'SUPABASE_SERVICE_KEY': SUPABASE_SERVICE_KEY
        }
        
        for var_name, var_value in required_vars.items():
            if not var_value:
                logger.error(f"❌ Missing required variable: {var_name}")
                return False
            else:
                # Mask sensitive values
                display_value = var_value[:10] + "..." if len(var_value) > 10 else var_value
                logger.info(f"✅ {var_name}: {display_value}")
        
        # Check optional variables
        optional_vars = {
            'SUPABASE_ANON_KEY': SUPABASE_ANON_KEY,
            'GEMINI_EMBEDDING_MODEL': GEMINI_EMBEDDING_MODEL
        }
        
        for var_name, var_value in optional_vars.items():
            if var_value:
                display_value = var_value[:10] + "..." if len(var_value) > 10 else var_value
                logger.info(f"✅ {var_name}: {display_value}")
            else:
                logger.info(f"ℹ️  {var_name}: Not set (using defaults)")
        
        logger.info("✅ Environment configuration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment configuration test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive API and configuration tests"""
    
    logger.info("🔄 Starting comprehensive API connection tests...")
    logger.info("="*60)
    
    # Test results tracking
    test_results = {}
    
    # Test 1: Environment Configuration
    logger.info("Test 1: Environment Configuration")
    test_results['environment'] = test_environment_configuration()
    logger.info("")
    
    # Test 2: Supabase Connection
    logger.info("Test 2: Supabase Database Connection")
    test_results['supabase'] = test_supabase_connection()
    logger.info("")
    
    # Test 3: Gemini API
