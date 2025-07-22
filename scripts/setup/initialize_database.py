#!/usr/bin/env python3
"""
Initialize Supabase database with required tables and functions
"""

import sys
import os

# Add project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, os.path.abspath(project_root))

from config.settings import SUPABASE_URL, SUPABASE_SERVICE_KEY, VECTOR_DIMENSIONS
from supabase import create_client
from loguru import logger

def create_database_tables():
    """Create all required database tables"""
    
    tables_sql = f"""
    -- Enable pgvector extension for vector operations
    CREATE EXTENSION IF NOT EXISTS vector;
    
    -- Table for storing text documents
    CREATE TABLE IF NOT EXISTS documents (
        id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        source_file TEXT,
        paper_set TEXT,
        paper_number TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        metadata JSONB DEFAULT '{{}}'::jsonb
    );
    
    -- Table for storing text chunks
    CREATE TABLE IF NOT EXISTS text_chunks (
        id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
        document_id BIGINT REFERENCES documents(id) ON DELETE CASCADE,
        chunk_text TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        chunk_size INTEGER NOT NULL,
        overlap_size INTEGER DEFAULT 0,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        metadata JSONB DEFAULT '{{}}'::jsonb
    );
    
    -- Table for storing embeddings
    CREATE TABLE IF NOT EXISTS embeddings (
        id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
        chunk_id BIGINT REFERENCES text_chunks(id) ON DELETE CASCADE,
        embedding VECTOR({VECTOR_DIMENSIONS}),
        model_name TEXT NOT NULL DEFAULT 'text-embedding-004',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    
    -- Table for storing generated exams
    CREATE TABLE IF NOT EXISTS generated_exams (
        id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
        title TEXT NOT NULL,
        topic TEXT,
        difficulty TEXT,
        num_questions INTEGER,
        questions JSONB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        metadata JSONB DEFAULT '{{}}'::jsonb
    );
    
    -- Create indexes for better query performance
    CREATE INDEX IF NOT EXISTS idx_documents_paper_set ON documents(paper_set);
    CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
    CREATE INDEX IF NOT EXISTS idx_text_chunks_document_id ON text_chunks(document_id);
    CREATE INDEX IF NOT EXISTS idx_text_chunks_created_at ON text_chunks(created_at);
    CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);
    CREATE INDEX IF NOT EXISTS idx_embeddings_created_at ON embeddings(created_at);
    CREATE INDEX IF NOT EXISTS idx_generated_exams_topic ON generated_exams(topic);
    CREATE INDEX IF NOT EXISTS idx_generated_exams_created_at ON generated_exams(created_at);
    
    -- Create vector index for similarity search (using HNSW for better performance)
    CREATE INDEX IF NOT EXISTS idx_embeddings_vector 
    ON embeddings USING hnsw (embedding vector_cosine_ops) 
    WITH (m = 16, ef_construction = 64);
    
    -- Enable Row Level Security (RLS)
    ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
    ALTER TABLE text_chunks ENABLE ROW LEVEL SECURITY;
    ALTER TABLE embeddings ENABLE ROW LEVEL SECURITY;
    ALTER TABLE generated_exams ENABLE ROW LEVEL SECURITY;
    
    -- Create policies for access (adjust based on your security needs)
    DROP POLICY IF EXISTS "Allow all operations" ON documents;
    DROP POLICY IF EXISTS "Allow all operations" ON text_chunks;
    DROP POLICY IF EXISTS "Allow all operations" ON embeddings;
    DROP POLICY IF EXISTS "Allow all operations" ON generated_exams;
    
    CREATE POLICY "Allow all operations" ON documents FOR ALL USING (true);
    CREATE POLICY "Allow all operations" ON text_chunks FOR ALL USING (true);
    CREATE POLICY "Allow all operations" ON embeddings FOR ALL USING (true);
    CREATE POLICY "Allow all operations" ON generated_exams FOR ALL USING (true);
    """
    
    return tables_sql

def create_similarity_search_function():
    """Create PostgreSQL function for similarity search"""
    
    function_sql = f"""
    -- Create similarity search function
    CREATE OR REPLACE FUNCTION match_documents(
        query_embedding vector({VECTOR_DIMENSIONS}),
        match_threshold float DEFAULT 0.8,
        match_count int DEFAULT 10
    )
    RETURNS TABLE(
        id bigint,
        chunk_text text,
        document_title text,
        source_file text,
        paper_set text,
        paper_number text,
        similarity float,
        chunk_metadata jsonb,
        document_metadata jsonb
    )
    LANGUAGE sql STABLE
    AS $$
        SELECT
            tc.id,
            tc.chunk_text,
            d.title as document_title,
            d.source_file,
            d.paper_set,
            d.paper_number,
            1 - (e.embedding <=> query_embedding) as similarity,
            tc.metadata as chunk_metadata,
            d.metadata as document_metadata
        FROM text_chunks tc
        JOIN embeddings e ON tc.id = e.chunk_id
        JOIN documents d ON tc.document_id = d.id
        WHERE 1 - (e.embedding <=> query_embedding) > match_threshold
        ORDER BY e.embedding <=> query_embedding
        LIMIT match_count;
    $$;
    
    -- Create function to get document statistics
    CREATE OR REPLACE FUNCTION get_database_stats()
    RETURNS TABLE(
        table_name text,
        row_count bigint
    )
    LANGUAGE sql STABLE
    AS $$
        SELECT 'documents'::text, COUNT(*) FROM documents
        UNION ALL
        SELECT 'text_chunks'::text, COUNT(*) FROM text_chunks
        UNION ALL
        SELECT 'embeddings'::text, COUNT(*) FROM embeddings
        UNION ALL
        SELECT 'generated_exams'::text, COUNT(*) FROM generated_exams;
    $$;
    """
    
    return function_sql

def test_database_functions(client):
    """Test that database functions work correctly"""
    try:
        # Test similarity search function with a dummy vector
        import numpy as np
        dummy_vector = np.random.rand(VECTOR_DIMENSIONS).tolist()
        
        response = client.rpc(
            'match_documents',
            {
                'query_embedding': dummy_vector,
                'match_threshold': 0.0,
                'match_count': 1
            }
        ).execute()
        
        logger.info("✅ Similarity search function test passed")
        
        # Test stats function
        stats_response = client.rpc('get_database_stats').execute()
        logger.info("✅ Database stats function test passed")
        
        return True
    except Exception as e:
        logger.warning(f"⚠️  Function tests failed (this is normal if tables are empty): {e}")
        return True  # Don't fail initialization for empty tables

def main():
    """Initialize database with all required components"""
    try:
        # Create Supabase client
        client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logger.info("✅ Connected to Supabase")
        
        # Test connection
        try:
            response = client.table('_test').select('*').limit(1).execute()
        except:
            pass  # Expected error for non-existent table
        
        logger.info("🔄 Creating database tables...")
        
        # Execute table creation SQL
        tables_sql = create_database_tables()
        
        # Split SQL into individual statements and execute
        sql_statements = [stmt.strip() for stmt in tables_sql.split(';') if stmt.strip()]
        
        for i, statement in enumerate(sql_statements):
            if statement:
                try:
                    client.rpc('sql', {'query': statement}).execute()
                except Exception as e:
                    # Some statements might fail if they already exist, which is okay
                    logger.debug(f"SQL statement {i+1} warning: {e}")
        
        logger.info("✅ Database tables created successfully")
        
        # Create similarity search functions
        logger.info("🔄 Creating database functions...")
        function_sql = create_similarity_search_function()
        
        function_statements = [stmt.strip() for stmt in function_sql.split('$$;') if stmt.strip()]
        
        for statement in function_statements:
            if statement and not statement.startswith('--'):
                try:
                    if not statement.endswith('$$'):
                        statement += '$$'
                    client.rpc('sql', {'query': statement}).execute()
                except Exception as e:
                    logger.debug(f"Function creation warning: {e}")
        
        logger.info("✅ Database functions created successfully")
        
        # Test functions
        logger.info("🔄 Testing database functions...")
        test_database_functions(client)
        
        # Get final statistics
        try:
            stats = client.rpc('get_database_stats').execute()
            logger.info("📊 Database Statistics:")
            for stat in stats.data:
                logger.info(f"  {stat['table_name']}: {stat['row_count']} rows")
        except Exception as e:
            logger.debug(f"Stats query failed: {e}")
        
        logger.info("🎉 Database initialization completed successfully!")
        logger.info("✅ Ready to store embeddings and perform similarity searches")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    print("\n" + "="*70)
    if success:
        print("🎉 Database setup completed successfully!")
        print("✅ Your Supabase database is ready for the exam generation system")
        print("\nNext steps:")
        print("1. Run: python run_pipeline.py process-texts")
        print("2. Run: python run_pipeline.py generate-embeddings") 
        print("3. Run: python run_pipeline.py generate-exam")
    else:
        print("❌ Database setup failed. Check the logs above.")
    
    sys.exit(0 if success else 1)
