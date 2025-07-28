"""
Complete logging configuration for the gemini-embedding-model pipeline
Configures comprehensive file logging to data/output/logs directory
"""

import sys
import os
from pathlib import Path
from loguru import logger
from datetime import datetime
import json
import platform
import psutil

def setup_comprehensive_logging():
    """Setup complete logging system with file output to data/output/logs"""
    
    # Create logs directory structure
    logs_dir = Path("data/output/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove default logger to avoid duplicates
    logger.remove()
    
    # === CONSOLE LOGGING (Enhanced with colors) ===
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
        backtrace=False,
        diagnose=False
    )
    
    # === MAIN APPLICATION LOG (Everything) ===
    logger.add(
        logs_dir / "pipeline_application.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="25 MB",
        retention="20 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
        encoding="utf-8"
    )
    
    # === ERROR-ONLY LOG ===
    logger.add(
        logs_dir / "pipeline_errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
        level="ERROR",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
        encoding="utf-8"
    )
    
    # === API OPERATIONS LOG (Gemini, Supabase) ===
    logger.add(
        logs_dir / "api_operations.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function} - {message}",
        level="INFO",
        rotation="15 MB",
        retention="10 days",
        compression="zip",
        filter=lambda record: any(keyword in record["message"].lower() for keyword in [
            "api", "embedding", "generation", "quota", "gemini", "supabase", "request", "response",
            "embed_text", "generate_content", "insert_document", "similarity_search"
        ]),
        encoding="utf-8"
    )
    
    # === PIPELINE OPERATIONS LOG (Success/Failure indicators) ===
    logger.add(
        logs_dir / "pipeline_operations.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function} - {message}",
        level="INFO",
        rotation="15 MB",
        retention="15 days",
        compression="zip",
        filter=lambda record: any(keyword in record["message"] for keyword in [
            "✅", "❌", "⚠️", "🔄", "📊", "📝", "🧠", "📄", "🔍", "💡", "🚀", "🎯", "📁", "💾", "📋", "🏁"
        ]),
        encoding="utf-8"
    )
    
    # === DATABASE OPERATIONS LOG ===
    logger.add(
        logs_dir / "database_operations.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function} - {message}",
        level="INFO",
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        filter=lambda record: any(keyword in record["message"].lower() for keyword in [
            "supabase", "database", "insert", "query", "vector", "embedding", "document", "chunk",
            "table", "insert_document", "insert_text_chunks", "insert_embeddings", "similarity_search"
        ]),
        encoding="utf-8"
    )
    
    # === TEXT PROCESSING LOG ===
    logger.add(
        logs_dir / "text_processing.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function} - {message}",
        level="INFO",
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        filter=lambda record: any(keyword in record["message"].lower() for keyword in [
            "processing", "chunk", "load", "text", "document", "content", "file", "directory"
        ]),
        encoding="utf-8"
    )
    
    # === EXAM GENERATION LOG ===
    logger.add(
        logs_dir / "exam_generation.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function} - {message}",
        level="INFO",
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        filter=lambda record: any(keyword in record["message"].lower() for keyword in [
            "exam", "question", "answer", "marking", "scheme", "generate", "structured"
        ]),
        encoding="utf-8"
    )
    
    # === SESSION-SPECIFIC LOG (Each pipeline run) ===
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.add(
        logs_dir / f"session_{session_timestamp}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function} - {message}",
        level="INFO",
        rotation="50 MB",
        encoding="utf-8"
    )
    
    # === PERFORMANCE LOG ===
    logger.add(
        logs_dir / "performance.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        level="INFO",
        rotation="8 MB",
        retention="7 days",
        filter=lambda record: any(keyword in record["message"].lower() for keyword in [
            "processing", "generated", "completed", "time", "duration", "speed", "batch", "seconds", "minutes"
        ]),
        encoding="utf-8"
    )
    
    # Log successful initialization
    logger.info("📝 Comprehensive file logging system initialized successfully")
    logger.info(f"📁 Log files location: {logs_dir.absolute()}")
    
    return logs_dir

def log_system_information():
    """Log comprehensive system and environment information"""
    logger.info("🖥️  COMPLETE SYSTEM INFORMATION")
    logger.info("=" * 70)
    
    # Platform information
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Architecture: {platform.architecture()[0]}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Python Executable: {sys.executable}")
    logger.info(f"Working Directory: {Path.cwd()}")
    logger.info(f"Process ID: {os.getpid()}")
    
    # Memory information
    try:
        memory = psutil.virtual_memory()
        logger.info(f"Total Memory: {memory.total // (1024**3)} GB")
        logger.info(f"Available Memory: {memory.available // (1024**3)} GB")
        logger.info(f"Memory Usage: {memory.percent}%")
        
        # Disk space
        disk = psutil.disk_usage('.')
        logger.info(f"Disk Total: {disk.total // (1024**3)} GB")
        logger.info(f"Disk Free: {disk.free // (1024**3)} GB")
        logger.info(f"Disk Usage: {(disk.used / disk.total) * 100:.1f}%")
    except:
        logger.info("Memory/Disk information not available")
    
    # Environment variables
    logger.info("🔧 ENVIRONMENT CONFIGURATION")
    env_vars = ['GEMINI_API_KEY', 'SUPABASE_URL', 'SUPABASE_SERVICE_KEY', 'SUPABASE_ANON_KEY']
    for var in env_vars:
        value = os.getenv(var)
        if value:
            if 'key' in var.lower():
                masked_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
                logger.info(f"{var}: {masked_value}")
            else:
                logger.info(f"{var}: {value}")
        else:
            logger.warning(f"{var}: NOT SET")
    
    # Python packages information
    logger.info("📦 KEY PACKAGES")
    try:
        import pkg_resources
        key_packages = ['loguru', 'supabase', 'google-generativeai', 'numpy', 'spacy', 'nltk']
        for package in key_packages:
            try:
                version = pkg_resources.get_distribution(package).version
                logger.info(f"{package}: {version}")
            except:
                logger.warning(f"{package}: Not found")
    except:
        logger.info("Package information not available")
    
    logger.info("=" * 70)

def log_pipeline_start(pipeline_name: str, parameters: dict = None):
    """Log the start of a pipeline operation with complete context"""
    logger.info("🚀 PIPELINE EXECUTION STARTED")
    logger.info("=" * 60)
    logger.info(f"Pipeline: {pipeline_name}")
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    logger.info(f"Session ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    if parameters:
        logger.info("📋 Execution Parameters:")
        for key, value in parameters.items():
            logger.info(f"  {key}: {value}")
    
    # Log current directory structure
    try:
        logger.info("📁 Current Directory Structure:")
        for item in ["data/input", "data/output", "src", "config"]:
            path = Path(item)
            if path.exists():
                if path.is_dir():
                    file_count = len(list(path.rglob("*")))
                    logger.info(f"  {item}/: {file_count} files")
                else:
                    logger.info(f"  {item}: exists")
            else:
                logger.info(f"  {item}: NOT FOUND")
    except:
        logger.info("  Directory structure scan failed")
    
    logger.info("=" * 60)

def log_pipeline_end(pipeline_name: str, success: bool = True, duration: float = None, 
                    results: dict = None, error: str = None):
    """Log the end of a pipeline operation with complete results"""
    status = "✅ COMPLETED SUCCESSFULLY" if success else "❌ FAILED"
    logger.info(f"🏁 PIPELINE EXECUTION {status}")
    logger.info("=" * 60)
    logger.info(f"Pipeline: {pipeline_name}")
    logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    logger.info(f"Status: {'SUCCESS' if success else 'FAILURE'}")
    
    if duration:
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = duration % 60
        if hours > 0:
            logger.info(f"Duration: {hours}h {minutes}m {seconds:.2f}s")
        elif minutes > 0:
            logger.info(f"Duration: {minutes}m {seconds:.2f}s")
        else:
            logger.info(f"Duration: {seconds:.2f}s")
    
    if results:
        logger.info("📊 Execution Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")
    
    if error:
        logger.error(f"❌ Error Details: {error}")
    
    logger.info("=" * 60)

def create_operation_context(operation_name: str):
    """Create a logging context decorator for pipeline operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger.info(f"🔄 Starting {operation_name}")
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Completed {operation_name} in {duration:.2f}s")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"❌ Failed {operation_name} after {duration:.2f}s: {e}")
                raise
        return wrapper
    return decorator

def get_log_stats():
    """Get statistics about current log files"""
    logs_dir = Path("data/output/logs")
    if not logs_dir.exists():
        return {}
    
    stats = {}
    for log_file in logs_dir.glob("*.log*"):
        size = log_file.stat().st_size
        stats[log_file.name] = {
            "size_bytes": size,
            "size_mb": size / (1024 * 1024),
            "modified": datetime.fromtimestamp(log_file.stat().st_mtime)
        }
    return stats

# Initialize logging when module is imported
def initialize_logging():
    """Initialize the complete logging system"""
    logs_dir = setup_comprehensive_logging()
    log_system_information()
    return logs_dir

# Export functions
__all__ = [
    'setup_comprehensive_logging', 
    'log_system_information', 
    'log_pipeline_start', 
    'log_pipeline_end',
    'create_operation_context',
    'initialize_logging',
    'get_log_stats'
]
