"""
Comprehensive logging configuration for the gemini-embedding-model pipeline.
Provides file-based logging with multiple log levels and operation tracking.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Dict, Any, Optional
import json

# Import loguru for advanced logging
try:
    from loguru import logger
except ImportError:
    print("❌ loguru not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "loguru"])
    from loguru import logger

def initialize_logging() -> Path:
    """Initialize comprehensive logging system with file outputs"""
    
    # Create logs directory
    logs_dir = Path("data/output/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add application log file
    logger.add(
        logs_dir / "pipeline_application.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="10 MB",
        retention="30 days"
    )
    
    # Add error log file
    logger.add(
        logs_dir / "pipeline_errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="5 MB",
        retention="30 days"
    )
    
    # Add operation-specific logs
    logger.add(
        logs_dir / "pipeline_operations.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="20 MB",
        retention="7 days"
    )
    
    # Add API operations log
    logger.add(
        logs_dir / "api_operations.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="10 MB",
        retention="14 days"
    )
    
    # Add database operations log
    logger.add(
        logs_dir / "database_operations.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="10 MB",
        retention="14 days"
    )
    
    # Add text processing log
    logger.add(
        logs_dir / "text_processing.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="15 MB",
        retention="7 days"
    )
    
    # Add embedding generation log
    logger.add(
        logs_dir / "embedding_generation.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="15 MB",
        retention="7 days"
    )
    
    # Add exam generation log
    logger.add(
        logs_dir / "exam_generation.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="10 MB",
        retention="7 days"
    )
    
    # Add performance log
    logger.add(
        logs_dir / "performance.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="5 MB",
        retention="7 days"
    )
    
    # Create session-specific log
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.add(
        logs_dir / f"session_{session_timestamp}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="50 MB"
    )
    
    logger.info("✅ Comprehensive logging system initialized")
    logger.info(f"📁 Log files location: {logs_dir}")
    
    return logs_dir

def log_system_information():
    """Log system information for debugging"""
    import platform
    import psutil
    
    logger.info("🖥️ System Information:")
    logger.info(f"   OS: {platform.system()} {platform.release()}")
    logger.info(f"   Python: {platform.python_version()}")
    logger.info(f"   CPU: {psutil.cpu_count()} cores")
    logger.info(f"   Memory: {psutil.virtual_memory().total // (1024**3)} GB")

def log_pipeline_start(operation_name: str, parameters: Dict[str, Any]):
    """Log the start of a pipeline operation"""
    logger.info(f"🚀 Starting pipeline operation: {operation_name}")
    logger.info(f"📋 Parameters: {json.dumps(parameters, indent=2, default=str)}")

def log_pipeline_end(operation_name: str, success: bool, duration: float, 
                    results: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
    """Log the end of a pipeline operation"""
    status = "✅ SUCCESS" if success else "❌ FAILED"
    logger.info(f"{status} Pipeline operation completed: {operation_name}")
    logger.info(f"⏱️ Duration: {duration:.2f} seconds")
    
    if results:
        logger.info(f"📊 Results: {json.dumps(results, indent=2, default=str)}")
    
    if error:
        logger.error(f"💥 Error: {error}")

def create_operation_context(operation_name: str):
    """Decorator to create operation context with logging"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"🎯 Entering operation context: {operation_name}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"✅ Exiting operation context: {operation_name}")
                return result
            except Exception as e:
                logger.error(f"❌ Operation context failed: {operation_name} - {e}")
                raise
        return wrapper
    return decorator

def get_log_stats() -> Dict[str, Any]:
    """Get statistics about logging operations"""
    logs_dir = Path("data/output/logs")
    
    if not logs_dir.exists():
        return {"error": "Logs directory not found"}
    
    log_files = list(logs_dir.glob("*.log"))
    total_size = sum(f.stat().st_size for f in log_files)
    
    return {
        "total_log_files": len(log_files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "logs_directory": str(logs_dir),
        "files": [f.name for f in log_files]
    }

# Additional utility functions for compatibility
def setup_logging():
    """Alternative initialization function"""
    return initialize_logging()

def get_logger(name: str = __name__):
    """Get a logger instance"""
    return logger

# Export commonly used functions
__all__ = [
    'initialize_logging',
    'log_system_information', 
    'log_pipeline_start',
    'log_pipeline_end',
    'create_operation_context',
    'get_log_stats',
    'setup_logging',
    'get_logger'
]
