"""
Logging configuration for the gemini-embedding-model project
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


def setup_logging(
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up comprehensive logging for the application
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        log_file: Custom log file path
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    log_dir = Path("data/output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Default log file name with timestamp
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"conversion_{timestamp}.log"
    
    # Default format string
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_to_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Create application-specific logger
    app_logger = logging.getLogger("gemini_embedding_model")
    app_logger.setLevel(level)
    
    # Log the setup completion
    app_logger.info(f"Logging initialized - Level: {logging.getLevelName(level)}")
    if log_to_file:
        app_logger.info(f"Log file: {log_file}")
    
    return app_logger


def setup_conversion_logging(session_id: str) -> logging.Logger:
    """
    Set up logging specifically for PDF conversion operations
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Configured conversion logger
    """
    log_dir = Path("data/output/logs/conversion")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"conversion_{session_id}.log"
    
    return setup_logging(
        level=logging.INFO,
        log_file=str(log_file),
        format_string="%(asctime)s - %(levelname)s - %(message)s"
    )


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger instance
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # If no handlers are set, use basic configuration
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    return logger


def setup_module_loggers(base_level: int = logging.INFO) -> Dict[str, logging.Logger]:
    """
    Set up loggers for all modules in the project
    
    Args:
        base_level: Base logging level for all modules
        
    Returns:
        Dictionary of module name to logger mappings
    """
    modules = [
        "conversion",
        "embedding", 
        "generation",
        "storage",
        "text",
        "utils"
    ]
    
    loggers = {}
    
    for module in modules:
        logger_name = f"gemini_embedding_model.{module}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(base_level)
        loggers[module] = logger
    
    return loggers


def log_system_info():
    """Log system and environment information"""
    logger = logging.getLogger("gemini_embedding_model.system")
    
    logger.info("=== System Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Environment variables:")
    
    # Log relevant environment variables (without exposing sensitive data)
    env_vars = ["MARKERPDF_API_KEY", "PYTHONPATH", "PATH"]
    for var in env_vars:
        value = os.getenv(var, "Not set")
        if "API_KEY" in var and value != "Not set":
            # Mask API keys for security
            masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            logger.info(f"  {var}: {masked_value}")
        else:
            logger.info(f"  {var}: {value}")
    
    logger.info("===========================")


def configure_third_party_loggers():
    """Configure logging for third-party libraries"""
    # Reduce noise from requests library
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    # Reduce noise from other common libraries
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


# Module-level setup when imported
def _initialize_logging():
    """Initialize logging when module is imported"""
    # Set up basic logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Configure third-party loggers
    configure_third_party_loggers()


# Auto-initialize when module is imported
_initialize_logging()
