import fcntl
import os
from pathlib import Path
from contextlib import contextmanager
from loguru import logger

@contextmanager
def pipeline_lock():
    """Ensure only one pipeline instance runs at a time"""
    lock_file = Path("data/output/pipeline.lock")
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(lock_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            f.write(f"PID: {os.getpid()}\n")
            logger.info("🔒 Pipeline lock acquired")
            yield
    except BlockingIOError:
        logger.error("❌ Another pipeline instance is already running")
        raise RuntimeError("Pipeline already in progress - wait for current pipeline to complete")
    finally:
        if lock_file.exists():
            lock_file.unlink()
        logger.info("🔓 Pipeline lock released")
