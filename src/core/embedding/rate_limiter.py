from ratelimit import limits, sleep_and_retry
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import google.api_core.exceptions
import time
import threading
from typing import Callable, Any
from loguru import logger

class APIRequestQueue:
    """Thread-safe request queue for API rate limiting"""
    def __init__(self, requests_per_minute: int = 12):
        self.rpm_limit = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute  # Seconds between requests
        self.last_request_time = 0
        self.request_lock = threading.Lock()
        
    def execute_request(self, api_function: Callable, *args, **kwargs) -> Any:
        """Execute API request with proper rate limiting"""
        with self.request_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                logger.info(f"⏱️ Rate limiting: waiting {sleep_time:.2f}s")
                time.sleep(sleep_time)
            
            try:
                result = api_function(*args, **kwargs)
                self.last_request_time = time.time()
                return result
            except Exception as e:
                self.last_request_time = time.time()  # Still update time on failure
                raise

# Global request queue instance
_global_request_queue = APIRequestQueue(requests_per_minute=12)

def gemini_rate_limiter(func):
    """Enhanced rate limiter with multiple protection layers"""
    @sleep_and_retry
    @limits(calls=12, period=60)  # Conservative: 12 requests per minute
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=5),
        stop=stop_after_attempt(2),
        retry=retry_if_exception_type((
            google.api_core.exceptions.InternalServerError,  # 500 errors
            google.api_core.exceptions.ServiceUnavailable,   # 503 errors
            google.api_core.exceptions.TooManyRequests,      # 429 errors
            TimeoutError  # Timeout errors
        )),
        reraise=True
    )
    def wrapper(*args, **kwargs):
        # Use global request queue for additional protection
        return _global_request_queue.execute_request(func, *args, **kwargs)
    
    return wrapper

def batch_rate_limiter(batch_size: int = 5):
    """Special rate limiter for batch operations"""
    @sleep_and_retry
    @limits(calls=batch_size, period=60)
    def limiter(func):
        return func
    return limiter
