from ratelimit import limits, sleep_and_retry
from tenacity import retry, wait_exponential, stop_after_attempt
from config.settings import RATE_LIMIT_RPM

# Decorator for rate limiting and retrying API calls
def gemini_rate_limiter(func):
    @sleep_and_retry
    @limits(calls=RATE_LIMIT_RPM, period=60)
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
