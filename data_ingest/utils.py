import time
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import requests

from .config import RATE_LIMIT_SLEEP, REQUEST_TIMEOUT

def sleep_rate_limit():
    time.sleep(RATE_LIMIT_SLEEP)

@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5), retry=retry_if_exception_type(requests.exceptions.RequestException))
def safe_get(url, headers=None, params=None):
    """GET with retries and conservative rate-limit sleep."""
    r = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    if r.status_code >= 500:
        # provoque un retry via exception
        r.raise_for_status()
    # backoff in client side to be polite
    sleep_rate_limit()
    return r
