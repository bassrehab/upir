"""
UPIR-Generated Rate Limiter
Pattern: rate_limiter
Generated at: 2025-08-11 16:06:41
"""

import time
from threading import Lock

class RateLimiter:
    """Token bucket rate limiter with optimized parameters."""
    
    def __init__(self):
        # UPIR-optimized parameters
        self.rate = 1000  # requests per second
        self.burst_size = 100  # max burst
        self.tokens = self.burst_size
        self.last_refill = time.time()
        self.lock = Lock()
        
    def allow_request(self):
        """Check if request is allowed."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            
            # Refill tokens
            tokens_to_add = elapsed * self.rate
            self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
            self.last_refill = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

# UPIR Verified Properties: rate_limiting, burst_control
