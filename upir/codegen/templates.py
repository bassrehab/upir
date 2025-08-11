"""
Code templates for common distributed system patterns.
"""

from typing import Dict, Any, List
from .generator import Template
import textwrap


class QueueWorkerTemplate(Template):
    """Template for queue worker pattern."""
    
    def __init__(self):
        super().__init__("queue_worker")
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'batch_size': {'type': 'int', 'min': 1, 'max': 1000, 'default': 10},
            'timeout_ms': {'type': 'int', 'min': 100, 'max': 30000, 'default': 5000},
            'max_retries': {'type': 'int', 'min': 0, 'max': 10, 'default': 3},
            'workers': {'type': 'int', 'min': 1, 'max': 100, 'default': 4}
        }
    
    def get_constraints(self) -> List:
        return [
            lambda v: v['batch_size'] * v['workers'] <= 1000,  # Total throughput limit
            lambda v: v['timeout_ms'] >= v['batch_size'] * 10  # Timeout scales with batch
        ]
    
    def generate_python(self, params: Dict[str, Any]) -> str:
        return textwrap.dedent(f'''
        class QueueWorker:
            """Auto-generated queue worker with optimized parameters."""
            
            def __init__(self, queue_name: str):
                self.queue_name = queue_name
                self.batch_size = {params.get('batch_size', 10)}
                self.timeout_ms = {params.get('timeout_ms', 5000)}
                self.max_retries = {params.get('max_retries', 3)}
                self.workers = {params.get('workers', 4)}
                self.queue = queue.Queue()
                self.running = False
            
            def process_batch(self, items: List[Any]) -> List[Any]:
                """Process a batch of items."""
                results = []
                for item in items:
                    try:
                        result = self.process_item(item)
                        results.append(result)
                        self.queue.task_done()
                    except Exception as e:
                        logging.error(f"Failed to process item: {{e}}")
                        if self.should_retry(item):
                            self.queue.put(item)
                return results
            
            def process_item(self, item: Any) -> Any:
                """Override this method with actual processing logic."""
                raise NotImplementedError
            
            def should_retry(self, item: Any) -> bool:
                """Determine if item should be retried."""
                retry_count = getattr(item, 'retry_count', 0)
                return retry_count < self.max_retries
            
            def run(self):
                """Main worker loop."""
                self.running = True
                while self.running:
                    batch = []
                    deadline = time.time() + (self.timeout_ms / 1000.0)
                    
                    while len(batch) < self.batch_size and time.time() < deadline:
                        try:
                            timeout = max(0, deadline - time.time())
                            item = self.queue.get(timeout=timeout)
                            batch.append(item)
                        except queue.Empty:
                            break
                    
                    if batch:
                        self.process_batch(batch)
            
            def stop(self):
                """Stop the worker."""
                self.running = False
        ''').strip()
    
    def generate_go(self, params: Dict[str, Any]) -> str:
        return textwrap.dedent(f'''
        type QueueWorker struct {{
            QueueName  string
            BatchSize  int
            TimeoutMs  int
            MaxRetries int
            Workers    int
            queue      chan interface{{}}
            running    bool
        }}
        
        func NewQueueWorker(queueName string) *QueueWorker {{
            return &QueueWorker{{
                QueueName:  queueName,
                BatchSize:  {params.get('batch_size', 10)},
                TimeoutMs:  {params.get('timeout_ms', 5000)},
                MaxRetries: {params.get('max_retries', 3)},
                Workers:    {params.get('workers', 4)},
                queue:      make(chan interface{{}}, 1000),
                running:    false,
            }}
        }}
        
        func (w *QueueWorker) ProcessBatch(items []interface{{}}) {{
            for _, item := range items {{
                if err := w.ProcessItem(item); err != nil {{
                    fmt.Printf("Failed to process item: %v\\n", err)
                    if w.ShouldRetry(item) {{
                        w.queue <- item
                    }}
                }}
            }}
        }}
        
        func (w *QueueWorker) Run(ctx context.Context) {{
            w.running = true
            for w.running {{
                batch := make([]interface{{}}, 0, w.BatchSize)
                timeout := time.After(time.Duration(w.TimeoutMs) * time.Millisecond)
                
                for len(batch) < w.BatchSize {{
                    select {{
                    case item := <-w.queue:
                        batch = append(batch, item)
                    case <-timeout:
                        goto ProcessBatch
                    case <-ctx.Done():
                        w.running = false
                        return
                    }}
                }}
                
            ProcessBatch:
                if len(batch) > 0 {{
                    w.ProcessBatch(batch)
                }}
            }}
        }}
        ''').strip()
    
    def generate_javascript(self, params: Dict[str, Any]) -> str:
        return textwrap.dedent(f'''
        class QueueWorker {{
            constructor(queueName) {{
                this.queueName = queueName;
                this.batchSize = {params.get('batch_size', 10)};
                this.timeoutMs = {params.get('timeout_ms', 5000)};
                this.maxRetries = {params.get('max_retries', 3)};
                this.workers = {params.get('workers', 4)};
                this.queue = [];
                this.running = false;
            }}
            
            async processBatch(items) {{
                const results = [];
                for (const item of items) {{
                    try {{
                        const result = await this.processItem(item);
                        results.push(result);
                    }} catch (error) {{
                        console.error(`Failed to process item: ${{error}}`);
                        if (this.shouldRetry(item)) {{
                            this.queue.push(item);
                        }}
                    }}
                }}
                return results;
            }}
            
            async run() {{
                this.running = true;
                while (this.running) {{
                    const batch = [];
                    const deadline = Date.now() + this.timeoutMs;
                    
                    while (batch.length < this.batchSize && Date.now() < deadline) {{
                        if (this.queue.length > 0) {{
                            batch.push(this.queue.shift());
                        }} else {{
                            await new Promise(r => setTimeout(r, 10));
                        }}
                    }}
                    
                    if (batch.length > 0) {{
                        await this.processBatch(batch);
                    }}
                }}
            }}
        }}
        ''').strip()


class RateLimiterTemplate(Template):
    """Template for rate limiter pattern."""
    
    def __init__(self):
        super().__init__("rate_limiter")
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'requests_per_second': {'type': 'int', 'min': 1, 'max': 100000},
            'burst_size': {'type': 'int', 'min': 1, 'max': 1000},
            'window_ms': {'type': 'int', 'min': 100, 'max': 60000}
        }
    
    def get_constraints(self) -> List:
        return [
            lambda v: v['burst_size'] >= v['requests_per_second'] / 10,
            lambda v: v['window_ms'] >= 1000 / v['requests_per_second']
        ]
    
    def generate_python(self, params: Dict[str, Any]) -> str:
        return textwrap.dedent(f'''
        class RateLimiter:
            """Token bucket rate limiter with synthesized parameters."""
            
            def __init__(self):
                self.rate = {params.get('requests_per_second', 100)}
                self.burst = {params.get('burst_size', 10)}
                self.window_ms = {params.get('window_ms', 1000)}
                self.tokens = self.burst
                self.last_refill = time.time()
                self.lock = Lock()
            
            def allow_request(self) -> bool:
                """Check if request is allowed under rate limit."""
                with self.lock:
                    self._refill()
                    if self.tokens >= 1:
                        self.tokens -= 1
                        return True
                    return False
            
            def _refill(self):
                """Refill tokens based on elapsed time."""
                now = time.time()
                elapsed = now - self.last_refill
                tokens_to_add = elapsed * self.rate
                self.tokens = min(self.burst, self.tokens + tokens_to_add)
                self.last_refill = now
            
            def wait_if_needed(self) -> float:
                """Return seconds to wait before next request allowed."""
                with self.lock:
                    if self.tokens >= 1:
                        return 0
                    return (1 - self.tokens) / self.rate
        ''').strip()
    
    def generate_go(self, params: Dict[str, Any]) -> str:
        return textwrap.dedent(f'''
        type RateLimiter struct {{
            rate       int
            burst      int
            windowMs   int
            tokens     float64
            lastRefill time.Time
            mu         sync.Mutex
        }}
        
        func NewRateLimiter() *RateLimiter {{
            return &RateLimiter{{
                rate:       {params.get('requests_per_second', 100)},
                burst:      {params.get('burst_size', 10)},
                windowMs:   {params.get('window_ms', 1000)},
                tokens:     float64({params.get('burst_size', 10)}),
                lastRefill: time.Now(),
            }}
        }}
        
        func (r *RateLimiter) AllowRequest() bool {{
            r.mu.Lock()
            defer r.mu.Unlock()
            
            r.refill()
            if r.tokens >= 1 {{
                r.tokens--
                return true
            }}
            return false
        }}
        
        func (r *RateLimiter) refill() {{
            now := time.Now()
            elapsed := now.Sub(r.lastRefill).Seconds()
            tokensToAdd := elapsed * float64(r.rate)
            r.tokens = math.Min(float64(r.burst), r.tokens+tokensToAdd)
            r.lastRefill = now
        }}
        ''').strip()
    
    def generate_javascript(self, params: Dict[str, Any]) -> str:
        return textwrap.dedent(f'''
        class RateLimiter {{
            constructor() {{
                this.rate = {params.get('requests_per_second', 100)};
                this.burst = {params.get('burst_size', 10)};
                this.windowMs = {params.get('window_ms', 1000)};
                this.tokens = this.burst;
                this.lastRefill = Date.now();
            }}
            
            allowRequest() {{
                this.refill();
                if (this.tokens >= 1) {{
                    this.tokens--;
                    return true;
                }}
                return false;
            }}
            
            refill() {{
                const now = Date.now();
                const elapsed = (now - this.lastRefill) / 1000;
                const tokensToAdd = elapsed * this.rate;
                this.tokens = Math.min(this.burst, this.tokens + tokensToAdd);
                this.lastRefill = now;
            }}
        }}
        ''').strip()


class CircuitBreakerTemplate(Template):
    """Template for circuit breaker pattern."""
    
    def __init__(self):
        super().__init__("circuit_breaker")
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'failure_threshold': {'type': 'int', 'min': 1, 'max': 100},
            'recovery_timeout_ms': {'type': 'int', 'min': 1000, 'max': 60000},
            'half_open_requests': {'type': 'int', 'min': 1, 'max': 10}
        }
    
    def get_constraints(self) -> List:
        return [
            lambda v: v['recovery_timeout_ms'] >= v['failure_threshold'] * 100
        ]
    
    def generate_python(self, params: Dict[str, Any]) -> str:
        return textwrap.dedent(f'''
        class CircuitBreaker:
            """Circuit breaker with auto-tuned thresholds."""
            
            def __init__(self, name: str):
                self.name = name
                self.failure_threshold = {params.get('failure_threshold', 5)}
                self.recovery_timeout_ms = {params.get('recovery_timeout_ms', 5000)}
                self.half_open_requests = {params.get('half_open_requests', 3)}
                
                self.state = "CLOSED"
                self.failures = 0
                self.last_failure_time = None
                self.half_open_successes = 0
            
            def call(self, func, *args, **kwargs):
                """Execute function with circuit breaker protection."""
                if self.state == "OPEN":
                    if self._should_attempt_reset():
                        self.state = "HALF_OPEN"
                        self.half_open_successes = 0
                    else:
                        raise Exception(f"Circuit breaker {{self.name}} is OPEN")
                
                try:
                    result = func(*args, **kwargs)
                    self._on_success()
                    return result
                except Exception as e:
                    self._on_failure()
                    raise e
            
            def _on_success(self):
                """Handle successful call."""
                self.failures = 0
                if self.state == "HALF_OPEN":
                    self.half_open_successes += 1
                    if self.half_open_successes >= self.half_open_requests:
                        self.state = "CLOSED"
            
            def _on_failure(self):
                """Handle failed call."""
                self.failures += 1
                self.last_failure_time = time.time()
                if self.failures >= self.failure_threshold:
                    self.state = "OPEN"
            
            def _should_attempt_reset(self) -> bool:
                """Check if we should try half-open state."""
                return (time.time() - self.last_failure_time) * 1000 >= self.recovery_timeout_ms
        ''').strip()
    
    def generate_go(self, params: Dict[str, Any]) -> str:
        return textwrap.dedent(f'''
        type CircuitBreaker struct {{
            Name               string
            FailureThreshold   int
            RecoveryTimeoutMs  int
            HalfOpenRequests   int
            
            state              string
            failures           int
            lastFailureTime    time.Time
            halfOpenSuccesses  int
            mu                 sync.Mutex
        }}
        
        func NewCircuitBreaker(name string) *CircuitBreaker {{
            return &CircuitBreaker{{
                Name:              name,
                FailureThreshold:  {params.get('failure_threshold', 5)},
                RecoveryTimeoutMs: {params.get('recovery_timeout_ms', 5000)},
                HalfOpenRequests:  {params.get('half_open_requests', 3)},
                state:             "CLOSED",
            }}
        }}
        
        func (cb *CircuitBreaker) Call(fn func() error) error {{
            cb.mu.Lock()
            defer cb.mu.Unlock()
            
            if cb.state == "OPEN" {{
                if cb.shouldAttemptReset() {{
                    cb.state = "HALF_OPEN"
                    cb.halfOpenSuccesses = 0
                }} else {{
                    return fmt.Errorf("circuit breaker %s is OPEN", cb.Name)
                }}
            }}
            
            err := fn()
            if err == nil {{
                cb.onSuccess()
            }} else {{
                cb.onFailure()
            }}
            return err
        }}
        ''').strip()
    
    def generate_javascript(self, params: Dict[str, Any]) -> str:
        return textwrap.dedent(f'''
        class CircuitBreaker {{
            constructor(name) {{
                this.name = name;
                this.failureThreshold = {params.get('failure_threshold', 5)};
                this.recoveryTimeoutMs = {params.get('recovery_timeout_ms', 5000)};
                this.halfOpenRequests = {params.get('half_open_requests', 3)};
                
                this.state = "CLOSED";
                this.failures = 0;
                this.lastFailureTime = null;
                this.halfOpenSuccesses = 0;
            }}
            
            async call(func) {{
                if (this.state === "OPEN") {{
                    if (this.shouldAttemptReset()) {{
                        this.state = "HALF_OPEN";
                        this.halfOpenSuccesses = 0;
                    }} else {{
                        throw new Error(`Circuit breaker ${{this.name}} is OPEN`);
                    }}
                }}
                
                try {{
                    const result = await func();
                    this.onSuccess();
                    return result;
                }} catch (error) {{
                    this.onFailure();
                    throw error;
                }}
            }}
        }}
        ''').strip()


class RetryTemplate(Template):
    """Template for retry logic with exponential backoff."""
    
    def __init__(self):
        super().__init__("retry")
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'max_attempts': {'type': 'int', 'min': 1, 'max': 10},
            'initial_delay_ms': {'type': 'int', 'min': 10, 'max': 1000},
            'max_delay_ms': {'type': 'int', 'min': 100, 'max': 60000},
            'backoff_factor': {'type': 'float', 'min': 1.1, 'max': 3.0}
        }
    
    def get_constraints(self) -> List:
        return [
            lambda v: v['max_delay_ms'] >= v['initial_delay_ms'],
            lambda v: v['initial_delay_ms'] * (v['backoff_factor'] ** v['max_attempts']) <= v['max_delay_ms'] * 10
        ]
    
    def generate_python(self, params: Dict[str, Any]) -> str:
        return textwrap.dedent(f'''
        def retry_with_backoff(func):
            """Decorator for retry with exponential backoff."""
            @wraps(func)
            def wrapper(*args, **kwargs):
                max_attempts = {params.get('max_attempts', 3)}
                initial_delay_ms = {params.get('initial_delay_ms', 100)}
                max_delay_ms = {params.get('max_delay_ms', 5000)}
                backoff_factor = {params.get('backoff_factor', 2.0)}
                
                last_exception = None
                delay_ms = initial_delay_ms
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            jitter = random.uniform(0.8, 1.2)
                            sleep_time = min(delay_ms * jitter, max_delay_ms) / 1000.0
                            time.sleep(sleep_time)
                            delay_ms *= backoff_factor
                
                raise last_exception
            return wrapper
        ''').strip()
    
    def generate_go(self, params: Dict[str, Any]) -> str:
        return textwrap.dedent(f'''
        func RetryWithBackoff(fn func() error) error {{
            maxAttempts := {params.get('max_attempts', 3)}
            initialDelayMs := {params.get('initial_delay_ms', 100)}
            maxDelayMs := {params.get('max_delay_ms', 5000)}
            backoffFactor := {params.get('backoff_factor', 2.0)}
            
            var lastErr error
            delayMs := float64(initialDelayMs)
            
            for attempt := 0; attempt < maxAttempts; attempt++ {{
                if err := fn(); err == nil {{
                    return nil
                }} else {{
                    lastErr = err
                    if attempt < maxAttempts-1 {{
                        jitter := 0.8 + rand.Float64()*0.4
                        sleepMs := math.Min(delayMs*jitter, float64(maxDelayMs))
                        time.Sleep(time.Duration(sleepMs) * time.Millisecond)
                        delayMs *= backoffFactor
                    }}
                }}
            }}
            
            return lastErr
        }}
        ''').strip()
    
    def generate_javascript(self, params: Dict[str, Any]) -> str:
        return textwrap.dedent(f'''
        async function retryWithBackoff(func) {{
            const maxAttempts = {params.get('max_attempts', 3)};
            const initialDelayMs = {params.get('initial_delay_ms', 100)};
            const maxDelayMs = {params.get('max_delay_ms', 5000)};
            const backoffFactor = {params.get('backoff_factor', 2.0)};
            
            let lastError;
            let delayMs = initialDelayMs;
            
            for (let attempt = 0; attempt < maxAttempts; attempt++) {{
                try {{
                    return await func();
                }} catch (error) {{
                    lastError = error;
                    if (attempt < maxAttempts - 1) {{
                        const jitter = 0.8 + Math.random() * 0.4;
                        const sleepMs = Math.min(delayMs * jitter, maxDelayMs);
                        await new Promise(r => setTimeout(r, sleepMs));
                        delayMs *= backoffFactor;
                    }}
                }}
            }}
            
            throw lastError;
        }}
        ''').strip()


class CacheTemplate(Template):
    """Template for cache implementation."""
    
    def __init__(self):
        super().__init__("cache")
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'max_size': {'type': 'int', 'min': 10, 'max': 100000},
            'ttl_seconds': {'type': 'int', 'min': 1, 'max': 3600},
            'eviction_policy': {'type': 'str', 'options': ['lru', 'lfu', 'fifo']}
        }
    
    def get_constraints(self) -> List:
        return [
            lambda v: v['max_size'] >= 10,
            lambda v: v['ttl_seconds'] >= 1
        ]
    
    def generate_python(self, params: Dict[str, Any]) -> str:
        policy = params.get('eviction_policy', 'lru')
        if policy == 'lru':
            return textwrap.dedent(f'''
            class LRUCache:
                """LRU cache with TTL support."""
                
                def __init__(self):
                    self.max_size = {params.get('max_size', 1000)}
                    self.ttl_seconds = {params.get('ttl_seconds', 300)}
                    self.cache = OrderedDict()
                    self.timestamps = {{}}
                
                def get(self, key: str) -> Any:
                    """Get value from cache."""
                    if key not in self.cache:
                        return None
                    
                    # Check TTL
                    if time.time() - self.timestamps[key] > self.ttl_seconds:
                        del self.cache[key]
                        del self.timestamps[key]
                        return None
                    
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    return self.cache[key]
                
                def put(self, key: str, value: Any):
                    """Put value in cache."""
                    if key in self.cache:
                        del self.cache[key]
                    
                    self.cache[key] = value
                    self.timestamps[key] = time.time()
                    
                    # Evict if necessary
                    if len(self.cache) > self.max_size:
                        oldest = next(iter(self.cache))
                        del self.cache[oldest]
                        del self.timestamps[oldest]
                
                def clear(self):
                    """Clear all cache entries."""
                    self.cache.clear()
                    self.timestamps.clear()
            ''').strip()
        else:
            # Default to simple cache for other policies
            return textwrap.dedent(f'''
            class Cache:
                """Simple cache with TTL."""
                
                def __init__(self):
                    self.max_size = {params.get('max_size', 1000)}
                    self.ttl_seconds = {params.get('ttl_seconds', 300)}
                    self.cache = {{}}
                    self.timestamps = {{}}
                
                def get(self, key: str) -> Any:
                    if key not in self.cache:
                        return None
                    if time.time() - self.timestamps[key] > self.ttl_seconds:
                        del self.cache[key]
                        del self.timestamps[key]
                        return None
                    return self.cache[key]
                
                def put(self, key: str, value: Any):
                    if len(self.cache) >= self.max_size and key not in self.cache:
                        # Simple FIFO eviction
                        oldest = min(self.timestamps, key=self.timestamps.get)
                        del self.cache[oldest]
                        del self.timestamps[oldest]
                    
                    self.cache[key] = value
                    self.timestamps[key] = time.time()
            ''').strip()
    
    def generate_go(self, params: Dict[str, Any]) -> str:
        return textwrap.dedent(f'''
        type Cache struct {{
            maxSize     int
            ttlSeconds  int
            cache       map[string]interface{{}}
            timestamps  map[string]time.Time
            mu          sync.RWMutex
        }}
        
        func NewCache() *Cache {{
            return &Cache{{
                maxSize:    {params.get('max_size', 1000)},
                ttlSeconds: {params.get('ttl_seconds', 300)},
                cache:      make(map[string]interface{{}}),
                timestamps: make(map[string]time.Time),
            }}
        }}
        
        func (c *Cache) Get(key string) (interface{{}}, bool) {{
            c.mu.RLock()
            defer c.mu.RUnlock()
            
            val, exists := c.cache[key]
            if !exists {{
                return nil, false
            }}
            
            if time.Since(c.timestamps[key]).Seconds() > float64(c.ttlSeconds) {{
                delete(c.cache, key)
                delete(c.timestamps, key)
                return nil, false
            }}
            
            return val, true
        }}
        
        func (c *Cache) Put(key string, value interface{{}}) {{
            c.mu.Lock()
            defer c.mu.Unlock()
            
            if len(c.cache) >= c.maxSize && c.cache[key] == nil {{
                // Simple eviction - remove oldest
                var oldest string
                var oldestTime time.Time
                for k, t := range c.timestamps {{
                    if oldest == "" || t.Before(oldestTime) {{
                        oldest = k
                        oldestTime = t
                    }}
                }}
                delete(c.cache, oldest)
                delete(c.timestamps, oldest)
            }}
            
            c.cache[key] = value
            c.timestamps[key] = time.Now()
        }}
        ''').strip()
    
    def generate_javascript(self, params: Dict[str, Any]) -> str:
        return textwrap.dedent(f'''
        class Cache {{
            constructor() {{
                this.maxSize = {params.get('max_size', 1000)};
                this.ttlSeconds = {params.get('ttl_seconds', 300)};
                this.cache = new Map();
                this.timestamps = new Map();
            }}
            
            get(key) {{
                if (!this.cache.has(key)) {{
                    return null;
                }}
                
                const timestamp = this.timestamps.get(key);
                if ((Date.now() - timestamp) / 1000 > this.ttlSeconds) {{
                    this.cache.delete(key);
                    this.timestamps.delete(key);
                    return null;
                }}
                
                return this.cache.get(key);
            }}
            
            put(key, value) {{
                if (this.cache.size >= this.maxSize && !this.cache.has(key)) {{
                    // Simple FIFO eviction
                    const firstKey = this.cache.keys().next().value;
                    this.cache.delete(firstKey);
                    this.timestamps.delete(firstKey);
                }}
                
                this.cache.set(key, value);
                this.timestamps.set(key, Date.now());
            }}
        }}
        ''').strip()


class LoadBalancerTemplate(Template):
    """Template for load balancer."""
    
    def __init__(self):
        super().__init__("load_balancer")
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'algorithm': {'type': 'str', 'options': ['round_robin', 'least_connections', 'random', 'weighted']},
            'health_check_interval_ms': {'type': 'int', 'min': 1000, 'max': 60000},
            'unhealthy_threshold': {'type': 'int', 'min': 1, 'max': 10}
        }
    
    def get_constraints(self) -> List:
        return [
            lambda v: v['health_check_interval_ms'] >= 1000
        ]
    
    def generate_python(self, params: Dict[str, Any]) -> str:
        algorithm = params.get('algorithm', 'round_robin')
        
        if algorithm == 'round_robin':
            selector = '''
            def select_backend(self) -> str:
                """Select next backend using round robin."""
                if not self.healthy_backends:
                    raise Exception("No healthy backends available")
                
                backend = self.healthy_backends[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.healthy_backends)
                return backend
            '''
        elif algorithm == 'random':
            selector = '''
            def select_backend(self) -> str:
                """Select random healthy backend."""
                if not self.healthy_backends:
                    raise Exception("No healthy backends available")
                return random.choice(self.healthy_backends)
            '''
        else:
            selector = '''
            def select_backend(self) -> str:
                """Select backend with least connections."""
                if not self.healthy_backends:
                    raise Exception("No healthy backends available")
                return min(self.healthy_backends, key=lambda b: self.connections.get(b, 0))
            '''
        
        return textwrap.dedent(f'''
        class LoadBalancer:
            """Load balancer with health checking."""
            
            def __init__(self, backends: List[str]):
                self.backends = backends
                self.healthy_backends = backends.copy()
                self.health_check_interval_ms = {params.get('health_check_interval_ms', 5000)}
                self.unhealthy_threshold = {params.get('unhealthy_threshold', 3)}
                self.current_index = 0
                self.connections = {{}}
                self.failure_counts = {{b: 0 for b in backends}}
            
            {selector}
            
            def mark_failure(self, backend: str):
                """Mark a backend as failed."""
                self.failure_counts[backend] += 1
                if self.failure_counts[backend] >= self.unhealthy_threshold:
                    if backend in self.healthy_backends:
                        self.healthy_backends.remove(backend)
            
            def mark_success(self, backend: str):
                """Mark a backend as successful."""
                self.failure_counts[backend] = 0
                if backend not in self.healthy_backends:
                    self.healthy_backends.append(backend)
        ''').strip()
    
    def generate_go(self, params: Dict[str, Any]) -> str:
        return textwrap.dedent(f'''
        type LoadBalancer struct {{
            backends              []string
            healthyBackends       []string
            healthCheckIntervalMs int
            unhealthyThreshold    int
            currentIndex          int
            failureCounts         map[string]int
            mu                    sync.RWMutex
        }}
        
        func NewLoadBalancer(backends []string) *LoadBalancer {{
            healthy := make([]string, len(backends))
            copy(healthy, backends)
            
            return &LoadBalancer{{
                backends:              backends,
                healthyBackends:       healthy,
                healthCheckIntervalMs: {params.get('health_check_interval_ms', 5000)},
                unhealthyThreshold:    {params.get('unhealthy_threshold', 3)},
                failureCounts:         make(map[string]int),
            }}
        }}
        
        func (lb *LoadBalancer) SelectBackend() (string, error) {{
            lb.mu.RLock()
            defer lb.mu.RUnlock()
            
            if len(lb.healthyBackends) == 0 {{
                return "", errors.New("no healthy backends available")
            }}
            
            backend := lb.healthyBackends[lb.currentIndex]
            lb.currentIndex = (lb.currentIndex + 1) % len(lb.healthyBackends)
            return backend, nil
        }}
        ''').strip()
    
    def generate_javascript(self, params: Dict[str, Any]) -> str:
        return textwrap.dedent(f'''
        class LoadBalancer {{
            constructor(backends) {{
                this.backends = backends;
                this.healthyBackends = [...backends];
                this.healthCheckIntervalMs = {params.get('health_check_interval_ms', 5000)};
                this.unhealthyThreshold = {params.get('unhealthy_threshold', 3)};
                this.currentIndex = 0;
                this.failureCounts = {{}};
                backends.forEach(b => this.failureCounts[b] = 0);
            }}
            
            selectBackend() {{
                if (this.healthyBackends.length === 0) {{
                    throw new Error("No healthy backends available");
                }}
                
                const backend = this.healthyBackends[this.currentIndex];
                this.currentIndex = (this.currentIndex + 1) % this.healthyBackends.length;
                return backend;
            }}
            
            markFailure(backend) {{
                this.failureCounts[backend]++;
                if (this.failureCounts[backend] >= this.unhealthyThreshold) {{
                    const index = this.healthyBackends.indexOf(backend);
                    if (index > -1) {{
                        this.healthyBackends.splice(index, 1);
                    }}
                }}
            }}
        }}
        ''').strip()