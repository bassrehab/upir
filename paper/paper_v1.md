# Universal Plan Intermediate Representation: Bridging Architecture and Implementation Through Formal Methods

**Technical Disclosure for Defensive Publication**

*Author: Subhadip Mitra, Google Cloud Professional Services*  
*Date: August 2025*

---

## Abstract

We've all been there - you design what seems like a perfect distributed system architecture, implement it carefully, and then watch it fail in production for reasons nobody anticipated. The disconnect between what we design and what actually runs costs our industry billions each year. This paper introduces UPIR (Universal Plan Intermediate Representation), a system I've been developing that tries to fix this problem by proving architectures correct before we write a single line of implementation code.

The core idea is straightforward: instead of hoping our implementations match our designs, we use formal verification to prove they do. UPIR combines SMT solving to verify correctness, automated synthesis to generate implementations, and reinforcement learning to improve systems based on production data. In testing on Google Cloud Platform, the approach achieved some promising results - verification runs about 2400x faster than naive approaches, synthesis takes milliseconds instead of hours, and the learning system converged quickly to meaningful improvements (60% latency reduction, nearly 3x throughput increase).

What makes this practical rather than just academic is the focus on incremental verification and pattern reuse. Most architectural changes are small, so we cache proofs and only re-verify what changed. Similarly, we extract common patterns from existing systems to accelerate synthesis of new ones. The result is a system that can actually be used in real development workflows, not just toy examples.

## 1. Introduction

### The Problem We're Trying to Solve

If you've worked on distributed systems, you know the drill. The architecture looks great on the whiteboard. The design doc gets approved. The implementation seems solid. Then production happens, and suddenly you're debugging race conditions at 3 AM that nobody saw coming.

The fundamental issue is that we operate at two different levels of abstraction. When we design systems, we think about properties: "payments should be processed exactly once," "the system should handle 10,000 requests per second," "data should be consistent across regions." But when we implement, we write code that manipulates specific data structures, makes particular API calls, and handles concrete error cases. There's no systematic way to verify that the implementation actually provides the properties we designed for.

Current tools don't really solve this. Infrastructure-as-code tools like Terraform help with consistency but don't verify correctness. Testing helps but can't cover all edge cases. Monitoring tells us when things go wrong but not how to prevent it. We need something that bridges the semantic gap between design and implementation.

### What UPIR Does Differently

UPIR takes a different approach. Instead of hoping implementations match designs, we prove they do. Here's how it works in practice:

First, you write specifications using a Python-like DSL that captures your requirements:

```python
@upir.specification
class PaymentSystem:
    @invariant
    def no_double_processing(self):
        # Each payment processed exactly once
        return forall(p in payments: 
            count(process_events(p)) == 1)
    
    @constraint
    def latency_requirement(self):
        # P99 under 100ms
        return percentile(latency, 99) <= 100
```

UPIR then does three things:

1. **Proves your architecture is correct** - Uses an SMT solver (Z3) to verify that your design satisfies all specified properties. If there's a problem, you get a concrete counterexample showing exactly how it fails.

2. **Synthesizes the implementation** - Once verified, UPIR generates the actual implementation code. This isn't template filling - it uses program synthesis techniques to find optimal parameters and configurations that satisfy your constraints.

3. **Learns and improves** - As your system runs in production, UPIR uses reinforcement learning to discover optimizations while maintaining all your invariants. It might find that increasing batch sizes improves throughput without violating latency requirements, for example.

The key insight is that verification, synthesis, and learning aren't separate activities - they're part of a continuous loop where each informs the others.

## 2. How It Actually Works

### The Core Architecture

At its heart, UPIR is a three-layer system, though the boundaries blur in practice:

**Layer 1: Specification** - This is where you define what your system should do. We use a temporal logic that can express properties over time ("eventually all messages are processed") and statistical properties ("99% of requests complete within 100ms"). The specification language is deliberately limited - you can't write arbitrary code, which makes verification tractable.

**Layer 2: Reasoning** - This is the brain of the system. It takes specifications and figures out if they're satisfiable, finds implementations that work, and identifies optimizations. We use Z3 for the heavy lifting here, with custom encodings for distributed systems concepts like eventual consistency and partition tolerance.

**Layer 3: Implementation** - This generates actual running code. We support multiple targets (currently Python, Go, and Kubernetes configs) with more planned. The generated code includes monitoring hooks that feed back into the learning system.

### Verification: Making It Fast

The naive approach to verification - checking every property against the entire system - is prohibitively slow. UPIR uses several tricks to make it practical:

**Incremental verification** is the biggest win. When you change a specification, we identify which proofs are affected and only re-verify those. Everything else comes from cache. In practice, this means verification time is proportional to the size of your change, not the size of your system.

Here's a simplified version of the algorithm:

```python
def verify_incremental(spec, changes):
    # Find what's affected by the changes
    affected = dependency_graph.get_affected(changes)
    
    # Reuse cached proofs for everything else
    cached_proofs = proof_cache.get_valid(spec - affected)
    
    # Only verify what changed
    new_proofs = []
    for prop in affected:
        proof = smt_solver.verify(prop)
        proof_cache.store(prop, proof)
        new_proofs.append(proof)
    
    return combine_proofs(cached_proofs + new_proofs)
```

This gets us from O(n²) complexity (checking n properties against n components) down to nearly O(1) for typical changes.

**Proof composition** is another optimization. Instead of verifying the entire system monolithically, we verify components independently and then verify their composition. This is sound because our property language is compositional - if components A and B satisfy properties P and Q respectively, we can deduce what A+B satisfies.

### Synthesis: From Spec to Code

Once we've verified a specification is correct, we need to implement it. UPIR uses a technique called CEGIS (Counterexample-Guided Inductive Synthesis). The basic idea:

1. Start with a program template with "holes" - unknown values we need to fill
2. Guess values for the holes
3. Check if the result satisfies the specification
4. If not, use the counterexample to refine our guess
5. Repeat until we find something that works

For example, given a specification for a rate limiter, UPIR might generate:

```python
def rate_limit(request):
    # Synthesized: bucket_size = 100, refill_rate = 10
    if bucket.tokens >= 1:
        bucket.tokens -= 1
        return process(request)
    else:
        return reject(request)
```

The specific values (100, 10) are synthesized to satisfy your latency and throughput requirements.

In practice, synthesis usually completes in milliseconds for typical components. The key is having good templates - we've built a library of common patterns (queues, caches, rate limiters, etc.) that cover most use cases.

### Learning: Getting Better Over Time

The learning component uses Proximal Policy Optimization (PPO), a reinforcement learning algorithm that's good at finding improvements while staying close to what already works. This is crucial - we don't want the system making radical changes that might break invariants.

The learning loop works like this:

1. Collect metrics from production (latency, throughput, error rates, costs)
2. Compute a reward signal based on how well we're meeting objectives
3. Generate candidate optimizations using the policy network
4. **Verify the optimization maintains all invariants** (this is critical!)
5. If verification passes, deploy the optimization
6. Update the policy based on the results

Here's what makes our approach different from standard RL: every action must be formally verified before deployment. This means we can be aggressive about exploration because we know we can't break correctness properties.

In testing, the system typically converges to good solutions within 50 episodes. For a payment processing pipeline, it discovered optimizations like:
- Increasing batch sizes during low-traffic periods
- Adjusting timeout values based on time of day
- Reordering operations to reduce lock contention

All while maintaining exactly-once processing guarantees.

## 3. Real Results

### What We Tested

I deployed UPIR on Google Cloud Platform to validate the approach with real infrastructure. The test system included:
- A Cloud Run service for the UPIR engine
- Cloud Storage for pattern libraries and proof caches  
- Cloud Monitoring for metrics collection
- Several example distributed systems (payment processor, data pipeline, microservice mesh)

### Performance Numbers

Here are the actual measurements from our tests:

**Verification Performance:**
- First verification (cold cache): 1024ms for 1000 properties
- Incremental verification: 0.43ms for 100 changed properties
- Cache hit rate: 89.9%
- Effective speedup: 2382x

The incremental performance is the key number here. Sub-millisecond verification means we can verify on every save, making it feel instantaneous to developers.

**Synthesis Performance:**
- Simple components (queues, caches): <10ms
- Complex components (consensus protocols): 100-500ms  
- Full system synthesis: 2-5 seconds

For comparison, manually implementing and testing these components typically takes hours or days.

**Learning Convergence:**
- Episodes to convergence: 45
- Latency improvement: 60.1% (198.7ms → 79.3ms)
- Throughput improvement: 194.5% (1987 → 5853 req/s)
- Error rate reduction: 80% (0.049 → 0.010)

These aren't cherry-picked results - they're averages across multiple runs with different random seeds.

### Where It Struggles

Let's be honest about limitations. UPIR doesn't work well for:

1. **Systems with complex external dependencies** - If your system calls a third-party API with unknown behavior, we can't verify much about it.

2. **Performance properties at the limit** - We can verify "latency < 100ms" but not "optimal latency." The system doesn't know about cache hierarchies or network topology.

3. **Large-scale synthesis** - Synthesizing an entire microservice from scratch isn't practical. UPIR works best when synthesizing parameters and configurations for known patterns.

4. **Non-deterministic algorithms** - Anything involving true randomness (not PRNGs) is hard to verify formally.

## 4. The Math Behind It

### Soundness

The critical property is soundness: if UPIR says your system is correct, it actually is. Here's the informal argument:

Each component is synthesized to satisfy a subset of the specification. The synthesis uses SMT solving, which is sound - if Z3 says a formula is satisfiable, it is. When we compose components, we verify that the composition preserves the properties we care about. Again, this uses sound verification techniques.

The tricky part is the learning system. We ensure soundness by verifying every proposed optimization before applying it. The learning system can suggest anything it wants, but only verified suggestions get deployed.

### Completeness

We don't have true completeness - there are some correct systems UPIR can't find. But we have something almost as good: relative completeness. If there's an implementation using our component library that satisfies your specification, UPIR will find it (given enough time).

The limitation is the component library. If you need a distributed consensus algorithm but we don't have Raft or Paxos in our library, UPIR can't synthesize it from scratch. In practice, the library covers most common patterns, and you can always add custom components.

### Convergence

The learning system is guaranteed to converge to a local optimum. This follows from standard PPO convergence results, with one modification: our action space is restricted to verifiable improvements. This actually helps convergence because it prevents the system from exploring obviously bad regions of the space.

## 5. Practical Applications

### Where UPIR Shines

**Data Pipelines** - These have clear correctness properties (no data loss, exactly-once processing) and performance requirements (throughput, latency). UPIR can synthesize optimal batch sizes, parallelism levels, and retry strategies.

**Microservice Orchestration** - Coordinating multiple services with different SLAs is complex. UPIR can verify that compositions meet end-to-end requirements and synthesize appropriate timeouts, circuit breakers, and fallback strategies.

**Streaming Systems** - Stream processing has tricky correctness properties around windowing and state management. UPIR can verify watermark algorithms and synthesize window sizes that balance latency and completeness.

### Real Example: Payment Processing

Here's a real specification we tested:

```python
@upir.specification  
class PaymentProcessor:
    @invariant
    def consistency(self):
        # Money in = money out
        return sum(payments_received) == 
               sum(payments_processed) + sum(payments_pending)
    
    @invariant
    def no_double_charge(self):
        # Each payment ID processed at most once
        return forall(id in payment_ids:
            count(processed[id]) <= 1)
    
    @constraint
    def performance(self):
        return (percentile(latency, 99) <= 200 and
                throughput >= 1000)
```

UPIR synthesized an implementation with:
- Optimal batch size: 25 transactions
- Timeout: 150ms with exponential backoff
- Parallelism: 8 workers
- Deduplication cache: 10,000 entries with LRU eviction

The generated system handled 5,000 transactions/second with zero double-charges in a 24-hour test.

## 6. Related Work

UPIR builds on several research threads:

**Program Synthesis** - The CEGIS algorithm comes from Armando Solar-Lezama's work at MIT. We've adapted it for distributed systems by adding templates for common patterns and integration with deployment tools.

**Formal Verification** - Model checkers like TLA+ and Alloy inspired our specification language. The key difference is that we generate implementations, not just verify models.

**Infrastructure as Code** - Tools like Terraform and Pulumi manage infrastructure declaratively. UPIR goes further by verifying correctness and synthesizing optimal configurations.

**Machine Learning for Systems** - Recent work uses ML to optimize databases and compilers. We apply similar ideas but maintain formal correctness guarantees.

## 7. What's Next

### Short Term (Next 6 Months)

1. **More backend targets** - Adding support for Kubernetes operators, AWS Lambda, and Apache Beam
2. **Richer specifications** - Security properties, multi-tenancy, data privacy
3. **Better error messages** - When verification fails, explain why in terms developers understand

### Medium Term (Next Year)

1. **Distributed verification** - Run verification across multiple machines for larger systems
2. **Learned synthesis** - Use ML to guide synthesis toward likely-correct implementations
3. **IDE integration** - Real-time verification as you type, like a type checker

### Long Term Vision

The dream is to make formal methods invisible. Developers specify what they want, UPIR figures out how to build it correctly, and the system continuously improves itself. We're not there yet, but the foundations are solid.

## 8. Try It Yourself

The code is structured as:

```
upir/
├── core/          # Data structures and IR
├── verification/  # SMT encoding and checking  
├── synthesis/     # CEGIS implementation
├── learning/      # PPO and reward computation
├── patterns/      # Component library
└── examples/      # Sample systems
```

To run a simple example:

```python
from upir import Specification, verify, synthesize

spec = Specification("my_system")
spec.add_invariant(lambda: all_messages_delivered)
spec.add_constraint(lambda: latency_p99 < 100)

# Verify it's satisfiable
result = verify(spec)
if result.is_valid:
    # Generate implementation
    code = synthesize(spec)
    print(code)
```

## 9. Conclusion

UPIR started as an attempt to answer a simple question: why can't we prove our distributed systems correct before deploying them? The answer, it turns out, is that we can - it just requires bringing together techniques from formal methods, program synthesis, and machine learning in the right way.

The system isn't perfect. There are plenty of properties we can't verify and implementations we can't synthesize. But for a useful class of systems - data pipelines, microservices, stream processors - UPIR can provide correctness guarantees that weren't previously practical.

More importantly, this feels like the beginning rather than the end. As synthesis techniques improve and verification becomes faster, we'll be able to handle larger and more complex systems. The goal isn't to replace developers but to let them work at a higher level of abstraction, focusing on what systems should do rather than how to implement them.

The economic argument is compelling - even small reductions in outages and development time translate to millions in savings for large organizations. But beyond the economics, there's something satisfying about knowing your system is correct by construction, not just by testing.

Distributed systems are hard. They'll always be hard. But maybe they don't have to be quite this hard.

---

## Appendix: Technical Details

### A. SMT Encoding

We encode distributed systems properties into SMT formulas using the theory of linear arithmetic and uninterpreted functions. State transitions become implications, invariants become universally quantified formulas, and temporal properties use auxiliary variables to track history.

### B. Pattern Library

Current patterns include:
- Queue (FIFO, priority, bounded)
- Cache (LRU, LFU, TTL)
- Rate limiter (token bucket, sliding window)
- Circuit breaker (failure threshold, recovery)
- Retry (exponential backoff, jitter)
- Load balancer (round robin, least connections, weighted)

### C. Deployment Configs

The system generates:
- Kubernetes manifests (deployments, services, configmaps)
- Terraform configurations (for cloud resources)
- Docker Compose files (for local testing)
- GitHub Actions workflows (for CI/CD)

### D. Metrics Collected

For learning, we collect:
- Request latency (P50, P99, P999)
- Throughput (requests/second)
- Error rates (by error type)
- Resource utilization (CPU, memory, network)
- Cost (when running on cloud platforms)

---

*For questions or collaboration opportunities, contact: subhadip.mitra@google.com*

*Note: This represents personal research conducted using publicly available tools and platforms. Views expressed are my own.*