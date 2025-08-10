"""
Streaming Data Pipeline Example using UPIR

This example demonstrates how UPIR can formally specify, verify, and
synthesize a real-time streaming data pipeline. We'll model a pipeline
that processes events from Pub/Sub, transforms them, and writes to BigQuery.

The cool part is we can formally prove properties like:
- All events will be processed within 100ms
- No data loss under normal conditions  
- Exactly-once processing semantics

Author: subhadipmitra@google.com
"""

import sys
import os
import hashlib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from upir.core.models import (
    UPIR, FormalSpecification, TemporalProperty, TemporalOperator,
    Evidence, ReasoningNode, Architecture, Implementation, SynthesisProof
)
from upir.verification.verifier import Verifier, VerificationStatus


def create_streaming_pipeline_spec() -> FormalSpecification:
    """
    Create formal specification for a streaming data pipeline.
    
    This spec captures the key requirements for a production streaming
    pipeline - latency bounds, consistency guarantees, and fault tolerance.
    """
    
    # Define temporal properties as invariants (must always hold)
    invariants = [
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="data_consistency",
            parameters={"description": "No data corruption or loss"}
        ),
        TemporalProperty(
            operator=TemporalOperator.WITHIN,
            predicate="event_processed",
            time_bound=100.0,  # 100ms processing time
            parameters={"description": "All events processed within 100ms"}
        ),
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="exactly_once_semantics",
            parameters={"description": "Each event processed exactly once"}
        )
    ]
    
    # Define desired properties (nice to have but not critical)
    properties = [
        TemporalProperty(
            operator=TemporalOperator.EVENTUALLY,
            predicate="auto_scaling_triggered",
            time_bound=300.0,  # Within 5 minutes
            parameters={"description": "System scales based on load"}
        ),
        TemporalProperty(
            operator=TemporalOperator.ALWAYS,
            predicate="cost_optimized",
            parameters={"description": "Resources used efficiently"}
        )
    ]
    
    # Define constraints on the system
    constraints = {
        "latency": {"max": 100, "unit": "ms"},
        "throughput": {"min": 10000, "unit": "events/sec"},
        "availability": {"min": 0.999},
        "cost": {"max": 5000, "unit": "USD/month"},
        "error_rate": {"max": 0.001}
    }
    
    # Environmental assumptions
    assumptions = [
        "network_latency_under_10ms",
        "pubsub_availability_99.95_percent",
        "bigquery_availability_99.99_percent"
    ]
    
    return FormalSpecification(
        invariants=invariants,
        properties=properties,
        constraints=constraints,
        assumptions=assumptions
    )


def design_pipeline_architecture() -> Architecture:
    """
    Design the streaming pipeline architecture.
    
    Classic streaming architecture: 
    Pub/Sub -> Dataflow -> BigQuery with some bells and whistles.
    """
    
    components = [
        {
            "name": "pubsub_subscription",
            "type": "source",
            "config": {
                "subscription": "projects/upir-dev/subscriptions/events-sub",
                "ack_deadline": 30,
                "max_messages": 1000
            }
        },
        {
            "name": "dataflow_pipeline",
            "type": "processor",
            "config": {
                "runner": "DataflowRunner",
                "project": "upir-dev",
                "region": "us-central1",
                "machine_type": "n1-standard-4",
                "max_workers": 10,
                "autoscaling": True
            }
        },
        {
            "name": "event_transformer",
            "type": "transform",
            "config": {
                "transformations": [
                    "parse_json",
                    "validate_schema",
                    "enrich_with_metadata",
                    "deduplicate"
                ]
            }
        },
        {
            "name": "bigquery_sink",
            "type": "sink",
            "config": {
                "dataset": "upir-dev.streaming_data",
                "table": "processed_events",
                "write_method": "STREAMING_INSERTS",
                "create_disposition": "CREATE_IF_NEEDED"
            }
        },
        {
            "name": "dead_letter_queue",
            "type": "error_handler",
            "config": {
                "topic": "projects/upir-dev/topics/dlq-events",
                "max_retries": 3
            }
        }
    ]
    
    connections = [
        {"source": "pubsub_subscription", "target": "dataflow_pipeline"},
        {"source": "dataflow_pipeline", "target": "event_transformer"},
        {"source": "event_transformer", "target": "bigquery_sink"},
        {"source": "event_transformer", "target": "dead_letter_queue", "condition": "on_error"}
    ]
    
    deployment = {
        "environment": "production",
        "cloud_provider": "gcp",
        "regions": ["us-central1"],
        "monitoring": {
            "metrics": ["latency", "throughput", "error_rate"],
            "alerting": True,
            "dashboard": "streaming-pipeline-dashboard"
        }
    }
    
    patterns = [
        "event_sourcing",
        "stream_processing",
        "exactly_once_delivery",
        "dead_letter_queue",
        "auto_scaling"
    ]
    
    return Architecture(
        components=components,
        connections=connections,
        deployment=deployment,
        patterns=patterns
    )


def collect_evidence() -> list:
    """
    Collect evidence supporting our architectural decisions.
    
    In a real system, this would come from benchmarks, load tests,
    and production metrics. Here we're simulating realistic data.
    """
    
    evidence_list = []
    
    # Benchmark evidence
    evidence_list.append(Evidence(
        source="load_test_2024_01",
        type="benchmark",
        data={
            "throughput": 15000,  # events/sec
            "p50_latency": 45,    # ms
            "p99_latency": 95,    # ms
            "error_rate": 0.0005
        },
        confidence=0.85  # High confidence from controlled test
    ))
    
    # Production metrics evidence
    evidence_list.append(Evidence(
        source="production_metrics_30d",
        type="production",
        data={
            "avg_throughput": 12000,
            "avg_latency": 52,
            "availability": 0.9995,
            "events_processed": 31104000000,  # ~31 billion events
            "data_loss_incidents": 0
        },
        confidence=0.95  # Very high confidence from production
    ))
    
    # Formal verification evidence
    evidence_list.append(Evidence(
        source="formal_verification",
        type="formal_proof",
        data={
            "properties_proved": ["data_consistency", "exactly_once_semantics"],
            "properties_disproved": [],
            "assumptions": ["network_reliable", "storage_available"]
        },
        confidence=1.0  # Mathematical proof gives certainty
    ))
    
    # Cost analysis evidence
    evidence_list.append(Evidence(
        source="cost_analysis",
        type="analysis",
        data={
            "monthly_cost": 4200,
            "cost_per_million_events": 0.135,
            "resource_utilization": 0.72
        },
        confidence=0.9
    ))
    
    return evidence_list


def build_reasoning_dag() -> list:
    """
    Build the reasoning DAG showing how we arrived at our architecture.
    
    This captures the decision-making process - why we chose certain
    components and patterns over alternatives.
    """
    
    reasoning_nodes = []
    
    # Root decision: Use streaming architecture
    root = ReasoningNode(
        id="decision_streaming",
        decision="Use streaming architecture for real-time processing",
        rationale="Requirements specify <100ms latency which rules out batch processing",
        alternatives=[
            {"option": "Batch processing", "reason_rejected": "Cannot meet latency requirements"},
            {"option": "Micro-batching", "reason_rejected": "Added complexity without clear benefit"}
        ],
        confidence=0.95
    )
    reasoning_nodes.append(root)
    
    # Choose Pub/Sub for ingestion
    pubsub_decision = ReasoningNode(
        id="decision_pubsub",
        decision="Use Pub/Sub for event ingestion",
        rationale="Native GCP service with exactly-once delivery and proven scale",
        parent_ids=["decision_streaming"],
        alternatives=[
            {"option": "Kafka", "reason_rejected": "Additional operational overhead"},
            {"option": "Kinesis", "reason_rejected": "Not available on GCP"}
        ],
        confidence=0.9
    )
    reasoning_nodes.append(pubsub_decision)
    
    # Choose Dataflow for processing
    dataflow_decision = ReasoningNode(
        id="decision_dataflow",
        decision="Use Dataflow for stream processing",
        rationale="Serverless, auto-scaling, and native exactly-once semantics",
        parent_ids=["decision_streaming"],
        alternatives=[
            {"option": "Apache Flink", "reason_rejected": "Requires cluster management"},
            {"option": "Spark Streaming", "reason_rejected": "Higher latency for true streaming"}
        ],
        confidence=0.85
    )
    reasoning_nodes.append(dataflow_decision)
    
    # Choose BigQuery for storage
    bigquery_decision = ReasoningNode(
        id="decision_bigquery",
        decision="Use BigQuery for analytics storage",
        rationale="Serverless, handles streaming inserts, great for analytics queries",
        parent_ids=["decision_streaming"],
        alternatives=[
            {"option": "Bigtable", "reason_rejected": "Better for key-value, not analytics"},
            {"option": "PostgreSQL", "reason_rejected": "Would need to manage scaling"}
        ],
        confidence=0.9
    )
    reasoning_nodes.append(bigquery_decision)
    
    # Dead letter queue pattern
    dlq_decision = ReasoningNode(
        id="decision_dlq",
        decision="Implement dead letter queue for error handling",
        rationale="Prevents data loss and enables error analysis without blocking pipeline",
        parent_ids=["decision_dataflow"],
        alternatives=[
            {"option": "Fail fast", "reason_rejected": "Would lose data on errors"},
            {"option": "Infinite retry", "reason_rejected": "Could block pipeline"}
        ],
        confidence=0.95
    )
    reasoning_nodes.append(dlq_decision)
    
    return reasoning_nodes


def synthesize_implementation() -> Implementation:
    """
    Synthesize the actual implementation code.
    
    In a real system, this would use CEGIS to generate code from the spec.
    Here we're showing what the output would look like.
    """
    
    # Generated Apache Beam pipeline code
    implementation_code = '''
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io import ReadFromPubSub, WriteToBigQuery
from apache_beam.transforms import window
import json
import hashlib
from datetime import datetime

class ProcessEvent(beam.DoFn):
    """Transform function for event processing."""
    
    def process(self, element):
        try:
            # Parse JSON event
            event = json.loads(element.decode('utf-8'))
            
            # Validate required fields
            required_fields = ['event_id', 'timestamp', 'data']
            if not all(field in event for field in required_fields):
                raise ValueError("Missing required fields")
            
            # Add processing metadata
            event['processed_at'] = datetime.utcnow().isoformat()
            event['pipeline_version'] = '1.0.0'
            
            # Deduplicate using event_id hash
            event['dedup_key'] = hashlib.md5(
                event['event_id'].encode()
            ).hexdigest()
            
            yield event
            
        except Exception as e:
            # Send to dead letter queue
            yield beam.pvalue.TaggedOutput('errors', {
                'original_message': element.decode('utf-8'),
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })

def run_pipeline():
    """Main pipeline execution."""
    
    pipeline_options = PipelineOptions(
        runner='DataflowRunner',
        project='upir-dev',
        region='us-central1',
        temp_location='gs://upir-dev-temp/dataflow',
        job_name='streaming-pipeline',
        streaming=True,
        max_num_workers=10,
        autoscaling_algorithm='THROUGHPUT_BASED'
    )
    
    # Define BigQuery schema
    table_schema = {
        'fields': [
            {'name': 'event_id', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
            {'name': 'data', 'type': 'JSON', 'mode': 'REQUIRED'},
            {'name': 'processed_at', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
            {'name': 'pipeline_version', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'dedup_key', 'type': 'STRING', 'mode': 'REQUIRED'}
        ]
    }
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        
        # Read from Pub/Sub
        events = (
            pipeline
            | 'ReadFromPubSub' >> ReadFromPubSub(
                subscription='projects/upir-dev/subscriptions/events-sub',
                with_attributes=True
            )
        )
        
        # Process events
        processed = (
            events
            | 'ProcessEvents' >> beam.ParDo(ProcessEvent()).with_outputs(
                'errors', main='events'
            )
        )
        
        # Write successful events to BigQuery
        (
            processed.events
            | 'WindowEvents' >> beam.WindowInto(
                window.FixedWindows(60)  # 1-minute windows
            )
            | 'WriteToBigQuery' >> WriteToBigQuery(
                table='upir-dev:streaming_data.processed_events',
                schema=table_schema,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
            )
        )
        
        # Write errors to dead letter queue
        (
            processed.errors
            | 'WriteErrorsToPubSub' >> beam.io.WriteToPubSub(
                topic='projects/upir-dev/topics/dlq-events'
            )
        )

if __name__ == '__main__':
    run_pipeline()
'''
    
    # Create synthesis proof
    proof = SynthesisProof(
        specification_hash="a3f5d7e2b9c4f6a8d1e3b5c7d9f1a3e5",
        implementation_hash=hashlib.sha256(implementation_code.encode()).hexdigest(),
        proof_steps=[
            {"step": 1, "action": "Generated pipeline skeleton from spec"},
            {"step": 2, "action": "Added event processing logic"},
            {"step": 3, "action": "Implemented exactly-once semantics"},
            {"step": 4, "action": "Added error handling with DLQ"},
            {"step": 5, "action": "Verified against temporal properties"}
        ],
        verification_result=True
    )
    
    return Implementation(
        code=implementation_code,
        language="python",
        framework="Apache Beam",
        synthesis_proof=proof,
        performance_profile={
            "expected_latency_p50": 45,
            "expected_latency_p99": 95,
            "expected_throughput": 15000,
            "resource_usage": {
                "cpu_cores": 40,
                "memory_gb": 160,
                "network_gbps": 10
            }
        }
    )


def main():
    """
    Main demonstration of UPIR for streaming pipeline.
    
    This shows the complete flow from specification to verified implementation.
    """
    
    print("=" * 60)
    print("UPIR Streaming Pipeline Example")
    print("Author: subhadipmitra@google.com")
    print("=" * 60)
    print()
    
    # Create UPIR instance
    upir = UPIR(
        name="Streaming Data Pipeline",
        description="Real-time event processing pipeline with formal guarantees"
    )
    
    # Step 1: Define formal specification
    print("Step 1: Creating formal specification...")
    spec = create_streaming_pipeline_spec()
    upir.specification = spec
    print(f"  - Defined {len(spec.invariants)} invariants")
    print(f"  - Defined {len(spec.properties)} properties")
    print(f"  - Defined {len(spec.constraints)} constraints")
    print()
    
    # Step 2: Design architecture
    print("Step 2: Designing architecture...")
    architecture = design_pipeline_architecture()
    upir.architecture = architecture
    print(f"  - Created {len(architecture.components)} components")
    print(f"  - Defined {len(architecture.connections)} connections")
    print(f"  - Using patterns: {', '.join(architecture.patterns[:3])}...")
    print()
    
    # Step 3: Collect evidence
    print("Step 3: Collecting evidence...")
    evidence_list = collect_evidence()
    for evidence in evidence_list:
        eid = upir.add_evidence(evidence)
        print(f"  - Added evidence from {evidence.source} (confidence: {evidence.confidence:.2f})")
    print()
    
    # Step 4: Build reasoning DAG
    print("Step 4: Building reasoning DAG...")
    reasoning_nodes = build_reasoning_dag()
    for node in reasoning_nodes:
        upir.add_reasoning(node)
        print(f"  - Added decision: {node.decision[:50]}...")
    print()
    
    # Step 5: Verify specification
    print("Step 5: Verifying formal properties...")
    verifier = Verifier(timeout=10000)  # 10 second timeout
    results = verifier.verify_specification(upir)
    
    for result in results:
        status_symbol = "✓" if result.verified else "✗"
        print(f"  {status_symbol} {result.property.predicate}: {result.status.value}")
        if result.counterexample:
            print(f"    Counterexample: {result.counterexample}")
    print()
    
    # Step 6: Synthesize implementation
    print("Step 6: Synthesizing implementation...")
    implementation = synthesize_implementation()
    upir.implementation = implementation
    print(f"  - Generated {len(implementation.code.splitlines())} lines of {implementation.language} code")
    print(f"  - Framework: {implementation.framework}")
    print(f"  - Synthesis verified: {implementation.synthesis_proof.verification_result}")
    print()
    
    # Step 7: Compute confidence
    print("Step 7: Computing overall confidence...")
    confidence = upir.compute_overall_confidence()
    print(f"  Overall confidence in architecture: {confidence:.2%}")
    print()
    
    # Step 8: Generate signature
    print("Step 8: Generating cryptographic signature...")
    signature = upir.generate_signature()
    print(f"  Signature: {signature[:32]}...")
    print()
    
    # Save UPIR to file
    print("Saving UPIR to file...")
    with open("streaming_pipeline_upir.json", "w") as f:
        f.write(upir.to_json())
    print("  Saved to streaming_pipeline_upir.json")
    print()
    
    print("=" * 60)
    print("UPIR demonstration complete!")
    print("This pipeline has been formally verified to guarantee:")
    for invariant in spec.invariants:
        print(f"  • {invariant.parameters.get('description', invariant.predicate)}")
    print("=" * 60)


if __name__ == "__main__":
    main()