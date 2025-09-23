#!/usr/bin/env python3
"""
Basic usage example for KARMA.

This example demonstrates how to use KARMA to process a document
and extract knowledge triples.
"""

import os
from pathlib import Path

from karma import KARMAPipeline
from karma.config import create_default_config


def main():
    """Main example function."""
    # Get API key from environment or set directly
    api_key = os.getenv("KARMA_API_KEY")
    if not api_key:
        print("Please set KARMA_API_KEY environment variable or provide API key directly")
        return

    # Create configuration
    config = create_default_config(api_key=api_key)

    # Optional: Customize configuration
    config.pipeline.relevance_threshold = 0.3
    config.pipeline.integration_threshold = 0.6

    # Initialize pipeline
    pipeline = KARMAPipeline.from_config(config)

    # Example with text input
    sample_text = """
    Aspirin is a medication used to reduce pain, fever, or inflammation.
    It works by inhibiting cyclooxygenase enzymes. Studies have shown that
    aspirin can reduce the risk of heart attack and stroke. The drug was
    first synthesized in 1897 by Felix Hoffmann at Bayer.
    """

    print("Processing sample text...")
    result = pipeline.process_document(sample_text)

    # Display results
    print(f"\nExtracted {len(result.integrated_triples)} knowledge triples:")
    for i, triple in enumerate(result.integrated_triples, 1):
        print(f"{i}. {triple.head} --[{triple.relation}]--> {triple.tail}")
        print(f"   Confidence: {triple.confidence:.2f}, "
              f"Clarity: {triple.clarity:.2f}, "
              f"Relevance: {triple.relevance:.2f}")

    print(f"\nExtracted {len(result.entities)} entities:")
    for entity in result.entities:
        print(f"- {entity.name} ({entity.entity_type})")

    # Export results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    kg_path = output_dir / "sample_knowledge_graph.json"
    pipeline.export_knowledge_graph(kg_path)
    print(f"\nKnowledge graph saved to: {kg_path}")

    # Show processing metrics
    print(f"\nProcessing Metrics:")
    print(f"- Total time: {result.metrics.processing_time:.2f} seconds")
    print(f"- Prompt tokens: {result.metrics.prompt_tokens}")
    print(f"- Completion tokens: {result.metrics.completion_tokens}")


if __name__ == "__main__":
    main()