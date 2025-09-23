#!/usr/bin/env python3
"""
KARMA Main Entry Point

This script provides an easy way to run the KARMA pipeline on documents
with sensible defaults and comprehensive output.
"""

import os
import sys
import argparse
import csv
from pathlib import Path
from datetime import datetime

# Add the karma package to Python path
sys.path.insert(0, '/Users/luyuxing/Projects/KARMA')

from karma import KARMAPipeline
from karma.config import create_default_config


def save_relationships_to_csv(result, csv_path, integration_threshold):
    """Save all extracted relationships to CSV with scores and pass/fail status."""
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'head_entity', 'relation', 'tail_entity',
            'confidence', 'clarity', 'relevance',
            'integration_score', 'passed_integration',
            'source_stage', 'processing_notes'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Collect all relationships from different stages
        all_relationships = []

        # Add relationships from each stage
        stages = [
            ('relationships', 'initial_extraction'),
            ('aligned_triples', 'after_alignment'),
            ('final_triples', 'after_conflict_resolution'),
            ('integrated_triples', 'integrated_final')
        ]

        for attr_name, stage_name in stages:
            if hasattr(result, attr_name):
                rels = getattr(result, attr_name)
                if rels:
                    for rel in rels:
                        all_relationships.append((rel, stage_name))

        # Remove duplicates while preserving the latest stage
        unique_relationships = {}
        stage_priority = {
            'initial_extraction': 1,
            'after_alignment': 2,
            'after_conflict_resolution': 3,
            'integrated_final': 4
        }

        for rel, stage in all_relationships:
            key = f"{rel.head.lower()}_{rel.relation.lower()}_{rel.tail.lower()}"
            if key not in unique_relationships or stage_priority[stage] > stage_priority[unique_relationships[key][1]]:
                unique_relationships[key] = (rel, stage)

        # Write relationships to CSV
        for rel, stage in unique_relationships.values():
            integration_score = 0.5 * rel.confidence + 0.25 * rel.clarity + 0.25 * rel.relevance
            passed_integration = integration_score >= integration_threshold

            if stage == 'integrated_final':
                notes = "Successfully integrated into knowledge graph"
            elif stage == 'after_conflict_resolution':
                notes = "Passed conflict resolution but failed final evaluation"
            elif stage == 'after_alignment':
                notes = "Passed schema alignment but failed conflict resolution or evaluation"
            else:
                notes = "Failed early in pipeline"

            writer.writerow({
                'head_entity': rel.head,
                'relation': rel.relation,
                'tail_entity': rel.tail,
                'confidence': f"{rel.confidence:.3f}",
                'clarity': f"{rel.clarity:.3f}",
                'relevance': f"{rel.relevance:.3f}",
                'integration_score': f"{integration_score:.3f}",
                'passed_integration': 'YES' if passed_integration else 'NO',
                'source_stage': stage,
                'processing_notes': notes
            })

    return len(unique_relationships)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="KARMA: Multi-Agent Knowledge Graph Enrichment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a PDF with default settings
  python main.py document.pdf

  # Process with custom API key and model
  python main.py document.pdf --api-key YOUR_KEY --model gpt-4

  # Process with custom thresholds
  python main.py document.pdf --relevance-threshold 0.3 --integration-threshold 0.7

  # Specify output directory
  python main.py document.pdf --output-dir results/experiment1
        """
    )

    parser.add_argument('input', help='Input document path (PDF or text file)')
    parser.add_argument('--api-key',
                       default=os.getenv('KARMA_API_KEY'),
                       help='OpenAI API key (default: from KARMA_API_KEY env var)')
    parser.add_argument('--model', default='gpt-4o-mini', help='Model to use (default: gpt-4o-mini)')
    parser.add_argument('--relevance-threshold', type=float, default=0.2, help='Relevance threshold (default: 0.2)')
    parser.add_argument('--integration-threshold', type=float, default=0.5, help='Integration threshold (default: 0.5)')
    parser.add_argument('--output-dir', default='karma_output', help='Output directory (default: karma_output)')
    parser.add_argument('--domain', default='biomedical', help='Domain context (default: biomedical)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return 1

    if not args.api_key:
        print("❌ Error: API key is required. Set KARMA_API_KEY environment variable or use --api-key")
        return 1

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 KARMA Knowledge Graph Enrichment Pipeline")
    print("=" * 60)
    print(f"📄 Input: {input_path}")
    print(f"🤖 Model: {args.model}")
    print(f"📊 Relevance threshold: {args.relevance_threshold}")
    print(f"📊 Integration threshold: {args.integration_threshold}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    try:
        # Create configuration
        config = create_default_config(api_key=args.api_key)
        config.model.name = args.model
        config.pipeline.relevance_threshold = args.relevance_threshold
        config.pipeline.integration_threshold = args.integration_threshold

        # Initialize pipeline
        print("🔧 Initializing KARMA pipeline...")
        pipeline = KARMAPipeline.from_config(config)

        # Process document
        print(f"📖 Processing document: {input_path.name}")
        print("⏳ This may take several minutes...")

        result = pipeline.process_document(
            input_path,
            domain=args.domain,
            relevance_threshold=args.relevance_threshold
        )

        print("✅ Processing completed successfully!")
        print("=" * 60)

        # Display results summary
        print("📊 PROCESSING RESULTS:")
        print(f"   📝 Text segments: {len(result.segments)}")
        print(f"   ✨ Relevant segments: {len(result.relevant_segments)}")
        print(f"   📄 Summaries: {len(result.summaries)}")
        print(f"   🏷️  Entities extracted: {len(result.entities)}")
        print(f"   🔗 Initial relationships: {len(result.relationships) if result.relationships else 0}")
        print(f"   ✅ Final integrated triples: {len(result.integrated_triples)}")
        print(f"   ⏱️  Total processing time: {result.metrics.processing_time:.2f} seconds")

        # Save all results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Save all relationships to CSV
        csv_path = output_dir / f"all_relationships_{timestamp}.csv"
        total_rels = save_relationships_to_csv(result, csv_path, args.integration_threshold)

        passed_count = len(result.integrated_triples)
        failed_count = total_rels - passed_count
        success_rate = (passed_count / total_rels * 100) if total_rels > 0 else 0

        print(f"\n📊 RELATIONSHIP ANALYSIS:")
        print(f"   📋 Total relationships extracted: {total_rels}")
        print(f"   ✅ Passed integration: {passed_count}")
        print(f"   ❌ Failed integration: {failed_count}")
        print(f"   📈 Success rate: {success_rate:.1f}%")
        print(f"   💾 Detailed CSV saved: {csv_path}")

        # 2. Export knowledge graph
        kg_path = output_dir / f"knowledge_graph_{timestamp}.json"
        pipeline.export_knowledge_graph(kg_path)
        print(f"   💾 Knowledge graph saved: {kg_path}")

        # 3. Save intermediate results
        intermediate_path = output_dir / f"intermediate_results_{timestamp}.json"
        result.save_to_file(intermediate_path)
        print(f"   💾 Intermediate results saved: {intermediate_path}")

        # Show sample results
        if result.entities and args.verbose:
            print(f"\n🏷️ SAMPLE ENTITIES:")
            entity_types = {}
            for entity in result.entities:
                entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1

            for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {entity_type}: {count}")

        if result.integrated_triples and args.verbose:
            print(f"\n🔗 TOP KNOWLEDGE TRIPLES:")
            for i, triple in enumerate(result.integrated_triples[:5], 1):
                integration_score = 0.5 * triple.confidence + 0.25 * triple.clarity + 0.25 * triple.relevance
                print(f"   {i}. {triple.head} --[{triple.relation}]--> {triple.tail}")
                print(f"      Score: {integration_score:.3f} (C:{triple.confidence:.2f}, Cl:{triple.clarity:.2f}, R:{triple.relevance:.2f})")

        print(f"\n🎉 KARMA processing completed successfully!")
        print(f"📁 All outputs saved to: {output_dir}")
        return 0

    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)