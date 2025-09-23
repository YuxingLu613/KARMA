"""
Command Line Interface for KARMA.

This module provides a command-line interface for running KARMA
on documents and managing configurations.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from karma import KARMAPipeline, __version__, get_info
from karma.config import KARMAConfig, load_config, save_config, create_default_config


def setup_logging(level: str = "INFO"):
    """Setup basic logging for CLI."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def process_command(args) -> int:
    """Handle the process command."""
    try:
        # Load or create configuration
        if args.config:
            config = load_config(args.config)
        else:
            # Create config from command line arguments
            config = create_default_config(
                api_key=args.api_key,
                output_dir=args.output_dir
            )

            if args.model:
                config.model.name = args.model
            if args.base_url:
                config.model.base_url = args.base_url
            if args.relevance_threshold:
                config.pipeline.relevance_threshold = args.relevance_threshold
            if args.integration_threshold:
                config.pipeline.integration_threshold = args.integration_threshold

        # Override API key if provided
        if args.api_key:
            config.model.api_key = args.api_key

        # Validate configuration
        if not config.model.api_key:
            print("Error: API key is required. Provide via --api-key or config file.")
            return 1

        # Setup logging
        setup_logging(args.log_level)

        # Create pipeline
        pipeline = KARMAPipeline.from_config(config)

        # Process input
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return 1

        print(f"Processing document: {input_path}")
        print(f"Using model: {config.model.name}")

        # Run processing
        result = pipeline.process_document(
            input_path,
            domain=args.domain,
            relevance_threshold=config.pipeline.relevance_threshold
        )

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export knowledge graph
        kg_path = output_path.with_suffix('.kg.json')
        pipeline.export_knowledge_graph(kg_path)

        # Save intermediate results if requested
        if args.save_intermediate:
            intermediate_path = output_path.with_suffix('.intermediate.json')
            result.save_to_file(intermediate_path)
            print(f"Intermediate results saved to: {intermediate_path}")

        # Print summary
        print(f"\\nProcessing completed successfully!")
        print(f"Extracted {len(result.integrated_triples)} knowledge triples")
        print(f"Found {len(result.entities)} entities")
        print(f"Processing time: {result.metrics.processing_time:.2f} seconds")
        print(f"Results saved to: {kg_path}")

        # Print statistics if verbose
        if args.verbose:
            stats = pipeline.get_pipeline_statistics()
            print(f"\\nPipeline Statistics:")
            print(f"Knowledge Graph: {stats['knowledge_graph']}")

        return 0

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def config_command(args) -> int:
    """Handle configuration commands."""
    try:
        if args.config_action == "create":
            # Create new configuration
            if not args.api_key:
                print("Error: API key is required for creating configuration.")
                return 1

            config = create_default_config(args.api_key, args.output_dir)

            if args.model:
                config.model.name = args.model
            if args.base_url:
                config.model.base_url = args.base_url

            # Save configuration
            config_path = Path(args.config_file)
            save_config(config, config_path)
            print(f"Configuration saved to: {config_path}")

        elif args.config_action == "show":
            # Show existing configuration
            config_path = Path(args.config_file)
            if not config_path.exists():
                print(f"Configuration file not found: {config_path}")
                return 1

            config = load_config(config_path)
            # Hide API key for security
            config_dict = config.to_dict()
            config_dict['model']['api_key'] = '***HIDDEN***'

            print(json.dumps(config_dict, indent=2))

        elif args.config_action == "validate":
            # Validate configuration
            config_path = Path(args.config_file)
            if not config_path.exists():
                print(f"Configuration file not found: {config_path}")
                return 1

            config = load_config(config_path)
            config.validate()
            print("Configuration is valid!")

        return 0

    except Exception as e:
        print(f"Configuration error: {str(e)}")
        return 1


def info_command(args) -> int:
    """Handle info command."""
    info = get_info()

    print(f"KARMA v{info['version']}")
    print(f"Author: {info['author']}")
    print(f"Description: {info['description']}")
    print(f"\\nAvailable Agents:")
    for agent in info['agents']:
        print(f"  - {agent}")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="KARMA: Multi-Agent LLMs for Automated Knowledge Graph Enrichment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a PDF document
  karma process document.pdf --api-key YOUR_KEY --output results.json

  # Process with custom model and thresholds
  karma process document.pdf --api-key YOUR_KEY --model gpt-4 \\
    --relevance-threshold 0.3 --integration-threshold 0.7

  # Use configuration file
  karma process document.pdf --config config.json --output results.json

  # Create configuration file
  karma config create --api-key YOUR_KEY --config-file karma_config.json

  # Show version and info
  karma info
        """
    )

    parser.add_argument('--version', action='version', version=f'KARMA {__version__}')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Process command
    process_parser = subparsers.add_parser('process', help='Process documents')
    process_parser.add_argument('input', help='Input document path (PDF or text)')
    process_parser.add_argument('--api-key', help='OpenAI API key')
    process_parser.add_argument('--base-url', help='API base URL (for Azure or custom endpoints)')
    process_parser.add_argument('--model', default='gpt-4o', help='Model name (default: gpt-4o)')
    process_parser.add_argument('--config', help='Configuration file path')
    process_parser.add_argument('--output', default='karma_output.json', help='Output file path')
    process_parser.add_argument('--output-dir', default='output', help='Output directory')
    process_parser.add_argument('--domain', default='biomedical', help='Domain context')
    process_parser.add_argument('--relevance-threshold', type=float, help='Relevance threshold (0-1)')
    process_parser.add_argument('--integration-threshold', type=float, help='Integration threshold (0-1)')
    process_parser.add_argument('--save-intermediate', action='store_true', help='Save intermediate results')
    process_parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    process_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    process_parser.set_defaults(func=process_command)

    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_subparsers = config_parser.add_subparsers(dest='config_action', help='Configuration actions')

    # Config create
    create_parser = config_subparsers.add_parser('create', help='Create new configuration')
    create_parser.add_argument('--api-key', required=True, help='OpenAI API key')
    create_parser.add_argument('--model', default='gpt-4o', help='Model name')
    create_parser.add_argument('--base-url', help='API base URL')
    create_parser.add_argument('--output-dir', default='output', help='Output directory')
    create_parser.add_argument('--config-file', default='karma_config.json', help='Configuration file path')

    # Config show
    show_parser = config_subparsers.add_parser('show', help='Show configuration')
    show_parser.add_argument('--config-file', default='karma_config.json', help='Configuration file path')

    # Config validate
    validate_parser = config_subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument('--config-file', default='karma_config.json', help='Configuration file path')

    config_parser.set_defaults(func=config_command)

    # Info command
    info_parser = subparsers.add_parser('info', help='Show package information')
    info_parser.set_defaults(func=info_command)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Execute command
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())