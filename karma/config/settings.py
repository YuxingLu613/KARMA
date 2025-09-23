"""
Configuration settings for the KARMA framework.

This module provides configuration management including model settings,
API configurations, and pipeline parameters.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for LLM model settings."""
    name: str = "gpt-4o"
    api_key: str = ""
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    timeout: int = 60


@dataclass
class PipelineConfig:
    """Configuration for pipeline processing parameters."""
    relevance_threshold: float = 0.2
    integration_threshold: float = 0.6
    batch_size: int = 5
    max_segments: int = 100
    max_entities_per_segment: int = 20
    enable_caching: bool = True
    parallel_processing: bool = False


@dataclass
class AgentConfig:
    """Configuration for individual agent settings."""
    ingestion: Dict = field(default_factory=lambda: {"extract_metadata": True})
    reader: Dict = field(default_factory=lambda: {"segment_min_length": 30})
    summarizer: Dict = field(default_factory=lambda: {"max_summary_length": 100})
    entity_extraction: Dict = field(default_factory=lambda: {"min_entity_length": 2})
    relationship_extraction: Dict = field(default_factory=lambda: {"min_confidence": 0.3})
    schema_alignment: Dict = field(default_factory=lambda: {"use_ontology_mapping": True})
    conflict_resolution: Dict = field(default_factory=lambda: {"resolution_strategy": "confidence"})
    evaluator: Dict = field(default_factory=lambda: {"confidence_weight": 0.5, "clarity_weight": 0.25, "relevance_weight": 0.25})


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10_000_000  # 10MB
    backup_count: int = 5


@dataclass
class KARMAConfig:
    """Main configuration class for KARMA framework."""
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Output settings
    output_dir: str = "output"
    save_intermediate: bool = True
    export_format: str = "json"  # json, csv, or both

    # Performance settings
    enable_metrics: bool = True
    memory_limit_mb: int = 1024

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'KARMAConfig':
        """Create configuration from dictionary."""
        config = cls()

        # Update model config
        if 'model' in data:
            model_data = data['model']
            config.model = ModelConfig(**model_data)

        # Update pipeline config
        if 'pipeline' in data:
            pipeline_data = data['pipeline']
            config.pipeline = PipelineConfig(**pipeline_data)

        # Update agent config
        if 'agents' in data:
            agents_data = data['agents']
            config.agents = AgentConfig(**agents_data)

        # Update logging config
        if 'logging' in data:
            logging_data = data['logging']
            config.logging = LoggingConfig(**logging_data)

        # Update other fields
        for key, value in data.items():
            if key not in ['model', 'pipeline', 'agents', 'logging'] and hasattr(config, key):
                setattr(config, key, value)

        return config

    def validate(self) -> bool:
        """
        Validate configuration settings.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate model settings
        if not self.model.api_key:
            raise ValueError("API key is required")

        if not (0.0 <= self.model.temperature <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")

        # Validate pipeline settings
        if not (0.0 <= self.pipeline.relevance_threshold <= 1.0):
            raise ValueError("Relevance threshold must be between 0.0 and 1.0")

        if not (0.0 <= self.pipeline.integration_threshold <= 1.0):
            raise ValueError("Integration threshold must be between 0.0 and 1.0")

        if self.pipeline.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        # Validate output directory
        output_path = Path(self.output_dir)
        if not output_path.exists():
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Cannot create output directory: {e}")

        return True

    def setup_logging(self):
        """Setup logging based on configuration."""
        logging_level = getattr(logging, self.logging.level.upper(), logging.INFO)

        # Configure basic logging
        handlers = []

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging_level)
        console_formatter = logging.Formatter(self.logging.format)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

        # File handler if specified
        if self.logging.file_path:
            try:
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    self.logging.file_path,
                    maxBytes=self.logging.max_file_size,
                    backupCount=self.logging.backup_count
                )
                file_handler.setLevel(logging_level)
                file_formatter = logging.Formatter(self.logging.format)
                file_handler.setFormatter(file_formatter)
                handlers.append(file_handler)
            except Exception as e:
                logger.warning(f"Failed to setup file logging: {e}")

        # Configure root logger
        logging.basicConfig(
            level=logging_level,
            format=self.logging.format,
            handlers=handlers,
            force=True
        )

        # Set specific logger levels
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)


def load_config(config_path: Union[str, Path]) -> KARMAConfig:
    """
    Load configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Loaded configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                # Try to parse as JSON anyway
                data = json.load(f)

        config = KARMAConfig.from_dict(data)
        config.validate()
        return config

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load configuration: {e}")


def save_config(config: KARMAConfig, config_path: Union[str, Path]):
    """
    Save configuration to file.

    Args:
        config: Configuration to save
        config_path: Path to save configuration
    """
    config_path = Path(config_path)

    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Configuration saved to {config_path}")

    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise


def create_default_config(api_key: str, output_dir: str = "output") -> KARMAConfig:
    """
    Create a default configuration with the provided API key.

    Args:
        api_key: OpenAI API key
        output_dir: Output directory path

    Returns:
        Default configuration
    """
    config = KARMAConfig()
    config.model.api_key = api_key
    config.output_dir = output_dir

    return config


def get_environment_config() -> KARMAConfig:
    """
    Create configuration from environment variables.

    Returns:
        Configuration built from environment variables
    """
    import os

    config = KARMAConfig()

    # Model configuration from environment
    if os.getenv('KARMA_API_KEY'):
        config.model.api_key = os.getenv('KARMA_API_KEY')
    if os.getenv('KARMA_BASE_URL'):
        config.model.base_url = os.getenv('KARMA_BASE_URL')
    if os.getenv('KARMA_MODEL'):
        config.model.name = os.getenv('KARMA_MODEL')

    # Pipeline configuration from environment
    if os.getenv('KARMA_RELEVANCE_THRESHOLD'):
        config.pipeline.relevance_threshold = float(os.getenv('KARMA_RELEVANCE_THRESHOLD'))
    if os.getenv('KARMA_INTEGRATION_THRESHOLD'):
        config.pipeline.integration_threshold = float(os.getenv('KARMA_INTEGRATION_THRESHOLD'))

    # Output configuration from environment
    if os.getenv('KARMA_OUTPUT_DIR'):
        config.output_dir = os.getenv('KARMA_OUTPUT_DIR')

    return config