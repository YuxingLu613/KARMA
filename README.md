# KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment 🤖

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ArXiv](https://img.shields.io/badge/arXiv-2502.06472-b31b1b.svg)](https://arxiv.org/abs/2502.06472)
[![PyPI version](https://badge.fury.io/py/karma-nlp.svg)](https://badge.fury.io/py/karma-nlp)

KARMA is a cutting-edge natural language processing framework that leverages a coordinated multi-agent system to automatically extract, validate, and integrate scientific knowledge into structured knowledge graphs. By employing specialized Large Language Model (LLM) agents, KARMA ensures high-quality knowledge extraction while maintaining semantic consistency and domain expertise.

![KARMA Pipeline Overview](https://github.com/user-attachments/assets/477485dc-8d56-4b05-95a4-77547e5ceb39)

## 🌟 Key Features

### Multi-Agent Architecture
- **9 Specialized Agents**: Each agent handles a distinct stage of knowledge extraction
- **Coordinated Processing**: Agents work together in a carefully orchestrated pipeline
- **Quality Assurance**: Multi-stage validation with confidence, clarity, and relevance scoring

### Domain Expertise
- **Biomedical Focus**: Optimized for scientific literature processing
- **Entity Recognition**: Identifies diseases, drugs, genes, proteins, and other biomedical entities
- **Relationship Extraction**: Captures complex relationships like "treats", "causes", "inhibits"
- **Ontology Alignment**: Links entities to standard biomedical ontologies (UMLS, MeSH, NCBI)

### Production Ready
- **Scalable Design**: Handles both single documents and large-scale batch processing
- **Flexible Configuration**: Comprehensive configuration management system
- **Professional CLI**: Command-line interface for easy integration
- **Robust Error Handling**: Graceful handling of processing failures

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## 🚀 Installation

### Using pip (Recommended)

```bash
pip install karma-nlp
```

### From Source

```bash
git clone https://github.com/YuxingLu613/KARMA.git
cd KARMA
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/YuxingLu613/KARMA.git
cd KARMA
pip install -e ".[dev]"
```

## ⚡ Quick Start

### Command Line Interface

```bash
# Process a PDF document
karma process document.pdf --api-key YOUR_OPENAI_KEY --output results.json

# Process with custom settings
karma process document.pdf --api-key YOUR_KEY --model gpt-4 \\
  --relevance-threshold 0.3 --integration-threshold 0.7

# Create and use configuration file
karma config create --api-key YOUR_KEY --config-file karma_config.json
karma process document.pdf --config karma_config.json
```

### Python API

```python
from karma import KARMAPipeline
from karma.config import create_default_config

# Create configuration
config = create_default_config(api_key="your-openai-api-key")

# Initialize pipeline
pipeline = KARMAPipeline.from_config(config)

# Process document
result = pipeline.process_document("path/to/document.pdf")

# Access results
print(f"Extracted {len(result.integrated_triples)} knowledge triples")
for triple in result.integrated_triples[:5]:
    print(f"{triple.head} --[{triple.relation}]--> {triple.tail} (confidence: {triple.confidence:.2f})")

# Export knowledge graph
pipeline.export_knowledge_graph("knowledge_graph.json")
```

### Batch Processing

```python
# Process multiple documents
documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
results = pipeline.process_batch(documents)

# Analyze results
total_triples = sum(len(result.integrated_triples) for result in results)
print(f"Extracted {total_triples} total knowledge triples from {len(documents)} documents")
```

## 🏗️ Architecture

KARMA employs a sophisticated multi-agent architecture where each agent specializes in a specific aspect of knowledge extraction:

### Pipeline Stages

1. **Ingestion Agent (IA)** 📥
   - Retrieves and standardizes raw documents (PDF, HTML, text)
   - Extracts metadata (authors, journal, publication date)
   - Handles OCR artifacts and text normalization

2. **Reader Agent (RA)** 📖
   - Segments documents into logical chunks
   - Scores segment relevance using domain knowledge
   - Filters non-relevant content (acknowledgments, references)

3. **Summarizer Agent (SA)** 📝
   - Condenses relevant segments into concise summaries
   - Preserves technical details and entity relationships
   - Maintains quantitative data and statistical findings

4. **Entity Extraction Agent (EEA)** 🔍
   - Identifies biomedical entities using few-shot learning
   - Classifies entity types (Drug, Disease, Gene, Protein, etc.)
   - Normalizes entities to canonical ontology forms

5. **Relationship Extraction Agent (REA)** 🔗
   - Detects relationships between entity pairs
   - Handles multi-label classification for overlapping relations
   - Recognizes negation and conditional statements

6. **Schema Alignment Agent (SAA)** 🎯
   - Maps entities to knowledge graph schema types
   - Standardizes relationship labels
   - Handles novel entities and relations

7. **Conflict Resolution Agent (CRA)** ⚖️
   - Resolves contradictions between new and existing knowledge
   - Uses LLM-based debate and evidence aggregation
   - Maintains knowledge graph consistency

8. **Evaluator Agent (EA)** 📊
   - Computes integration confidence using multiple signals
   - Evaluates confidence, clarity, and relevance
   - Applies threshold-based approval for final integration

## 💡 Usage Examples

### Basic Document Processing

```python
from karma import KARMAPipeline

# Initialize with API key
pipeline = KARMAPipeline(api_key="your-key", model_name="gpt-4o")

# Process a research paper
result = pipeline.process_document("alzheimer_research.pdf")

# Examine extracted knowledge
print("Extracted Entities:")
for entity in result.entities[:10]:
    print(f"  {entity.name} ({entity.entity_type})")

print("\\nExtracted Relationships:")
for triple in result.integrated_triples[:5]:
    print(f"  {triple.head} {triple.relation} {triple.tail}")
```

### Advanced Configuration

```python
from karma.config import KARMAConfig, ModelConfig, PipelineConfig

# Create custom configuration
config = KARMAConfig(
    model=ModelConfig(
        name="gpt-4",
        api_key="your-key",
        temperature=0.1
    ),
    pipeline=PipelineConfig(
        relevance_threshold=0.3,
        integration_threshold=0.7,
        batch_size=10
    )
)

# Use configuration
pipeline = KARMAPipeline.from_config(config)
```

### Domain-Specific Processing

```python
# Process with custom domain focus
result = pipeline.process_document(
    "cancer_research.pdf",
    domain="oncology",
    relevance_threshold=0.4
)

# Filter for specific entity types
drugs = [e for e in result.entities if e.entity_type == "Drug"]
diseases = [e for e in result.entities if e.entity_type == "Disease"]

print(f"Found {len(drugs)} drugs and {len(diseases)} diseases")
```

### Knowledge Graph Analysis

```python
# Get knowledge graph statistics
kg = pipeline.get_knowledge_graph()
stats = kg.get_statistics()

print(f"Knowledge Graph Contains:")
print(f"  Entities: {stats['entity_count']}")
print(f"  Relationships: {stats['triple_count']}")
print(f"  Unique Relations: {stats['unique_relations']}")
print(f"  Average Confidence: {stats['avg_confidence']:.2f}")
```

## 📊 Output Format

KARMA generates comprehensive results including:

### Knowledge Triples
```json
{
  "head": "Metformin",
  "relation": "treats",
  "tail": "Type 2 Diabetes",
  "confidence": 0.95,
  "clarity": 0.90,
  "relevance": 0.88,
  "source": "relationship_extraction"
}
```

### Entities
```json
{
  "entity_id": "metformin",
  "entity_type": "Drug",
  "name": "Metformin",
  "normalized_id": "MESH:D008687",
  "aliases": ["Glucophage", "Dimethylbiguanide"]
}
```

### Processing Metrics
```json
{
  "processing_time": 45.2,
  "prompt_tokens": 15420,
  "completion_tokens": 3240,
  "agent_times": {
    "ingestion": 2.1,
    "reader": 8.3,
    "summarizer": 12.4,
    "entity_extraction": 9.8,
    "relationship_extraction": 8.9
  }
}
```

## ⚙️ Configuration

KARMA supports comprehensive configuration through files or environment variables:

### Configuration File Example

```json
{
  "model": {
    "name": "gpt-4o",
    "api_key": "your-api-key",
    "temperature": 0.1,
    "max_tokens": 4096
  },
  "pipeline": {
    "relevance_threshold": 0.2,
    "integration_threshold": 0.6,
    "batch_size": 5
  },
  "agents": {
    "entity_extraction": {
      "min_entity_length": 2
    },
    "relationship_extraction": {
      "min_confidence": 0.3
    }
  },
  "output_dir": "results",
  "save_intermediate": true
}
```

### Environment Variables

```bash
export KARMA_API_KEY="your-api-key"
export KARMA_MODEL="gpt-4"
export KARMA_RELEVANCE_THRESHOLD="0.3"
export KARMA_OUTPUT_DIR="./output"
```

## 🛠️ Requirements

- **Python**: 3.8 or higher
- **Dependencies**:
  - `openai>=1.0.0`: LLM integration
  - `PyPDF2>=3.0.0`: PDF processing
  - `typing-extensions>=4.0.0`: Type hints support

### Optional Dependencies
- `spacy>=3.4.0`: Enhanced NLP processing
- `networkx>=2.8.0`: Knowledge graph operations
- `matplotlib>=3.5.0`: Visualization
- `pandas>=1.4.0`: Data analysis

## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=karma --cov-report=html
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/YuxingLu613/KARMA.git
cd KARMA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **Pytest** for testing

## 📚 Documentation

- **API Documentation**: [https://karma-nlp.readthedocs.io/](https://karma-nlp.readthedocs.io/)
- **Research Paper**: [ArXiv:2502.06472](https://arxiv.org/abs/2502.06472)
- **Examples**: See the `examples/` directory
- **Tutorials**: Available in `docs/tutorials/`

## 🏆 Citation

If you use KARMA in your research, please cite our paper:

```bibtex
@article{lu2025karma,
  title={KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment},
  author={Lu, Yuxing and Wang, Jinzhuo},
  journal={arXiv preprint arXiv:2502.06472},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Author**: Yuxing Lu
- **Email**: yxlu0613@gmail.com
- **Issues**: [GitHub Issues](https://github.com/YuxingLu613/KARMA/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YuxingLu613/KARMA/discussions)

## 🙏 Acknowledgments

- OpenAI for providing the LLM infrastructure
- The biomedical research community for inspiration and validation
- All contributors and users of the KARMA framework

---

<p align="center">
  <strong>KARMA: Transforming Scientific Literature into Structured Knowledge 🧬</strong>
</p>