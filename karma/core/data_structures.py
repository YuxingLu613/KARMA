"""
Core data structures for the KARMA framework.

This module defines the fundamental data structures used throughout the pipeline,
including knowledge triples, entities, and intermediate processing results.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union
import json


@dataclass
class KnowledgeTriple:
    """
    Represents a knowledge triple in the biomedical domain.

    A knowledge triple consists of a subject (head), predicate (relation),
    and object (tail) along with quality metrics.

    Attributes:
        head: The subject entity
        relation: The relationship type
        tail: The object entity
        confidence: Model confidence score [0-1]
        source: Origin of the triple
        relevance: Domain relevance score [0-1]
        clarity: Linguistic clarity score [0-1]
    """
    head: str
    relation: str
    tail: str
    confidence: float = 0.0
    source: str = "unknown"
    relevance: float = 0.0
    clarity: float = 0.0

    def __str__(self) -> str:
        """String representation of the knowledge triple."""
        return f"({self.head}) -[{self.relation}]-> ({self.tail})"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeTriple':
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class KGEntity:
    """
    Represents a canonical entity in the knowledge graph.

    Attributes:
        entity_id: Unique identifier
        entity_type: Semantic type (e.g. Drug, Disease, Gene, Protein)
        name: Display name
        normalized_id: Reference to standard ontology (e.g., UMLS:C0004238)
        aliases: Alternative names for the entity
    """
    entity_id: str
    entity_type: str = "Unknown"
    name: str = ""
    normalized_id: str = "N/A"
    aliases: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        """String representation of the entity."""
        return f"{self.name} ({self.entity_type})"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'KGEntity':
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class DocumentMetadata:
    """
    Metadata for processed documents.

    Attributes:
        title: Document title
        authors: List of authors
        journal: Journal or publication venue
        pub_date: Publication date
        doi: Digital Object Identifier
        pmid: PubMed identifier
        document_type: Type of document (article, review, etc.)
    """
    title: str = "Unknown Title"
    authors: List[str] = field(default_factory=list)
    journal: str = "Unknown Journal"
    pub_date: str = "N/A"
    doi: str = "N/A"
    pmid: str = "N/A"
    document_type: str = "article"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ProcessingMetrics:
    """
    Metrics for tracking processing performance.

    Attributes:
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens generated
        processing_time: Total processing time in seconds
        agent_times: Processing time for each agent
        error_count: Number of errors encountered
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    processing_time: float = 0.0
    agent_times: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0

    def add_agent_time(self, agent_name: str, time: float):
        """Add processing time for a specific agent."""
        self.agent_times[agent_name] = time
        self.processing_time += time

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class IntermediateOutput:
    """
    Container for storing intermediate outputs from each pipeline stage.

    This class tracks the full pipeline state including raw inputs,
    intermediate results, and final outputs for debugging and analysis.
    """
    # Input data
    raw_text: str = ""
    metadata: Optional[DocumentMetadata] = None

    # Stage outputs
    segments: List[Dict] = field(default_factory=list)
    relevant_segments: List[Dict] = field(default_factory=list)
    summaries: List[str] = field(default_factory=list)
    entities: List[KGEntity] = field(default_factory=list)
    relationships: List[KnowledgeTriple] = field(default_factory=list)
    aligned_entities: List[KGEntity] = field(default_factory=list)
    aligned_triples: List[KnowledgeTriple] = field(default_factory=list)
    final_triples: List[KnowledgeTriple] = field(default_factory=list)
    integrated_triples: List[KnowledgeTriple] = field(default_factory=list)

    # Performance metrics
    metrics: ProcessingMetrics = field(default_factory=ProcessingMetrics)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, list) and value:
                if hasattr(value[0], 'to_dict'):
                    result[key] = [item.to_dict() for item in value]
                else:
                    result[key] = value
            else:
                result[key] = value
        return result

    def save_to_file(self, filepath: str):
        """Save intermediate results to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'IntermediateOutput':
        """Load intermediate results from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Reconstruct objects from dictionaries
        instance = cls()
        for key, value in data.items():
            if key == 'entities' and value:
                instance.entities = [KGEntity.from_dict(item) for item in value]
            elif key in ['relationships', 'aligned_triples', 'final_triples', 'integrated_triples'] and value:
                setattr(instance, key, [KnowledgeTriple.from_dict(item) for item in value])
            elif key == 'metadata' and value:
                instance.metadata = DocumentMetadata(**value)
            elif key == 'metrics' and value:
                instance.metrics = ProcessingMetrics(**value)
            else:
                setattr(instance, key, value)

        return instance


@dataclass
class KnowledgeGraph:
    """
    Represents the complete knowledge graph structure.

    Attributes:
        entities: Set of entity names in the graph
        triples: List of knowledge triples
        metadata: Graph-level metadata
    """
    entities: set = field(default_factory=set)
    triples: List[KnowledgeTriple] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def add_entity(self, entity: Union[str, KGEntity]):
        """Add an entity to the knowledge graph."""
        if isinstance(entity, KGEntity):
            self.entities.add(entity.name)
        else:
            self.entities.add(entity)

    def add_triple(self, triple: KnowledgeTriple):
        """Add a knowledge triple to the graph."""
        self.triples.append(triple)
        # Automatically add entities from the triple
        self.entities.add(triple.head)
        self.entities.add(triple.tail)

    def get_statistics(self) -> Dict:
        """Get statistics about the knowledge graph."""
        relation_counts = {}
        entity_type_counts = {}

        for triple in self.triples:
            relation_counts[triple.relation] = relation_counts.get(triple.relation, 0) + 1

        return {
            'entity_count': len(self.entities),
            'triple_count': len(self.triples),
            'unique_relations': len(relation_counts),
            'relation_distribution': relation_counts,
            'avg_confidence': sum(t.confidence for t in self.triples) / len(self.triples) if self.triples else 0
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'entities': list(self.entities),
            'triples': [triple.to_dict() for triple in self.triples],
            'metadata': self.metadata,
            'statistics': self.get_statistics()
        }

    def save_to_file(self, filepath: str):
        """Save knowledge graph to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'KnowledgeGraph':
        """Load knowledge graph from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        kg = cls()
        kg.entities = set(data.get('entities', []))
        kg.triples = [KnowledgeTriple.from_dict(triple_data) for triple_data in data.get('triples', [])]
        kg.metadata = data.get('metadata', {})

        return kg