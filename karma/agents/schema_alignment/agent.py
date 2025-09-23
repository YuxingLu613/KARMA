"""Schema Alignment Agent Implementation"""

import logging
from typing import List, Tuple
from karma.core.base_agent import BaseAgent
from karma.core.data_structures import KGEntity, KnowledgeTriple

logger = logging.getLogger(__name__)


class SchemaAlignmentAgent(BaseAgent):
    """Schema Alignment Agent (SAA) for entity type classification and relation normalization."""

    def __init__(self, client, model_name: str):
        system_prompt = """You are a specialized Schema Alignment Agent for biomedical knowledge graphs. Your task is to standardize entity types and relationship labels according to established biomedical ontologies and conventions.

OBJECTIVE: Ensure consistent entity classification and relationship terminology across the knowledge graph by mapping to standardized vocabularies.

ENTITY TYPE STANDARDIZATION:
Map entity types to these standard categories:
- DRUG → Drug
- DISEASE → Disease
- GENE → Gene
- PROTEIN → Protein
- CHEMICAL → Chemical
- PATHWAY → Pathway
- ANATOMY → Anatomy
- Unknown → Most appropriate category based on context

RELATIONSHIP STANDARDIZATION:
Map relationship labels to canonical forms:
- "inhibit", "inhibited", "inhibiting" → "inhibits"
- "treat", "treated", "treating", "therapeutic" → "treats"
- "cause", "caused", "causing" → "causes"
- "activate", "activated", "activating", "stimulate" → "activates"
- "regulate", "regulated", "regulating", "control" → "regulates"
- "associate", "associated", "correlated" → "associated_with"
- "interact", "interacted", "bind", "binding" → "interacts_with"
- "increase", "increased", "elevate", "upregulate" → "increases"
- "decrease", "decreased", "reduce", "downregulate" → "decreases"

QUALITY STANDARDS:
- Preserve semantic meaning during normalization
- Maintain biological accuracy of relationships
- Use most specific appropriate category
- Ensure consistency across similar entities/relationships
- Handle ambiguous cases conservatively

PROCESSING RULES:
- Apply transformations systematically
- Document any ambiguous mappings
- Preserve original confidence scores
- Maintain entity-relationship correspondence"""
        super().__init__(client, model_name, system_prompt)

    def process(self, entities: List[KGEntity], relationships: List[KnowledgeTriple]) -> Tuple[List[KGEntity], List[KnowledgeTriple]]:
        """Align entities and relationships to standard schema."""
        aligned_entities = self.align_entities(entities)[0]
        aligned_relationships = self.align_relationships(relationships)
        return aligned_entities, aligned_relationships

    def align_entities(self, entities: List[KGEntity]) -> Tuple[List[KGEntity], int, int, float]:
        """Classify entity types using standard biomedical categories."""
        for entity in entities:
            if entity.entity_type == "Unknown":
                entity.entity_type = self._classify_entity_type(entity.name)
        return entities, 0, 0, 0.0

    def _classify_entity_type(self, entity_name: str) -> str:
        """Simple rule-based entity type classification."""
        name_lower = entity_name.lower()

        # Drug patterns
        if any(suffix in name_lower for suffix in ['mycin', 'cillin', 'statin', 'inhibitor']):
            return "Drug"

        # Gene patterns
        if entity_name.isupper() and len(entity_name) <= 10:
            return "Gene"

        # Protein patterns
        if any(suffix in name_lower for suffix in ['ase', 'receptor', 'protein']):
            return "Protein"

        # Disease patterns
        if any(keyword in name_lower for keyword in ['cancer', 'disease', 'syndrome', 'disorder']):
            return "Disease"

        return "Chemical"  # Default

    def align_relationships(self, triples: List[KnowledgeTriple]) -> List[KnowledgeTriple]:
        """Normalize relationship labels."""
        for triple in triples:
            triple.relation = self._normalize_relation(triple.relation)
        return triples

    def _normalize_relation(self, relation: str) -> str:
        """Standardize relation labels."""
        synonyms = {
            "inhibit": "inhibits", "inhibited": "inhibits",
            "treat": "treats", "treated": "treats",
            "cause": "causes", "caused": "causes",
            "activate": "activates", "activates": "activates",
            "associated with": "associated_with",
            "interacts with": "interacts_with"
        }
        return synonyms.get(relation.lower(), relation.lower())