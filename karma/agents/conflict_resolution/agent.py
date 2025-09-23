"""Conflict Resolution Agent Implementation"""

import logging
from typing import List, Tuple, Optional
from karma.core.base_agent import BaseAgent
from karma.core.data_structures import KnowledgeTriple

logger = logging.getLogger(__name__)


class ConflictResolutionAgent(BaseAgent):
    """Conflict Resolution Agent (CRA) for handling contradictory knowledge."""

    def __init__(self, client, model_name: str):
        system_prompt = """You are a specialized Conflict Resolution Agent for biomedical knowledge graphs. Your task is to identify and resolve contradictory relationships between biomedical entities using evidence-based decision making.

OBJECTIVE: Detect contradictory knowledge triples and resolve conflicts by selecting the most reliable and evidence-supported relationships.

CONFLICT TYPES:
1. DIRECT CONTRADICTIONS:
   - "Drug A treats Disease B" vs "Drug A causes Disease B"
   - "Protein X inhibits Protein Y" vs "Protein X activates Protein Y"
   - "Gene A increases Risk B" vs "Gene A decreases Risk B"

2. SEMANTIC CONFLICTS:
   - Opposing directional relationships (increases vs decreases)
   - Conflicting therapeutic effects (treats vs contraindicated)
   - Contradictory regulatory effects (upregulates vs downregulates)

RESOLUTION STRATEGY:
Priority factors (highest to lowest):
1. Confidence score of the relationship
2. Quality of supporting evidence
3. Specificity of the relationship statement
4. Recency of the source information
5. Consistency with established knowledge

DECISION RULES:
- Higher confidence score takes precedence
- Specific relationships override general ones
- Quantitative evidence beats qualitative claims
- Recent findings supersede older contradictory data
- Maintain relationships with strongest supporting evidence

PROCESSING APPROACH:
1. Identify potential conflicts between new and existing triples
2. Evaluate evidence quality for conflicting relationships
3. Apply resolution criteria systematically
4. Document resolution reasoning
5. Preserve non-conflicting relationships unchanged

OUTPUT REQUIREMENTS:
- Return only non-conflicting relationships
- Maintain all quality scores and metadata
- Preserve entity integrity
- Document conflict resolution decisions when applicable"""
        super().__init__(client, model_name, system_prompt)

    def process(self, new_triples: List[KnowledgeTriple], existing_triples: List[KnowledgeTriple]) -> List[KnowledgeTriple]:
        """Resolve conflicts between new and existing triples."""
        return self.resolve_conflicts(new_triples, existing_triples)[0]

    def resolve_conflicts(self, new_triples: List[KnowledgeTriple], existing_triples: List[KnowledgeTriple]) -> Tuple[List[KnowledgeTriple], int, int, float]:
        """Check for conflicts and resolve them."""
        final_triples = []

        for new_triple in new_triples:
            conflicting_triple = self._find_contradiction(new_triple, existing_triples)

            if conflicting_triple:
                # Simple resolution: keep higher confidence triple
                if new_triple.confidence > conflicting_triple.confidence:
                    final_triples.append(new_triple)
            else:
                final_triples.append(new_triple)

        return final_triples, 0, 0, 0.0

    def _find_contradiction(self, new_triple: KnowledgeTriple, existing_triples: List[KnowledgeTriple]) -> Optional[KnowledgeTriple]:
        """Find contradicting triples."""
        contradiction_pairs = {
            ("treats", "causes"), ("inhibits", "activates"),
            ("increases", "decreases"), ("upregulates", "downregulates")
        }

        for existing in existing_triples:
            if (existing.head.lower() == new_triple.head.lower() and
                existing.tail.lower() == new_triple.tail.lower()):

                rel_pair = (existing.relation, new_triple.relation)
                if rel_pair in contradiction_pairs or rel_pair[::-1] in contradiction_pairs:
                    return existing

        return None