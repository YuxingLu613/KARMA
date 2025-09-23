"""
Relationship Extraction Agent Implementation

The Relationship Extraction Agent identifies relationships between entities
using contextual analysis and domain knowledge.
"""

import logging
from typing import List, Tuple
import json

from karma.core.base_agent import BaseAgent
from karma.core.data_structures import KGEntity, KnowledgeTriple

logger = logging.getLogger(__name__)


class RelationshipExtractionAgent(BaseAgent):
    """
    Relationship Extraction Agent (REA) for identifying entity relationships.

    This agent:
    1. Identifies relationships between extracted entities
    2. Classifies relationship types using domain patterns
    3. Handles negation and conditional relationships
    4. Provides confidence scores for extracted relationships
    """

    def __init__(self, client, model_name: str):
        """
        Initialize the Relationship Extraction Agent.

        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
        """
        system_prompt = """You are a specialized Relationship Extraction Agent for biomedical knowledge graphs. Your task is to identify precise relationships between biomedical entities with high accuracy and appropriate confidence scoring.

OBJECTIVE: Extract explicit relationships between biomedical entities from text, focusing on scientifically validated interactions and associations.

RELATIONSHIP TYPES:
1. TREATS: Drug/intervention treats disease/condition
   - Examples: "aspirin treats headache", "chemotherapy treats cancer"
   - Indicators: treats, cures, alleviates, therapeutic for

2. INHIBITS: Entity blocks or reduces activity/function
   - Examples: "aspirin inhibits COX-2", "statins inhibit cholesterol synthesis"
   - Indicators: inhibits, blocks, suppresses, reduces activity

3. ACTIVATES: Entity stimulates or increases activity/function
   - Examples: "insulin activates glucose uptake", "growth factors activate cell division"
   - Indicators: activates, stimulates, enhances, upregulates

4. CAUSES: Entity directly causes condition/effect
   - Examples: "smoking causes lung cancer", "mutations cause disease"
   - Indicators: causes, leads to, results in, triggers

5. ASSOCIATED_WITH: Statistical or observational association
   - Examples: "obesity associated with diabetes", "gene variants associated with risk"
   - Indicators: associated with, correlated with, linked to, related to

6. REGULATES: Entity controls expression/activity of another
   - Examples: "transcription factors regulate gene expression"
   - Indicators: regulates, controls, modulates, governs

7. INCREASES/DECREASES: Entity raises/lowers levels or activity
   - Examples: "exercise increases insulin sensitivity", "age decreases bone density"
   - Indicators: increases, raises, decreases, reduces, elevates

8. INTERACTS_WITH: Direct molecular interaction
   - Examples: "protein A interacts with protein B", "drug binds receptor"
   - Indicators: interacts with, binds to, complexes with

EXTRACTION CRITERIA:
REQUIRED CONDITIONS:
- Both entities must be explicitly mentioned in text
- Relationship must be explicitly stated or clearly implied
- Evidence must be present within the analyzed text segment
- Relationship direction must be determinable from context

CONFIDENCE SCORING:
HIGH CONFIDENCE (0.8-1.0):
- Direct experimental evidence stated
- Quantitative measurements provided
- Established scientific facts
- Clear causal language

MODERATE CONFIDENCE (0.5-0.7):
- Observational evidence
- Statistical associations
- Literature citations mentioned
- Qualified statements (may, appears to, suggests)

LOW CONFIDENCE (0.3-0.4):
- Preliminary findings
- Hypothetical relationships
- Weak associations
- Speculative statements

QUALITY CONTROLS:
- Ignore negated relationships ("does not treat", "no association")
- Avoid circular relationships (A-B, B-A unless distinct)
- Prioritize specific over general relationships
- Ensure entity names match exactly from entity extraction
- Include supporting evidence text

OUTPUT FORMAT:
Return only valid JSON array:
[
  {
    "head": "exact_entity_name",
    "relation": "RELATIONSHIP_TYPE",
    "tail": "exact_entity_name",
    "confidence": 0.85,
    "evidence": "direct quote supporting relationship"
  }
]

EXAMPLE:
Text: "Aspirin significantly inhibited COX-2 activity (p<0.001), reducing PGE2 production by 60%."
Entities: ["aspirin", "COX-2", "PGE2"]
Output:
[
  {"head": "aspirin", "relation": "INHIBITS", "tail": "COX-2", "confidence": 0.95, "evidence": "Aspirin significantly inhibited COX-2 activity (p<0.001)"},
  {"head": "aspirin", "relation": "DECREASES", "tail": "PGE2", "confidence": 0.90, "evidence": "reducing PGE2 production by 60%"}
]"""

        super().__init__(client, model_name, system_prompt)

    def process(self, summaries: List[str], entities: List[KGEntity]) -> List[KnowledgeTriple]:
        """
        Extract relationships from summaries using the provided entities.

        Args:
            summaries: List of text summaries
            entities: List of entities to find relationships between

        Returns:
            List of extracted knowledge triples
        """
        if not entities:
            return []

        all_relationships = []

        for summary in summaries:
            if summary and summary not in ["[OMITTED - Low Relevance]", "[LOW CONTENT]"]:
                relationships = self._extract_relationships_from_text(summary, entities)
                all_relationships.extend(relationships)

        # Deduplicate relationships
        unique_relationships = self._deduplicate_relationships(all_relationships)

        return unique_relationships

    def _extract_relationships_from_text(self, text: str, entities: List[KGEntity]) -> List[KnowledgeTriple]:
        """
        Extract relationships from text using LLM analysis.

        Args:
            text: Text to analyze
            entities: List of entities to consider

        Returns:
            List of KnowledgeTriple objects
        """
        if len(entities) < 2:  # Need at least 2 entities for relationships
            return []

        # Create entity reference for the prompt
        entity_names = [ent.name for ent in entities]
        entity_bullets = "\n".join(f"- {name}" for name in entity_names)

        prompt = f"""
        Entities of interest:
        {entity_bullets}

        From the text below, identify direct relationships between these entities.
        Only extract relationships that are explicitly stated or clearly implied in the text.

        Text to analyze:
        {text}

        Return only a JSON array of relationships:
        """

        try:
            response, _, _, _ = self._make_llm_call(prompt, temperature=0.1)

            # Parse JSON response
            relations_data = self._parse_json_response(response)

            triples = []
            for rel_data in relations_data:
                if isinstance(rel_data, dict) and all(key in rel_data for key in ["head", "relation", "tail"]):
                    # Validate that entities exist in our entity list
                    head = rel_data.get("head", "").strip()
                    tail = rel_data.get("tail", "").strip()

                    if self._entity_exists(head, entity_names) and self._entity_exists(tail, entity_names):
                        confidence = float(rel_data.get("confidence", 0.5))
                        evidence = rel_data.get("evidence", "")

                        # Estimate clarity and relevance
                        clarity = self._estimate_clarity(head, rel_data.get("relation", ""), tail)
                        relevance = self._estimate_relevance(head, rel_data.get("relation", ""), tail)

                        triple = KnowledgeTriple(
                            head=head,
                            relation=rel_data.get("relation", "").strip(),
                            tail=tail,
                            confidence=confidence,
                            clarity=clarity,
                            relevance=relevance,
                            source="relationship_extraction"
                        )
                        triples.append(triple)

            return triples

        except Exception as e:
            logger.warning(f"Relationship extraction failed: {str(e)}")
            return []

    def _entity_exists(self, entity_name: str, entity_list: List[str]) -> bool:
        """
        Check if entity name exists in the entity list (case-insensitive).

        Args:
            entity_name: Name to check
            entity_list: List of entity names

        Returns:
            True if entity exists
        """
        entity_lower = entity_name.lower()
        return any(ent.lower() == entity_lower for ent in entity_list)

    def _estimate_clarity(self, head: str, relation: str, tail: str) -> float:
        """
        Estimate clarity of a relationship triple.

        Args:
            head: Subject entity
            relation: Relationship type
            tail: Object entity

        Returns:
            Clarity score (0-1)
        """
        clarity = 0.5  # Base score

        # Higher clarity for specific relations
        specific_relations = ["treats", "inhibits", "activates", "causes", "regulates"]
        if relation.lower() in specific_relations:
            clarity += 0.2

        # Higher clarity for well-defined entities (not too generic)
        generic_terms = ["protein", "gene", "drug", "disease", "chemical"]
        if head.lower() not in generic_terms:
            clarity += 0.1
        if tail.lower() not in generic_terms:
            clarity += 0.1

        # Lower clarity for very general relations
        if relation.lower() in ["associated_with", "interacts_with"]:
            clarity -= 0.1

        return min(1.0, max(0.1, clarity))

    def _estimate_relevance(self, head: str, relation: str, tail: str) -> float:
        """
        Estimate biomedical relevance of a relationship triple.

        Args:
            head: Subject entity
            relation: Relationship type
            tail: Object entity

        Returns:
            Relevance score (0-1)
        """
        relevance = 0.5  # Base score

        # Higher relevance for therapeutic relationships
        therapeutic_relations = ["treats", "prevents", "causes", "inhibits", "activates"]
        if relation.lower() in therapeutic_relations:
            relevance += 0.2

        # Higher relevance for disease-related triples
        disease_keywords = ["cancer", "disease", "disorder", "syndrome", "infection"]
        if any(keyword in head.lower() or keyword in tail.lower() for keyword in disease_keywords):
            relevance += 0.1

        # Higher relevance for drug-related triples
        drug_keywords = ["drug", "medication", "inhibitor", "agonist", "antagonist", "therapy"]
        if any(keyword in head.lower() or keyword in tail.lower() for keyword in drug_keywords):
            relevance += 0.1

        return min(1.0, max(0.1, relevance))

    def _deduplicate_relationships(self, relationships: List[KnowledgeTriple]) -> List[KnowledgeTriple]:
        """
        Remove duplicate relationships.

        Args:
            relationships: List of relationships to deduplicate

        Returns:
            List of unique relationships
        """
        if not relationships:
            return []

        unique_rels = {}

        for rel in relationships:
            # Create key for deduplication
            key = f"{rel.head.lower()}__{rel.relation.lower()}__{rel.tail.lower()}"

            # Keep the one with higher confidence if duplicates exist
            if key not in unique_rels or rel.confidence > unique_rels[key].confidence:
                unique_rels[key] = rel

        return list(unique_rels.values())

    def extract_relationships(self, text: str, entities: List[KGEntity]) -> Tuple[List[KnowledgeTriple], int, int, float]:
        """
        Legacy method for backward compatibility.

        Args:
            text: Source text
            entities: List of entities to find relationships between

        Returns:
            Tuple of (relationship_list, prompt_tokens, completion_tokens, processing_time)
        """
        relationships = self._extract_relationships_from_text(text, entities)
        return relationships, 0, 0, 0.0  # Token counts handled internally now