"""Evaluator Agent Implementation"""

import logging
from typing import List, Tuple
from karma.core.base_agent import BaseAgent
from karma.core.data_structures import KnowledgeTriple

logger = logging.getLogger(__name__)


class EvaluatorAgent(BaseAgent):
    """Evaluator Agent (EA) for final quality assessment and integration decisions."""

    def __init__(self, client, model_name: str, integrate_threshold: float = 0.6):
        system_prompt = """You are a specialized Evaluator Agent for biomedical knowledge graph integration. Your task is to assess the quality of extracted knowledge triples and make final integration decisions based on comprehensive quality metrics.

OBJECTIVE: Evaluate knowledge triples using multi-dimensional quality assessment and determine integration eligibility based on established thresholds.

EVALUATION DIMENSIONS:

1. CONFIDENCE (Weight: 50%):
   - Reliability of the extraction process
   - Strength of textual evidence
   - Consistency with domain knowledge
   - Statistical significance if applicable

2. CLARITY (Weight: 25%):
   - Linguistic precision of the relationship
   - Specificity of entity references
   - Absence of ambiguity
   - Definiteness of the relationship statement

3. RELEVANCE (Weight: 25%):
   - Biomedical significance
   - Potential impact on knowledge discovery
   - Novelty or confirmatory value
   - Alignment with domain priorities

INTEGRATION SCORING:
Integration Score = (0.5 × Confidence) + (0.25 × Clarity) + (0.25 × Relevance)

QUALITY THRESHOLDS:
- EXCELLENT (≥0.8): High-quality, well-supported relationships
- GOOD (0.6-0.79): Reliable relationships with minor limitations
- ACCEPTABLE (0.5-0.59): Adequate quality for integration
- QUESTIONABLE (0.3-0.49): Limited reliability, review required
- POOR (<0.3): Insufficient quality for integration

DECISION CRITERIA:
INTEGRATE IF:
- Integration score ≥ threshold (typically 0.6)
- All dimension scores ≥ 0.3 (minimum quality floor)
- No significant quality concerns identified
- Relationship adds value to knowledge base

REJECT IF:
- Integration score < threshold
- Any dimension score < 0.3
- Contradicts well-established facts
- Insufficient supporting evidence

PROCESSING REQUIREMENTS:
- Apply consistent evaluation standards
- Document quality assessment reasoning
- Maintain transparency in decision making
- Preserve original quality metrics
- Enable threshold adjustment for different use cases"""
        super().__init__(client, model_name, system_prompt)
        self.integrate_threshold = integrate_threshold

    def process(self, triples: List[KnowledgeTriple]) -> List[KnowledgeTriple]:
        """Evaluate and filter triples based on quality metrics."""
        return self.finalize_triples(triples)[0]

    def finalize_triples(self, candidate_triples: List[KnowledgeTriple]) -> Tuple[List[KnowledgeTriple], int, int, float]:
        """Filter triples based on integration threshold."""
        integrated_triples = []

        for triple in candidate_triples:
            # Ensure all metrics are set
            if triple.confidence <= 0:
                triple.confidence = 0.5
            if triple.clarity <= 0:
                triple.clarity = 0.5
            if triple.relevance <= 0:
                triple.relevance = 0.5

            # Calculate integration score
            integration_score = self._aggregate_scores(triple)

            # Keep triple if it meets threshold
            if integration_score >= self.integrate_threshold:
                integrated_triples.append(triple)

        return integrated_triples, 0, 0, 0.0

    def _aggregate_scores(self, triple: KnowledgeTriple) -> float:
        """Combine quality metrics into final score."""
        # Weighted average: confidence=50%, clarity=25%, relevance=25%
        return (0.5 * triple.confidence + 0.25 * triple.clarity + 0.25 * triple.relevance)