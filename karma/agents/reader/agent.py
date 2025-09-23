"""
Reader Agent Implementation

The Reader Agent handles document segmentation and relevance scoring
for efficient downstream processing.
"""

import logging
from typing import List, Dict, Tuple
import re

from karma.core.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ReaderAgent(BaseAgent):
    """
    Reader Agent (RA) for document segmentation and relevance scoring.

    This agent:
    1. Splits normalized text into logical segments (paragraph-level chunks)
    2. Assigns relevance scores to each segment based on domain knowledge
    3. Filters content based on relevance thresholds
    4. Handles structural document elements (headers, sections, etc.)
    """

    def __init__(self, client, model_name: str):
        """
        Initialize the Reader Agent.

        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
        """
        system_prompt = """You are a specialized Reader Agent for biomedical literature analysis. Your task is to evaluate text segments and assign precise relevance scores for knowledge extraction.

OBJECTIVE: Score text segments (0.0-1.0) based on their potential to contain extractable biomedical knowledge relationships.

SCORING CRITERIA:
HIGH RELEVANCE (0.8-1.0):
- Results sections with quantitative findings
- Discussion of mechanisms, pathways, or interactions
- Novel discoveries or experimental outcomes
- Specific drug-disease relationships
- Quantitative data (IC50, p-values, concentrations)

MODERATE RELEVANCE (0.4-0.7):
- Background information with entity relationships
- Clinical observations or case descriptions
- General statements about biological processes
- Abstract summaries of findings
- Introduction sections with known facts

LOW RELEVANCE (0.1-0.3):
- Methodological descriptions without outcomes
- Acknowledgments, funding information
- Pure statistical analysis without interpretation
- Equipment descriptions or protocols
- Author affiliations and contact information

MINIMAL RELEVANCE (0.0-0.1):
- References and citations
- Copyright notices
- Figure/table captions without data
- Purely administrative content

SCORING REQUIREMENTS:
- Return only a single float value between 0.0 and 1.0
- Consider biomedical terminology density
- Prioritize content describing relationships between entities
- Account for presence of specific measurements or outcomes
- Evaluate potential for entity-relationship extraction

EXAMPLES:
"IL-6 significantly reduced tumor growth (p<0.001) compared to control" → 0.95
"Patients were enrolled according to institutional guidelines" → 0.15
"Aspirin inhibits COX-2 through covalent binding" → 0.90
"Statistical analysis was performed using SPSS software" → 0.20"""

        super().__init__(client, model_name, system_prompt)

    def process(self, content: str, relevance_threshold: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
        """
        Segment content and score relevance of each segment.

        Args:
            content: Text content to segment
            relevance_threshold: Minimum relevance score to keep segments

        Returns:
            Tuple of (all_segments, relevant_segments)
        """
        # Split into logical segments
        segments = self._split_into_segments(content)

        # Score relevance for each segment
        scored_segments = self._score_segments_batch(segments)

        # Filter relevant segments
        relevant_segments = [
            seg for seg in scored_segments
            if seg.get('score', 0.0) >= relevance_threshold
        ]

        return scored_segments, relevant_segments

    def _split_into_segments(self, content: str) -> List[Dict]:
        """
        Split content into logical segments based on structure.

        Args:
            content: Content to segment

        Returns:
            List of segment dictionaries
        """
        segments = []

        # Split on double newlines (paragraph breaks)
        raw_segments = content.split('\n\n')

        for i, segment_text in enumerate(raw_segments):
            segment_text = segment_text.strip()
            if not segment_text:
                continue

            # Determine section type based on content
            section_type = self._identify_section_type(segment_text)

            segment = {
                'text': segment_text,
                'score': 0.0,  # Will be filled by scoring
                'section': section_type,
                'position': i,
                'word_count': len(segment_text.split())
            }

            segments.append(segment)

        return segments

    def _identify_section_type(self, text: str) -> str:
        """
        Identify the type of section based on content patterns.

        Args:
            text: Segment text to analyze

        Returns:
            Section type identifier
        """
        text_lower = text.lower()

        # Common section headers
        section_patterns = {
            'abstract': [r'^abstract\b', r'summary\b'],
            'introduction': [r'^introduction\b', r'^background\b'],
            'methods': [r'^methods?\b', r'^methodology\b', r'^materials\b', r'^experimental\b'],
            'results': [r'^results?\b', r'^findings\b', r'^outcomes?\b'],
            'discussion': [r'^discussion\b', r'^conclusion\b', r'^implications\b'],
            'references': [r'^references?\b', r'^bibliography\b', r'^\d+\.\s+\w+.*et al'],
            'acknowledgments': [r'^acknowledgments?\b', r'^acknowledgements?\b'],
            'funding': [r'^funding\b', r'^grants?\b', r'^financial\b'],
            'supplementary': [r'^supplement', r'^appendix\b', r'^additional\b']
        }

        for section_type, patterns in section_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return section_type

        # If no specific section identified, categorize by content characteristics
        if len(text.split()) < 10:
            return 'header'
        elif re.search(r'\d+\.\s+\w+.*\(\d{4}\)', text):  # Citation pattern
            return 'references'
        elif text.startswith(('Figure', 'Table', 'Fig.')):
            return 'figure_caption'
        else:
            return 'content'

    def _score_segments_batch(self, segments: List[Dict]) -> List[Dict]:
        """
        Score relevance for multiple segments efficiently.

        Args:
            segments: List of segments to score

        Returns:
            List of segments with relevance scores
        """
        # Process segments in batches to be more efficient
        batch_size = 5
        scored_segments = []

        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            scores = self._batch_score_relevance(batch)

            for j, segment in enumerate(batch):
                if j < len(scores):
                    segment['score'] = scores[j]
                else:
                    segment['score'] = self._get_default_score(segment)
                scored_segments.append(segment)

        return scored_segments

    def _batch_score_relevance(self, segments: List[Dict]) -> List[float]:
        """
        Score relevance for a batch of segments using LLM.

        Args:
            segments: List of segments to score

        Returns:
            List of relevance scores
        """
        if not segments:
            return []

        # Create batch prompt
        segment_texts = []
        for i, seg in enumerate(segments):
            section_info = f" [Section: {seg['section']}]" if seg.get('section') != 'content' else ""
            segment_texts.append(f"Segment {i+1}{section_info}:\n{seg['text'][:500]}...")

        prompt = f"""
        You are a biomedical text relevance scorer.
        Rate how relevant each of the following segments is (0.0 to 1.0) for extracting
        new biomedical knowledge (e.g., relationships between diseases, drugs, genes, proteins).

        Consider:
        - Results and discussion sections usually have higher relevance (0.7-0.9)
        - Abstract and introduction sections have moderate relevance (0.5-0.7)
        - Methods sections without findings have lower relevance (0.2-0.4)
        - References, acknowledgments have very low relevance (0.0-0.2)
        - Content with specific biomedical entities, relationships, or findings scores higher

        For each segment, return only a single float value between 0.0 and 1.0, with no other text.

        {chr(10).join(segment_texts)}

        Return one score per line, with no labels:
        """

        try:
            response, _, _, _ = self._make_llm_call(prompt, temperature=0.1)

            lines = response.strip().split('\n')
            scores = []

            for line in lines:
                score = self._extract_float_from_text(line.strip(), default=0.5)
                scores.append(score)

            # Ensure we have scores for all segments
            while len(scores) < len(segments):
                scores.append(0.5)

            return scores[:len(segments)]

        except Exception as e:
            logger.warning(f"Batch scoring failed: {str(e)}")
            return [self._get_default_score(seg) for seg in segments]

    def _get_default_score(self, segment: Dict) -> float:
        """
        Get default relevance score based on segment characteristics.

        Args:
            segment: Segment to score

        Returns:
            Default relevance score
        """
        section_type = segment.get('section', 'content')
        word_count = segment.get('word_count', 0)

        # Default scores based on section type
        section_scores = {
            'results': 0.8,
            'discussion': 0.7,
            'abstract': 0.6,
            'introduction': 0.5,
            'content': 0.5,
            'methods': 0.3,
            'references': 0.1,
            'acknowledgments': 0.1,
            'funding': 0.1,
            'supplementary': 0.2,
            'header': 0.2,
            'figure_caption': 0.4
        }

        base_score = section_scores.get(section_type, 0.5)

        # Adjust based on content length
        if word_count < 10:
            base_score *= 0.5  # Very short segments less relevant
        elif word_count > 100:
            base_score *= 1.1  # Longer segments might be more informative

        return min(1.0, max(0.0, base_score))

    def score_relevance(self, segment: str) -> Tuple[float, int, int, float]:
        """
        Legacy method for backward compatibility.

        Args:
            segment: Text segment to score

        Returns:
            Tuple of (relevance_score, prompt_tokens, completion_tokens, processing_time)
        """
        segment_dict = {'text': segment, 'section': 'content', 'word_count': len(segment.split())}
        scored_segments = self._score_segments_batch([segment_dict])

        if scored_segments:
            return scored_segments[0]['score'], 0, 0, 0.0
        else:
            return 0.5, 0, 0, 0.0

    def split_into_segments(self, content: str) -> List[Dict]:
        """
        Legacy method for backward compatibility.

        Args:
            content: Content to segment

        Returns:
            List of segment dictionaries
        """
        all_segments, _ = self.process(content)
        return all_segments