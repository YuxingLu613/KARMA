"""
Summarizer Agent Implementation

The Summarizer Agent converts high-relevance text segments into concise summaries
while preserving technical details crucial for knowledge extraction.
"""

import logging
from typing import List, Tuple

from karma.core.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SummarizerAgent(BaseAgent):
    """
    Summarizer Agent (SA) for text segment summarization.

    This agent:
    1. Converts high-relevance segments into concise summaries
    2. Preserves technical details (gene symbols, chemical names, numeric data)
    3. Maintains entity relationships and quantitative information
    4. Filters out very low relevance content
    """

    def __init__(self, client, model_name: str):
        """
        Initialize the Summarizer Agent.

        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
        """
        system_prompt = """You are a specialized Summarizer Agent for biomedical literature processing. Your task is to create concise, information-dense summaries that preserve all knowledge-relevant content.

OBJECTIVE: Transform text segments into concise summaries (≤100 words) while preserving all entities, relationships, and quantitative data essential for knowledge extraction.

PRESERVATION REQUIREMENTS:
MANDATORY TO RETAIN:
- All biomedical entities (genes, proteins, drugs, diseases, chemicals)
- Exact terminology (IL-6, p53, BRCA1, aspirin, etc.)
- Quantitative measurements (IC50 values, p-values, concentrations, percentages)
- Relationship indicators (inhibits, activates, treats, causes, correlates with)
- Statistical significance markers (p<0.05, 95% CI, fold-change)
- Dosages, timeframes, and experimental conditions

LINGUISTIC REQUIREMENTS:
- Use precise scientific language
- Maintain causal relationships and logical flow
- Preserve technical specificity without oversimplification
- Include contextual qualifiers (significant, moderate, slight)
- Keep comparative language (more than, less than, similar to)

SUMMARIZATION STRATEGY:
1. Identify core knowledge claims
2. Extract all named entities
3. Preserve relationship statements
4. Condense methodology while keeping key parameters
5. Maintain quantitative precision

OUTPUT FORMAT:
- Single paragraph, maximum 100 words
- Complete sentences with proper scientific grammar
- No bullet points or fragmented phrases
- Preserve original terminology exactly as written

EXAMPLE:
Input: "Treatment with aspirin (100mg daily) significantly reduced inflammatory markers in skeletal muscle. COX-2 activity decreased by 45% (p<0.001) compared to placebo group. PGE2 levels were also significantly reduced (2.1 ± 0.3 vs 4.2 ± 0.5 ng/ml, p<0.05)."

Output: "Aspirin treatment (100mg daily) significantly reduced inflammatory markers in skeletal muscle by decreasing COX-2 activity 45% (p<0.001) versus placebo. PGE2 levels were significantly reduced (2.1 ± 0.3 vs 4.2 ± 0.5 ng/ml, p<0.05)."

QUALITY CONTROL:
- Never invent or modify entity names
- Never approximate exact numerical values
- Never omit statistical significance indicators
- Never simplify technical relationships"""

        super().__init__(client, model_name, system_prompt)

    def process(self, segments: List[str], relevance_threshold: float = 0.2) -> List[str]:
        """
        Summarize a list of text segments.

        Args:
            segments: List of text segments to summarize
            relevance_threshold: Minimum relevance to process segment

        Returns:
            List of summaries
        """
        summaries = []

        for segment in segments:
            if isinstance(segment, dict):
                text = segment.get('text', '')
                relevance = segment.get('score', 1.0)
            else:
                text = segment
                relevance = 1.0  # Assume relevant if no score provided

            if relevance < relevance_threshold:
                summaries.append("[OMITTED - Low Relevance]")
                continue

            summary = self._summarize_single_segment(text)
            summaries.append(summary)

        return summaries

    def _summarize_single_segment(self, text: str) -> str:
        """
        Summarize a single text segment.

        Args:
            text: Text segment to summarize

        Returns:
            Summarized text
        """
        # Skip very short segments
        if len(text.split()) < 15:
            return text  # Return as-is if too short to meaningfully summarize

        prompt = f"""
        Summarize the following biomedical text in 2-4 sentences, keeping it under 100 words.

        Critical Requirements:
        - Retain ALL technical terms (genes, proteins, drugs, diseases, chemicals)
        - Preserve ALL numeric data (concentrations, p-values, percentages, doses)
        - Keep relationship indicators (inhibits, activates, treats, causes, etc.)
        - Maintain scientific precision and accuracy
        - Use clear, unambiguous language

        If the text contains very little scientific information, provide a brief summary or return "[LOW CONTENT]".

        Provide only the summary with no additional text, formatting, or explanations.

        Text to summarize:
        {text}
        """

        try:
            summary, _, _, _ = self._make_llm_call(prompt, temperature=0.2)

            # Basic validation and cleanup
            summary = summary.strip()

            # Handle empty or low-quality responses
            if not summary or summary.lower() in ['[low content]', 'low content', 'n/a']:
                # Fallback: extract key sentences
                return self._extract_key_sentences(text)

            # Ensure summary is within length limit
            if len(summary.split()) > 120:  # Allow some flexibility
                # Truncate while preserving sentence structure
                sentences = summary.split('. ')
                truncated_summary = ""
                word_count = 0

                for sentence in sentences:
                    sentence_words = len(sentence.split())
                    if word_count + sentence_words <= 100:
                        truncated_summary += sentence + ". "
                        word_count += sentence_words
                    else:
                        break

                summary = truncated_summary.strip()

            return summary

        except Exception as e:
            logger.warning(f"Summarization failed for segment: {str(e)}")
            return self._extract_key_sentences(text)

    def _extract_key_sentences(self, text: str) -> str:
        """
        Fallback method to extract key sentences when LLM summarization fails.

        Args:
            text: Original text

        Returns:
            Key sentences extracted from text
        """
        sentences = text.split('. ')

        # Score sentences based on biomedical content
        scored_sentences = []

        for sentence in sentences:
            score = self._score_sentence_importance(sentence)
            scored_sentences.append((sentence, score))

        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # Select top sentences within word limit
        selected_sentences = []
        word_count = 0

        for sentence, score in scored_sentences:
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= 100 and score > 0.3:
                selected_sentences.append(sentence)
                word_count += sentence_words

        if selected_sentences:
            return '. '.join(selected_sentences) + '.'
        else:
            # Return first 100 words as last resort
            words = text.split()[:100]
            return ' '.join(words) + '...'

    def _score_sentence_importance(self, sentence: str) -> float:
        """
        Score the importance of a sentence for biomedical knowledge extraction.

        Args:
            sentence: Sentence to score

        Returns:
            Importance score (0-1)
        """
        sentence_lower = sentence.lower()
        score = 0.0

        # High-value biomedical terms
        high_value_terms = [
            'inhibit', 'activate', 'regulate', 'express', 'bind', 'interact',
            'cause', 'treat', 'prevent', 'induce', 'suppress', 'enhance',
            'protein', 'gene', 'enzyme', 'receptor', 'pathway', 'mechanism',
            'disease', 'cancer', 'tumor', 'therapy', 'treatment', 'drug',
            'significant', 'increase', 'decrease', 'effect', 'response'
        ]

        # Count high-value terms
        for term in high_value_terms:
            if term in sentence_lower:
                score += 0.1

        # Bonus for numeric data
        import re
        if re.search(r'\d+\.?\d*\s*(%|mg|μg|ng|mM|μM|nM|p\s*[<>=])', sentence):
            score += 0.3

        # Bonus for entity mentions (capitalized terms)
        capitalized_words = re.findall(r'\b[A-Z][A-Za-z0-9-]+\b', sentence)
        score += len(capitalized_words) * 0.05

        # Penalty for very short or very long sentences
        word_count = len(sentence.split())
        if word_count < 5:
            score *= 0.5
        elif word_count > 50:
            score *= 0.8

        return min(1.0, score)

    def summarize_segment(self, segment: str) -> Tuple[str, int, int, float]:
        """
        Legacy method for backward compatibility.

        Args:
            segment: Text segment to summarize

        Returns:
            Tuple of (summary, prompt_tokens, completion_tokens, processing_time)
        """
        summary = self._summarize_single_segment(segment)
        return summary, 0, 0, 0.0  # Token counts handled internally now