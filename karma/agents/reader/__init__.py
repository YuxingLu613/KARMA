"""
Reader Agent Module

The Reader Agent (RA) is responsible for:
1. Segmenting normalized text into logical chunks
2. Scoring segment relevance using domain knowledge
3. Filtering non-relevant content (e.g., acknowledgments, references)
"""

from .agent import ReaderAgent

__all__ = ['ReaderAgent']