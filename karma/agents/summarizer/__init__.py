"""
Summarizer Agent Module

The Summarizer Agent (SA) is responsible for:
1. Converting high-relevance segments into concise summaries
2. Preserving technical details important for knowledge extraction
3. Maintaining entity relationships and quantitative data
"""

from .agent import SummarizerAgent

__all__ = ['SummarizerAgent']