"""
Ingestion Agent Module

The Ingestion Agent (IA) is responsible for:
1. Retrieving and standardizing raw documents (PDF, text)
2. Extracting metadata such as title, authors, journal, publication date
3. Normalizing text format and handling OCR artifacts
"""

from .agent import IngestionAgent

__all__ = ['IngestionAgent']