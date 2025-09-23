"""
Entity Extraction Agent Module

The Entity Extraction Agent (EEA) is responsible for:
1. Identifying biomedical entities in summarized text
2. Classifying entity types (Disease, Drug, Gene, Protein, etc.)
3. Linking entities to ontologies where possible
"""

from .agent import EntityExtractionAgent

__all__ = ['EntityExtractionAgent']