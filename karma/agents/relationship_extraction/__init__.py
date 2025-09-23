"""
Relationship Extraction Agent Module

The Relationship Extraction Agent (REA) is responsible for:
1. Identifying relationships between extracted entities
2. Classifying relationship types (treats, causes, inhibits, etc.)
3. Handling multi-label classification for overlapping relations
"""

from .agent import RelationshipExtractionAgent

__all__ = ['RelationshipExtractionAgent']