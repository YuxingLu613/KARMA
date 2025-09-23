"""
KARMA Core Module

This module contains the core data structures and base classes for the KARMA framework.
"""

from .data_structures import KnowledgeTriple, KGEntity, IntermediateOutput
from .base_agent import BaseAgent
from .pipeline import KARMAPipeline

__all__ = [
    'KnowledgeTriple',
    'KGEntity',
    'IntermediateOutput',
    'BaseAgent',
    'KARMAPipeline'
]