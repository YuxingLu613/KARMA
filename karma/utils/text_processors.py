"""
Text processing utilities for KARMA.

This module contains various text processing functions
for cleaning, normalizing, and preprocessing text data.
"""

import re
import logging
from typing import List, Dict, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Text processing utilities for document preprocessing.

    This class provides various text cleaning and normalization
    functions optimized for biomedical text processing.
    """

    def __init__(self):
        """Initialize text processor with biomedical patterns."""
        self.biomedical_abbreviations = self._load_biomedical_abbreviations()
        self.stopwords = self._load_stopwords()

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())

        # Normalize whitespace
        text = re.sub(r'\\s+', ' ', text)
        text = re.sub(r'\\n\\s*\\n', '\\n\\n', text)

        # Fix encoding issues
        encoding_fixes = {
            'â€™': "'",
            'â€œ': '"',
            'â€\\x9d': '"',
            'â€"': '-',
            'Ã¡': 'á',
            'Ã©': 'é',
            'Ã­': 'í',
            'Ã³': 'ó',
            'Ãº': 'ú'
        }

        for bad, good in encoding_fixes.items():
            text = text.replace(bad, good)

        return text.strip()

    def normalize_biomedical_text(self, text: str) -> str:
        """
        Normalize text specifically for biomedical content.

        Args:
            text: Text to normalize

        Returns:
            Normalized biomedical text
        """
        # Normalize Greek letters and symbols
        greek_mapping = {
            'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
            'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'θ': 'theta',
            'ι': 'iota', 'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu',
            'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron', 'π': 'pi',
            'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon',
            'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega'
        }

        for greek, latin in greek_mapping.items():
            text = text.replace(greek, latin)

        # Normalize units and symbols
        unit_mapping = {
            '°C': ' degrees Celsius',
            '°F': ' degrees Fahrenheit',
            '±': ' plus/minus ',
            '≤': ' less than or equal to ',
            '≥': ' greater than or equal to ',
            '→': ' leads to ',
            '←': ' derived from ',
            '↑': ' increases ',
            '↓': ' decreases '
        }

        for symbol, replacement in unit_mapping.items():
            text = text.replace(symbol, replacement)

        return text

    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text with biomedical context awareness.

        Args:
            text: Text to split into sentences

        Returns:
            List of sentences
        """
        # Handle abbreviations that shouldn't be sentence breaks
        protected_text = text

        # Protect common biomedical abbreviations
        for abbrev in self.biomedical_abbreviations:
            pattern = f"\\\\b{re.escape(abbrev)}\\\\.\\\\s+"
            protected_text = re.sub(pattern, f"{abbrev}[DOT] ", protected_text)

        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+\\s+', protected_text)

        # Restore protected abbreviations
        sentences = [sent.replace('[DOT]', '.') for sent in sentences]

        # Filter out very short sentences
        sentences = [sent.strip() for sent in sentences if len(sent.strip()) > 10]

        return sentences

    def remove_references(self, text: str) -> str:
        """
        Remove reference sections and citations from text.

        Args:
            text: Text containing references

        Returns:
            Text with references removed
        """
        # Remove reference sections
        ref_patterns = [
            r'\\n\\s*REFERENCES\\s*\\n.*$',
            r'\\n\\s*References\\s*\\n.*$',
            r'\\n\\s*Bibliography\\s*\\n.*$',
            r'\\n\\s*BIBLIOGRAPHY\\s*\\n.*$'
        ]

        for pattern in ref_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

        # Remove inline citations
        citation_patterns = [
            r'\\([^)]*\\d{4}[^)]*\\)',  # (Author, 2020)
            r'\\[\\d+(?:,\\s*\\d+)*\\]',  # [1, 2, 3]
            r'\\([^)]*et al\\.?[^)]*\\)',  # (Smith et al., 2020)
        ]

        for pattern in citation_patterns:
            text = re.sub(pattern, '', text)

        return text

    def extract_keywords(self, text: str, min_length: int = 3) -> Set[str]:
        """
        Extract potential keywords from biomedical text.

        Args:
            text: Text to extract keywords from
            min_length: Minimum keyword length

        Returns:
            Set of potential keywords
        """
        # Convert to lowercase for processing
        text_lower = text.lower()

        # Extract potential biomedical terms
        patterns = [
            r'\\b[a-z]+-?[0-9]+[a-z]*\\b',  # Gene symbols with numbers
            r'\\b[a-z]+(?:ase|in|ism|osis|itis|oma)\\b',  # Medical suffixes
            r'\\b(?:anti|pro|pre|post|sub|super|hyper|hypo)-[a-z]+\\b'  # Medical prefixes
        ]

        keywords = set()

        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            keywords.update(match for match in matches if len(match) >= min_length)

        # Remove stopwords
        keywords = keywords - self.stopwords

        return keywords

    def _load_biomedical_abbreviations(self) -> Set[str]:
        """Load common biomedical abbreviations."""
        abbreviations = {
            'Dr', 'Prof', 'vs', 'etc', 'i.e', 'e.g', 'cf', 'et al',
            'DNA', 'RNA', 'mRNA', 'tRNA', 'rRNA', 'miRNA',
            'PCR', 'qPCR', 'RT-PCR', 'ELISA', 'FACS', 'HPLC',
            'mg', 'μg', 'ng', 'kg', 'mL', 'μL', 'nL',
            'mM', 'μM', 'nM', 'pM', 'pH', 'pI',
            'min', 'hr', 'sec', 'mol', 'bp', 'kb', 'Mb',
            'Fig', 'Table', 'Eq', 'Ref', 'Suppl'
        }
        return abbreviations

    def _load_stopwords(self) -> Set[str]:
        """Load common English stopwords."""
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
            'that', 'the', 'to', 'was', 'were', 'will', 'with', 'the',
            'this', 'but', 'they', 'have', 'had', 'what', 'said', 'each',
            'which', 'she', 'do', 'how', 'their', 'if', 'up', 'out', 'many',
            'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make',
            'like', 'into', 'him', 'time', 'two', 'more', 'go', 'no', 'way',
            'could', 'my', 'than', 'first', 'been', 'call', 'who', 'oil',
            'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come',
            'made', 'may', 'part'
        }
        return stopwords