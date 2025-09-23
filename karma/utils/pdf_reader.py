"""
PDF reading utilities for KARMA.

This module provides functionality to extract text from PDF files
with error handling and optimization for academic papers.
"""

import logging
from pathlib import Path
from typing import Optional

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

logger = logging.getLogger(__name__)


class PDFReader:
    """
    PDF text extraction utility with error handling.

    This class provides robust PDF text extraction with fallback
    mechanisms and special handling for academic papers.
    """

    def __init__(self):
        """Initialize PDF reader."""
        if PyPDF2 is None:
            logger.warning("PyPDF2 not installed. PDF reading will be limited.")

    def extract_text(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF reading fails
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required for PDF reading. Install with: pip install PyPDF2")

        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    logger.warning(f"PDF {pdf_path} is encrypted. Attempting to decrypt...")
                    try:
                        pdf_reader.decrypt('')  # Try empty password
                    except Exception as e:
                        logger.error(f"Failed to decrypt PDF: {e}")
                        raise

                # Extract text from all pages
                total_pages = len(pdf_reader.pages)
                logger.info(f"Extracting text from {total_pages} pages...")

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\\n\\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue

                if not text.strip():
                    logger.warning(f"No text extracted from PDF: {pdf_path}")
                    return ""

                # Post-process the extracted text
                text = self._post_process_text(text)

                logger.info(f"Successfully extracted {len(text)} characters from PDF")
                return text

        except Exception as e:
            logger.error(f"Failed to read PDF {pdf_path}: {str(e)}")
            raise

    def _post_process_text(self, text: str) -> str:
        """
        Post-process extracted text to improve quality.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""

        # Remove excessive whitespace
        import re
        text = re.sub(r'\\n\\s*\\n', '\\n\\n', text)  # Normalize paragraph breaks
        text = re.sub(r'[ \\t]+', ' ', text)  # Normalize spaces
        text = re.sub(r'\\n ', '\\n', text)  # Remove spaces at line starts

        # Fix common PDF extraction issues
        # Remove hyphenation at line breaks
        text = re.sub(r'-\\n([a-z])', r'\\1', text)

        # Fix broken words across lines
        text = re.sub(r'([a-z])\\n([a-z])', r'\\1\\2', text)

        # Clean up extra spaces
        text = re.sub(r'\\s+', ' ', text)

        return text.strip()

    def is_pdf_readable(self, pdf_path: Path) -> bool:
        """
        Check if a PDF file can be read.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            True if PDF is readable
        """
        try:
            self.extract_text(pdf_path)
            return True
        except Exception:
            return False

    def get_pdf_info(self, pdf_path: Path) -> Optional[dict]:
        """
        Get metadata information from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary with PDF metadata or None if failed
        """
        if PyPDF2 is None:
            return None

        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                info = {
                    'pages': len(pdf_reader.pages),
                    'encrypted': pdf_reader.is_encrypted,
                    'metadata': {}
                }

                # Extract metadata if available
                if pdf_reader.metadata:
                    metadata = pdf_reader.metadata
                    info['metadata'] = {
                        'title': metadata.get('/Title', ''),
                        'author': metadata.get('/Author', ''),
                        'subject': metadata.get('/Subject', ''),
                        'creator': metadata.get('/Creator', ''),
                        'producer': metadata.get('/Producer', ''),
                        'creation_date': str(metadata.get('/CreationDate', '')),
                        'modification_date': str(metadata.get('/ModDate', ''))
                    }

                return info

        except Exception as e:
            logger.error(f"Failed to get PDF info: {e}")
            return None