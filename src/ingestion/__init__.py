"""
Data ingestion modules for ObSynapse.

This package contains modules for loading, processing, and preparing
various document types (Markdown, PDF, etc.) for vector storage and RAG operations.
"""

from .pdf_extractor import PDFExtractor

__all__ = [
    "PDFExtractor",
]
