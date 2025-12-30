"""
Data ingestion modules for ObSynapse.

This package contains modules for loading, processing, and preparing
Obsidian notes for vector storage and RAG operations.
"""

from .data_loader import (
    load_and_chunk_markdown,
    embed_texts,
    chunk_from_structure,
    EMBED_MODEL,
    EMBED_DIM
)
from .structure_extractor import (
    extract_structure,
    parse_chapter_structure,
    is_numbered_subsection,
    DocumentStructure,
    Chapter,
    Subsection,
    Subhead,
    Block
)

__all__ = [
    "load_and_chunk_markdown",
    "embed_texts",
    "chunk_from_structure",
    "EMBED_MODEL",
    "EMBED_DIM",
    "extract_structure",
    "parse_chapter_structure",
    "is_numbered_subsection",
    "DocumentStructure",
    "Chapter",
    "Subsection",
    "Subhead",
    "Block"
]


