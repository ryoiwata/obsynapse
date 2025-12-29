"""
Data ingestion modules for ObSynapse.

This package contains modules for loading, processing, and preparing
Obsidian notes for vector storage and RAG operations.
"""

from .data_loader import (
    load_and_chunk_markdown,
    embed_texts,
    EMBED_MODEL,
    EMBED_DIM
)

__all__ = [
    "load_and_chunk_markdown",
    "embed_texts",
    "EMBED_MODEL",
    "EMBED_DIM"
]


