"""
Data loader for Obsidian markdown files.

This module provides functions to load, chunk, and embed Obsidian
markdown notes for vector storage and RAG operations.
"""

from pathlib import Path
from openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
import frontmatter
from dotenv import load_dotenv
from typing import List

load_dotenv()

client = OpenAI()
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_markdown(path: str) -> List[str]:
    """
    Load and chunk an Obsidian markdown file.

    Parses the markdown file, extracts content (removing frontmatter),
    and splits it into chunks for embedding.

    Args:
        path: Path to the markdown file (.md)

    Returns:
        List of text chunks ready for embedding

    Example:
        >>> chunks = load_and_chunk_markdown("notes/study.md")
        >>> print(f"Created {len(chunks)} chunks")
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {path}")

    if not file_path.suffix == ".md":
        raise ValueError(
            f"File must be a markdown file (.md), got: {file_path.suffix}"
        )

    # Read and parse frontmatter
    with open(file_path, 'r', encoding='utf-8') as f:
        post = frontmatter.load(f)

    # Extract content (frontmatter is automatically removed)
    content = post.content

    # Split into chunks
    chunks = splitter.split_text(content)

    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks using OpenAI.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (each is a list of floats)

    Example:
        >>> chunks = ["First chunk", "Second chunk"]
        >>> embeddings = embed_texts(chunks)
        >>> print(f"Generated {len(embeddings)} embeddings")
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]

