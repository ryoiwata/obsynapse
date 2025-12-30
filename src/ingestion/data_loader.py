"""
Data loader for Obsidian markdown files.

This module provides functions to load, chunk, and embed Obsidian
markdown notes for vector storage and RAG operations.
"""

import hashlib
import re
from pathlib import Path
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import frontmatter
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional

from .structure_extractor import (
    DocumentStructure,
    Block
)

load_dotenv()

client = OpenAI()
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Chunking constants
CHUNK_TARGET_MIN = 150  # Target minimum characters (~150 tokens)
CHUNK_TARGET_MAX = 350  # Target maximum characters (~350 tokens)
CHUNK_HARD_MAX = 3000  # Hard maximum characters per chunk
LEAD_IN_THRESHOLD = 240  # Max chars for lead-in paragraph
LIST_SPLIT_THRESHOLD = 5  # Split lists with >5 items
LIST_ITEM_LENGTH_THRESHOLD = 120  # Split per item if avg > 120 chars

# Sentence detection (simple heuristic)
SENTENCE_RE = re.compile(r'[.!?]+\s+')


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
    # RecursiveCharacterTextSplitter.split_text returns a list of strings
    chunks = splitter.split_text(content)

    # Ensure all chunks are strings and filter out empty chunks
    chunks = [
        str(chunk).strip()
        for chunk in chunks
        if chunk and str(chunk).strip()
    ]

    return chunks


def embed_texts(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks using OpenAI.
    Batches requests to handle large numbers of chunks efficiently.

    Args:
        texts: List of text strings to embed
        batch_size: Number of texts to embed per API request (default: 100)

    Returns:
        List of embedding vectors (each is a list of floats)

    Example:
        >>> chunks = ["First chunk", "Second chunk"]
        >>> embeddings = embed_texts(chunks)
        >>> print(f"Generated {len(embeddings)} embeddings")
    """
    all_embeddings = []

    # Process in batches to avoid rate limits and improve reliability
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def _is_short_paragraph(text: str) -> bool:
    """Check if paragraph is short (<= 240 chars OR <= 2 sentences)."""
    if len(text) <= LEAD_IN_THRESHOLD:
        return True
    sentences = SENTENCE_RE.split(text)
    return len([s for s in sentences if s.strip()]) <= 2


def _split_list_items(items: List[str]) -> List[List[str]]:
    """
    Split list items into groups based on size and item length.

    Args:
        items: List of item strings

    Returns:
        List of item groups
    """
    if len(items) <= LIST_SPLIT_THRESHOLD:
        return [items]

    # Check average item length
    avg_length = sum(len(item) for item in items) / len(items)
    if avg_length > LIST_ITEM_LENGTH_THRESHOLD:
        # Split per item
        return [[item] for item in items]

    # Group into chunks of 3-5 items (prefer 5)
    groups = []
    i = 0
    while i < len(items):
        # Try to take 5, but ensure we don't exceed hard max
        group = []
        group_size = 0
        target_count = 5

        while (len(group) < target_count and i < len(items) and
               group_size + len(items[i]) < CHUNK_HARD_MAX):
            group.append(items[i])
            group_size += len(items[i]) + 2  # +2 for "- " prefix
            i += 1

        if not group:
            # Force add at least one item
            group.append(items[i])
            i += 1

        groups.append(group)

    return groups


def _format_chunk_header(
    chapter_title: str,
    subsection_index: Optional[int] = None,
    subsection_title: Optional[str] = None,
    subhead_title: Optional[str] = None
) -> str:
    """Format context header for a chunk."""
    lines = [f"Chapter: {chapter_title}"]
    if subsection_index is not None and subsection_title:
        lines.append(
            f"Subsection: {subsection_index}. {subsection_title}"
        )
    if subhead_title:
        lines.append(f"Subhead: {subhead_title}")
    return "\n".join(lines) + "\n\n"


def _format_block_text(block: Block) -> str:
    """Format a block's text for chunk inclusion."""
    if block.type == "media":
        # Skip media blocks in chunk text (keep in metadata only)
        return ""
    return block.text


def chunk_from_structure(
    doc_structure: DocumentStructure
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Chunk a document based on its structured representation.

    Respects heading boundaries and uses semantic block rules for chunking.

    Args:
        doc_structure: DocumentStructure with chapter and subsections

    Returns:
        Tuple of (chunk_texts, chunk_metadata_list)
    """
    if doc_structure.chapter is None:
        return [], []

    chapter = doc_structure.chapter
    chunks = []
    chunk_metadata = []

    chunk_index = 0

    # Process preface blocks (blocks_before_subsections)
    if chapter.blocks_before_subsections:
        preface_blocks = chapter.blocks_before_subsections
        preface_text_parts = []
        preface_block_refs = []

        for block in preface_blocks:
            block_text = _format_block_text(block)
            if block_text:
                preface_text_parts.append(block_text)
                preface_block_refs.append(block.meta.get("block_index", 0))

        if preface_text_parts:
            header = _format_chunk_header(chapter.title)
            chunk_text = header + "\n\n".join(preface_text_parts)
            content_hash = hashlib.sha256(
                chunk_text.encode('utf-8')
            ).hexdigest()[:16]
            preface_ref_str = '-'.join(map(str, preface_block_refs))
            chunk_id_source = (
                f"{doc_structure.source_path}::preface::{preface_ref_str}"
            )
            chunk_id = hashlib.sha256(
                chunk_id_source.encode('utf-8')
            ).hexdigest()[:16]

            chunks.append(chunk_text)
            chunk_metadata.append({
                "chunk_index": chunk_index,
                "chunk_id": chunk_id,
                "content_hash": content_hash,
                "note_path": doc_structure.source_path,
                "chapter_title": chapter.title,
                "subsection_index": None,
                "subsection_title": None,
                "subhead_path": None,
                "block_refs": preface_block_refs,
                "block_types": list(set(b.type for b in preface_blocks))
            })
            chunk_index += 1

    # Process each subsection
    for subsection in chapter.subsections:
        if subsection.subheads:
            # Chunk per subhead
            for subhead in subsection.subheads:
                subhead_chunks, subhead_metadata = _chunk_blocks_in_scope(
                    blocks=subhead.blocks,
                    chapter_title=chapter.title,
                    subsection_index=subsection.order_index,
                    subsection_title=subsection.title,
                    subhead_path=subhead.heading_path,
                    source_path=doc_structure.source_path,
                    scope_id=f"subhead_{subhead.title}",
                    start_chunk_index=chunk_index
                )
                chunks.extend(subhead_chunks)
                chunk_metadata.extend(subhead_metadata)
                chunk_index += len(subhead_chunks)
        else:
            # Chunk within subsection blocks
            subsection_chunks, subsection_metadata = _chunk_blocks_in_scope(
                blocks=subsection.blocks,
                chapter_title=chapter.title,
                subsection_index=subsection.order_index,
                subsection_title=subsection.title,
                subhead_path=None,
                source_path=doc_structure.source_path,
                scope_id=f"subsection_{subsection.order_index}",
                start_chunk_index=chunk_index
            )
            chunks.extend(subsection_chunks)
            chunk_metadata.extend(subsection_metadata)
            chunk_index += len(subsection_chunks)

    return chunks, chunk_metadata


def _chunk_blocks_in_scope(
    blocks: List[Block],
    chapter_title: str,
    subsection_index: Optional[int],
    subsection_title: Optional[str],
    subhead_path: Optional[List[str]],
    source_path: str,
    scope_id: str,
    start_chunk_index: int
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Chunk blocks within a scope (subsection or subhead).

    Implements semantic chunking rules with hard boundaries and grouping.
    """
    if not blocks:
        return [], []

    chunks = []
    chunk_metadata = []
    chunk_index = start_chunk_index

    current_chunk_blocks = []
    current_chunk_text_parts = []
    current_chunk_block_refs = []
    current_chunk_size = 0
    previous_was_short_paragraph = False

    i = 0
    while i < len(blocks):
        block = blocks[i]
        block_text = _format_block_text(block)
        block_size = len(block_text)
        block_ref = block.meta.get("block_index", i)

        # Hard boundaries: callout, code, quote, media
        if block.type in ("callout", "code", "quote", "media"):
            # Check for lead-in attachment BEFORE flushing
            should_attach_lead_in = (
                previous_was_short_paragraph and
                current_chunk_text_parts and
                len(current_chunk_text_parts) == 1 and
                current_chunk_blocks and
                current_chunk_blocks[0].type == "paragraph"
            )

            if should_attach_lead_in:
                # Merge previous short paragraph into this chunk
                lead_in = current_chunk_text_parts[0]
                lead_in_ref = current_chunk_block_refs[0]
                lead_in_block = current_chunk_blocks[0]
                current_chunk_size = block_size + len(lead_in) + 2
                current_chunk_text_parts = [lead_in, block_text]
                current_chunk_block_refs = [lead_in_ref, block_ref]
                current_chunk_blocks = [lead_in_block, block]
            else:
                # Flush current chunk if exists
                if current_chunk_text_parts:
                    chunk_text, metadata = _finalize_chunk(
                        current_chunk_text_parts,
                        current_chunk_block_refs,
                        current_chunk_blocks,
                        chapter_title,
                        subsection_index,
                        subsection_title,
                        subhead_path,
                        source_path,
                        scope_id,
                        chunk_index
                    )
                    chunks.append(chunk_text)
                    chunk_metadata.append(metadata)
                    chunk_index += 1
                    current_chunk_blocks = []
                    current_chunk_text_parts = []
                    current_chunk_block_refs = []
                    current_chunk_size = 0

                # Start new chunk with this block
                if block_text:  # Skip empty media blocks
                    current_chunk_text_parts = [block_text]
                    current_chunk_block_refs = [block_ref]
                    current_chunk_blocks = [block]
                    current_chunk_size = block_size

            # Finalize this chunk (hard boundary blocks are their own chunk)
            if current_chunk_text_parts:
                chunk_text, metadata = _finalize_chunk(
                    current_chunk_text_parts,
                    current_chunk_block_refs,
                    current_chunk_blocks,
                    chapter_title,
                    subsection_index,
                    subsection_title,
                    subhead_path,
                    source_path,
                    scope_id,
                    chunk_index
                )
                chunks.append(chunk_text)
                chunk_metadata.append(metadata)
                chunk_index += 1
                current_chunk_blocks = []
                current_chunk_text_parts = []
                current_chunk_block_refs = []
                current_chunk_size = 0
                previous_was_short_paragraph = False

        # Lists
        elif block.type == "list":
            items = block.meta.get("items", [])
            if not items:
                i += 1
                continue

            item_groups = _split_list_items(items)
            for group in item_groups:
                group_text = "\n".join(f"- {item}" for item in group)
                group_size = len(group_text)

                # Check if we can add to current chunk
                if (current_chunk_size + group_size + 2 <= CHUNK_HARD_MAX and
                        current_chunk_size < CHUNK_TARGET_MAX):
                    current_chunk_text_parts.append(group_text)
                    current_chunk_block_refs.append(block_ref)
                    current_chunk_blocks.append(block)
                    current_chunk_size += group_size + 2
                else:
                    # Flush current chunk
                    if current_chunk_text_parts:
                        chunk_text, metadata = _finalize_chunk(
                            current_chunk_text_parts,
                            current_chunk_block_refs,
                            current_chunk_blocks,
                            chapter_title,
                            subsection_index,
                            subsection_title,
                            subhead_path,
                            source_path,
                            scope_id,
                            chunk_index
                        )
                        chunks.append(chunk_text)
                        chunk_metadata.append(metadata)
                        chunk_index += 1
                        current_chunk_blocks = []
                        current_chunk_text_parts = []
                        current_chunk_block_refs = []
                        current_chunk_size = 0

                    # Start new chunk with this list group
                    current_chunk_text_parts = [group_text]
                    current_chunk_block_refs = [block_ref]
                    current_chunk_blocks = [block]
                    current_chunk_size = group_size

            previous_was_short_paragraph = False

        # Paragraphs
        elif block.type == "paragraph":
            is_short = _is_short_paragraph(block_text)

            # Check if we should flush before adding
            if (current_chunk_size + block_size + 2 > CHUNK_HARD_MAX):
                # Must flush
                if current_chunk_text_parts:
                    chunk_text, metadata = _finalize_chunk(
                        current_chunk_text_parts,
                        current_chunk_block_refs,
                        current_chunk_blocks,
                        chapter_title,
                        subsection_index,
                        subsection_title,
                        subhead_path,
                        source_path,
                        scope_id,
                        chunk_index
                    )
                    chunks.append(chunk_text)
                    chunk_metadata.append(metadata)
                    chunk_index += 1
                    current_chunk_blocks = []
                    current_chunk_text_parts = []
                    current_chunk_block_refs = []
                    current_chunk_size = 0

            # Add paragraph to current chunk
            if (current_chunk_size == 0 or
                    current_chunk_size + block_size + 2 <= CHUNK_HARD_MAX):
                current_chunk_text_parts.append(block_text)
                current_chunk_block_refs.append(block_ref)
                current_chunk_blocks.append(block)
                current_chunk_size += block_size + 2
                previous_was_short_paragraph = is_short
            else:
                # Start new chunk
                current_chunk_text_parts = [block_text]
                current_chunk_block_refs = [block_ref]
                current_chunk_blocks = [block]
                current_chunk_size = block_size
                previous_was_short_paragraph = is_short

        i += 1

    # Flush remaining chunk
    if current_chunk_text_parts:
        chunk_text, metadata = _finalize_chunk(
            current_chunk_text_parts,
            current_chunk_block_refs,
            current_chunk_blocks,
            chapter_title,
            subsection_index,
            subsection_title,
            subhead_path,
            source_path,
            scope_id,
            chunk_index
        )
        chunks.append(chunk_text)
        chunk_metadata.append(metadata)

    return chunks, chunk_metadata


def _finalize_chunk(
    text_parts: List[str],
    block_refs: List[int],
    blocks: List[Block],
    chapter_title: str,
    subsection_index: Optional[int],
    subsection_title: Optional[str],
    subhead_path: Optional[List[str]],
    source_path: str,
    scope_id: str,
    chunk_index: int
) -> Tuple[str, Dict[str, Any]]:
    """Finalize a chunk by formatting text and generating metadata."""
    # Filter out empty parts
    text_parts = [p for p in text_parts if p.strip()]

    # Format header
    if subhead_path and len(subhead_path) > 2:
        subhead_title = subhead_path[-1]
    else:
        subhead_title = None
    header = _format_chunk_header(
        chapter_title,
        subsection_index,
        subsection_title,
        subhead_title
    )

    # Join block texts
    body = "\n\n".join(text_parts)
    chunk_text = header + body

    # Generate hashes
    content_hash = hashlib.sha256(
        chunk_text.encode('utf-8')
    ).hexdigest()[:16]

    # Generate chunk_id from scope path + block refs
    scope_path = f"{chapter_title}"
    if subsection_index is not None:
        scope_path += f"::{subsection_index}"
    if subhead_path:
        # Skip chapter/subsection from path
        subhead_part = '-'.join(subhead_path[2:])
        scope_path += f"::{subhead_part}"
    block_ref_str = "-".join(map(str, block_refs))
    chunk_id_source = f"{source_path}::{scope_path}::{block_ref_str}"
    chunk_id = hashlib.sha256(chunk_id_source.encode('utf-8')).hexdigest()[:16]

    # Build metadata
    metadata = {
        "chunk_index": chunk_index,
        "chunk_id": chunk_id,
        "content_hash": content_hash,
        "note_path": source_path,
        "chapter_title": chapter_title,
        "subsection_index": subsection_index,
        "subsection_title": subsection_title,
        "subhead_path": subhead_path,
        "block_refs": block_refs,
        "block_types": list(set(b.type for b in blocks))
    }

    return chunk_text, metadata
