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
CHUNK_HARD_MAX_CHARS = 3000  # Hard maximum characters per chunk
SHORT_PARAGRAPH_MAX_CHARS = 240  # Max chars for short paragraph
MIN_CHUNK_CHARS = 140  # Minimum chunk size before merging
LIST_SPLIT_THRESHOLD = 5  # Split lists with >5 items
LIST_ITEM_LENGTH_THRESHOLD = 120  # Split per item if avg > 120 chars
LIST_MAX_ITEMS_PER_CHUNK = 8  # Max items before splitting
LIST_GROUP_SIZE = 6  # Preferred group size when splitting

# Subhead-only mode config
MERGE_TINY_INTRO_WITH_SINGLE_SUBHEAD = True
TINY_INTRO_MAX_CHARS = 160

# Optional block detection patterns (strengthened)
OPTIONAL_PATTERNS = re.compile(
    r'^(if you want|next we can|just tell me|you can also|next steps|'
    r'continue|up next|if you\'d like)',
    re.IGNORECASE
)

# Example detection pattern
EXAMPLE_PATTERN = re.compile(r'^example:', re.IGNORECASE)


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


def _normalize_text(text: str) -> str:
    """
    Normalize chunk text for hashing.

    Strips trailing spaces, collapses 3+ newlines to 2, and trims.
    """
    # Collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip trailing spaces from each line
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)
    # Trim overall
    return text.strip()


def _is_short_paragraph(text: str) -> bool:
    """Check if paragraph is short (<= SHORT_PARAGRAPH_MAX_CHARS)."""
    return len(text) <= SHORT_PARAGRAPH_MAX_CHARS


def _is_lead_in_paragraph(
    text: str,
    next_block: Optional[Block] = None
) -> bool:
    """
    Check if paragraph qualifies as a lead-in (merge-forward).

    A paragraph is a lead-in if:
    - Ends with `:`
    - Length <= SHORT_PARAGRAPH_MAX_CHARS
    - Single line (no \n) AND next block is list/quote/callout/code
    """
    text_stripped = text.strip()

    # Ends with colon
    if text_stripped.endswith(':'):
        return True

    # Short paragraph
    if len(text_stripped) <= SHORT_PARAGRAPH_MAX_CHARS:
        return True

    # Single line AND next block is anchor type
    if '\n' not in text_stripped and next_block:
        if next_block.type in ("list", "quote", "callout", "code"):
            return True

    return False


def _is_optional_block(block: Block) -> bool:
    """Check if block contains optional/navigation content."""
    if block.type != "paragraph":
        return False
    text = block.text.strip()
    if not text:
        return False
    first_line = text.split('\n')[0]
    return bool(OPTIONAL_PATTERNS.match(first_line))


def _mark_optional_blocks(blocks: List[Block]) -> None:
    """Mark optional blocks in-place."""
    for block in blocks:
        if _is_optional_block(block):
            if not hasattr(block, 'meta') or block.meta is None:
                block.meta = {}
            block.meta["is_optional"] = True


def _compute_chunk_role(
    blocks: List[Block],
    chunk_scope: str,
    has_optional: bool
) -> str:
    """
    Compute deterministic chunk role from block mix + position.

    Returns: intro, concept, enumeration, rule, example, navigation
    """
    if has_optional:
        return "navigation"

    # Count block types
    block_types = [b.type for b in blocks]
    type_counts = {}
    for bt in block_types:
        type_counts[bt] = type_counts.get(bt, 0) + 1

    # Check for code blocks
    if "code" in type_counts:
        return "example"

    # Check for example label
    for block in blocks:
        if block.type == "paragraph":
            text = block.text.strip()
            if EXAMPLE_PATTERN.match(text):
                return "example"

    # Check if quote/callout dominate
    quote_callout_count = (
        type_counts.get("quote", 0) + type_counts.get("callout", 0)
    )
    total_blocks = len(blocks)
    if total_blocks > 0 and quote_callout_count / total_blocks >= 0.5:
        return "rule"

    # Check if list dominates
    list_count = type_counts.get("list", 0)
    if total_blocks > 0 and list_count / total_blocks >= 0.5:
        return "enumeration"

    # Check if intro scope
    if chunk_scope == "subsection":
        return "intro"

    # Default
    return "concept"


def _format_breadcrumb(
    chapter_title: str,
    subsection_index: Optional[int],
    subsection_title: Optional[str],
    subhead_title: Optional[str]
) -> str:
    """Format breadcrumb string."""
    parts = [chapter_title]
    if subsection_index is not None and subsection_title:
        parts.append(f"{subsection_index}. {subsection_title}")
    if subhead_title:
        parts.append(subhead_title)
    return " > ".join(parts)


def _format_embed_header(
    chapter_title: str,
    subsection_index: Optional[int],
    subsection_title: Optional[str],
    subhead_title: Optional[str]
) -> str:
    """Format minimal context header for embedding."""
    lines = []
    if chapter_title:
        lines.append(f"Chapter: {chapter_title}")
    if subsection_index is not None and subsection_title:
        lines.append(f"{subsection_index}. {subsection_title}")
    if subhead_title:
        lines.append(subhead_title)
    if lines:
        return "\n".join(lines) + "\n\n"
    return ""


def _format_study_header(
    chapter_title: str,
    subsection_index: Optional[int],
    subsection_title: Optional[str],
    subhead_title: Optional[str]
) -> str:
    """Format full context header for study."""
    lines = [f"Chapter: {chapter_title}"]
    if subsection_index is not None and subsection_title:
        lines.append(
            f"Subsection: {subsection_index}. {subsection_title}"
        )
    if subhead_title:
        lines.append(f"Subhead: {subhead_title}")
    return "\n".join(lines) + "\n\n"


def _compute_scope_hash(
    source_path: str,
    chunk_scope: str,
    subsection_index: Optional[int],
    subhead_path: Optional[List[str]]
) -> str:
    """Compute scope hash for regen friendliness."""
    parts = [source_path, chunk_scope]
    if subsection_index is not None:
        parts.append(str(subsection_index))
    if subhead_path:
        parts.append('-'.join(subhead_path))
    identity = '::'.join(parts)
    return hashlib.sha256(identity.encode('utf-8')).hexdigest()


def _split_list_items(items: List[str]) -> List[List[str]]:
    """
    Split list items into groups based on size and item length.

    Improved: caps at LIST_MAX_ITEMS_PER_CHUNK, prefers LIST_GROUP_SIZE.

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

    # If items > LIST_MAX_ITEMS_PER_CHUNK, split into groups
    if len(items) > LIST_MAX_ITEMS_PER_CHUNK:
        # Group into chunks of 4-6 items (prefer LIST_GROUP_SIZE)
        groups = []
        i = 0
        while i < len(items):
            group = []
            group_size = 0
            target_count = LIST_GROUP_SIZE

            while (len(group) < target_count and i < len(items) and
                   group_size + len(items[i]) < CHUNK_HARD_MAX_CHARS):
                group.append(items[i])
                group_size += len(items[i]) + 2  # +2 for "- " prefix
                i += 1

            if not group:
                # Force add at least one item
                group.append(items[i])
                i += 1

            groups.append(group)
        return groups

    # Group into chunks of 3-5 items (prefer 5) for smaller lists
    groups = []
    i = 0
    while i < len(items):
        group = []
        group_size = 0
        target_count = 5

        while (len(group) < target_count and i < len(items) and
               group_size + len(items[i]) < CHUNK_HARD_MAX_CHARS):
            group.append(items[i])
            group_size += len(items[i]) + 2
            i += 1

        if not group:
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


def _get_chunk_scope(
    subsection_index: Optional[int],
    subhead_path: Optional[List[str]]
) -> str:
    """Determine chunk scope: preface, subsection, or subhead."""
    if subsection_index is None:
        return "preface"
    if subhead_path:
        return "subhead"
    return "subsection"


def _get_scope_key(
    chunk_scope: str,
    subsection_index: Optional[int],
    subhead_path: Optional[List[str]]
) -> str:
    """Generate a scope key for merge matching."""
    if chunk_scope == "preface":
        return "preface"
    if chunk_scope == "subsection":
        return f"subsection_{subsection_index}"
    if chunk_scope == "subhead":
        subhead_key = '-'.join(subhead_path) if subhead_path else ""
        return f"subhead_{subsection_index}_{subhead_key}"
    return "unknown"


def _scope_keys_match(
    scope1: str,
    key1: str,
    scope2: str,
    key2: str
) -> bool:
    """Check if two chunks can be merged (same scope and key)."""
    return scope1 == scope2 and key1 == key2


def _compute_chunk_id(
    source_path: str,
    chunk_scope: str,
    subsection_index: Optional[int],
    subhead_path: Optional[List[str]],
    block_refs: List[int]
) -> str:
    """
    Compute stable chunk_id from structural identity.

    Uses sha256 with full 64 hex chars.
    """
    # Deduplicate and sort block refs
    block_refs_dedup = sorted(list(dict.fromkeys(block_refs)))
    block_ref_str = '-'.join(map(str, block_refs_dedup))

    # Build identity string
    parts = [source_path, chunk_scope]
    if subsection_index is not None:
        parts.append(str(subsection_index))
    if subhead_path:
        parts.append('-'.join(subhead_path))
    parts.append(block_ref_str)

    identity = '::'.join(parts)
    return hashlib.sha256(identity.encode('utf-8')).hexdigest()


def _finalize_chunk(
    text_parts: List[str],
    block_refs: List[int],
    blocks: List[Block],
    chapter_title: str,
    subsection_index: Optional[int],
    subsection_title: Optional[str],
    subhead_path: Optional[List[str]],
    source_path: str,
    chunk_scope: str,
    chunk_index: int,
    chunking_decisions: List[str],
    list_item_count: Optional[int] = None,
    list_group_index: Optional[int] = None
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Finalize a chunk by formatting text and generating metadata.

    Returns: (embed_text, study_text, metadata)
    """
    # Filter out empty parts
    text_parts = [p for p in text_parts if p.strip()]

    # Format headers
    if subhead_path and len(subhead_path) > 2:
        subhead_title = subhead_path[-1]
    else:
        subhead_title = None

    embed_header = _format_embed_header(
        chapter_title,
        subsection_index,
        subsection_title,
        subhead_title
    )
    study_header = _format_study_header(
        chapter_title,
        subsection_index,
        subsection_title,
        subhead_title
    )

    # Join block texts
    body = "\n\n".join(text_parts)
    embed_text = embed_header + body
    study_text = study_header + body

    # Normalize for hashing (use embed_text)
    normalized_text = _normalize_text(embed_text)

    # Generate hashes
    content_hash = hashlib.sha256(
        normalized_text.encode('utf-8')
    ).hexdigest()

    # Deduplicate block refs
    block_refs_dedup = list(dict.fromkeys(block_refs))

    # Compute chunk_id from structural identity
    chunk_id = _compute_chunk_id(
        source_path,
        chunk_scope,
        subsection_index,
        subhead_path,
        block_refs_dedup
    )

    # Compute scope_hash
    scope_hash = _compute_scope_hash(
        source_path,
        chunk_scope,
        subsection_index,
        subhead_path
    )

    # Check for optional blocks
    has_optional_blocks = any(_is_optional_block(b) for b in blocks)

    # Compute chunk role
    chunk_role = _compute_chunk_role(blocks, chunk_scope, has_optional_blocks)

    # Format breadcrumb
    breadcrumb = _format_breadcrumb(
        chapter_title,
        subsection_index,
        subsection_title,
        subhead_title
    )

    # Build hierarchy dict
    hierarchy = {
        "chapter": chapter_title,
        "subsection_index": subsection_index,
        "subsection_title": subsection_title,
        "subhead_title": subhead_title
    }

    # Build metadata
    metadata = {
        "chunk_index": chunk_index,
        "chunk_id": chunk_id,
        "content_hash": content_hash,
        "scope_hash": scope_hash,
        "note_path": source_path,
        "chapter_title": chapter_title,
        "subsection_index": subsection_index,
        "subsection_title": subsection_title,
        "subhead_path": subhead_path,
        "chunk_scope": chunk_scope,
        "scope_key": _get_scope_key(
            chunk_scope,
            subsection_index,
            subhead_path
        ),
        "block_refs": block_refs_dedup,
        "block_types": list(set(b.type for b in blocks)),
        "chunking_decisions": chunking_decisions,
        "has_optional_blocks": has_optional_blocks,
        "chunk_role": chunk_role,
        "breadcrumb": breadcrumb,
        "hierarchy": hierarchy,
        "embed_text": embed_text,
        "study_text": study_text,
        "list_item_count": list_item_count,
        "list_group_index": list_group_index
    }

    return embed_text, study_text, metadata


def _apply_min_quality_merges(
    chunks: List[str],
    chunk_metadata: List[Dict[str, Any]]
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Post-pass: merge tiny standalone paragraph chunks.

    Only merges if:
    - Chunk is single paragraph block
    - Below SHORT_PARAGRAPH_MAX_CHARS
    - chunk_scope != preface
    - Same scope key as neighbor
    """
    if len(chunks) <= 1:
        return chunks, chunk_metadata

    merged_chunks = []
    merged_metadata = []
    i = 0

    while i < len(chunks):
        current_chunk = chunks[i]
        current_meta = chunk_metadata[i]

        # Check if this is a dangling fragment (tiny chunk)
        # Use embed_text length from metadata if available, else chunk length
        chunk_content_length = len(
            current_meta.get("embed_text", current_chunk)
        )
        # Subtract header length (approximate)
        body_length = chunk_content_length - 100  # Rough estimate

        is_dangling_fragment = (
            body_length < MIN_CHUNK_CHARS and
            current_meta.get("chunk_scope") != "preface"
        )

        if not is_dangling_fragment:
            merged_chunks.append(current_chunk)
            merged_metadata.append(current_meta)
            i += 1
            continue

        # Try to merge backward
        if merged_chunks:
            prev_meta = merged_metadata[-1]
            if _scope_keys_match(
                current_meta.get("chunk_scope", ""),
                current_meta.get("scope_key", ""),
                prev_meta.get("chunk_scope", ""),
                prev_meta.get("scope_key", "")
            ):
                # Merge backward
                prev_chunk = merged_chunks.pop()
                prev_meta = merged_metadata.pop()

                # Combine chunks
                combined_text = prev_chunk.rstrip() + "\n\n" + current_chunk
                combined_blocks = (
                    prev_meta.get("block_refs", []) +
                    current_meta.get("block_refs", [])
                )
                combined_decisions = (
                    prev_meta.get("chunking_decisions", []) +
                    ["min_quality_merge_backward"] +
                    current_meta.get("chunking_decisions", [])
                )

                # Re-finalize (simplified - use existing metadata structure)
                # Extract body from combined text
                body_parts = combined_text.split("\n\n", 2)
                if len(body_parts) > 2:
                    body = body_parts[2]  # Skip headers
                else:
                    body = combined_text

                embed_text, study_text, new_meta = _finalize_chunk(
                    [body],
                    combined_blocks,
                    [],  # Blocks not available here
                    prev_meta.get("chapter_title", ""),
                    prev_meta.get("subsection_index"),
                    prev_meta.get("subsection_title"),
                    prev_meta.get("subhead_path"),
                    prev_meta.get("note_path", ""),
                    prev_meta.get("chunk_scope", ""),
                    len(merged_chunks),
                    combined_decisions
                )
                merged_chunks.append(embed_text)
                merged_metadata.append(new_meta)
                i += 1
                continue

        # Try to merge forward
        if i + 1 < len(chunks):
            next_meta = chunk_metadata[i + 1]
            next_chunk = chunks[i + 1]

            # Check if next chunk starts with hard boundary
            next_block_types = next_meta.get("block_types", [])
            starts_with_hard_boundary = any(
                bt in ("quote", "callout", "code") for bt in next_block_types
            )
            starts_with_list = "list" in next_block_types

            can_merge = (
                (starts_with_hard_boundary or starts_with_list) and
                _scope_keys_match(
                    current_meta.get("chunk_scope", ""),
                    current_meta.get("scope_key", ""),
                    next_meta.get("chunk_scope", ""),
                    next_meta.get("scope_key", "")
                )
            )
            if can_merge:
                # Merge forward
                combined_text = current_chunk.rstrip() + "\n\n" + next_chunk
                combined_blocks = (
                    current_meta.get("block_refs", []) +
                    next_meta.get("block_refs", [])
                )
                combined_decisions = (
                    current_meta.get("chunking_decisions", []) +
                    ["min_quality_merge_forward"] +
                    next_meta.get("chunking_decisions", [])
                )

                # Re-finalize
                body_parts = combined_text.split("\n\n", 2)
                if len(body_parts) > 2:
                    body = body_parts[2]
                else:
                    body = combined_text

                embed_text, study_text, new_meta = _finalize_chunk(
                    [body],
                    combined_blocks,
                    [],
                    current_meta.get("chapter_title", ""),
                    current_meta.get("subsection_index"),
                    current_meta.get("subsection_title"),
                    current_meta.get("subhead_path"),
                    current_meta.get("note_path", ""),
                    current_meta.get("chunk_scope", ""),
                    len(merged_chunks),
                    combined_decisions
                )
                merged_chunks.append(embed_text)
                merged_metadata.append(new_meta)
                i += 2  # Skip next chunk
                continue

        # Can't merge, keep as-is
        merged_chunks.append(current_chunk)
        merged_metadata.append(current_meta)
        i += 1

    return merged_chunks, merged_metadata


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
        _mark_optional_blocks(chapter.blocks_before_subsections)
        preface_chunks, preface_metadata = _chunk_blocks_in_scope(
            blocks=chapter.blocks_before_subsections,
            chapter_title=chapter.title,
            subsection_index=None,
            subsection_title=None,
            subhead_path=None,
            source_path=doc_structure.source_path,
            chunk_scope="preface",
            start_chunk_index=chunk_index
        )
        chunks.extend(preface_chunks)
        chunk_metadata.extend(preface_metadata)
        chunk_index += len(preface_chunks)

    # Process each subsection
    for subsection in chapter.subsections:
        if subsection.subheads:
            # Mark optional blocks in subhead blocks
            for subhead in subsection.subheads:
                _mark_optional_blocks(subhead.blocks)

            # Process subsection root blocks (intro, before first subhead)
            # Only if there are blocks before subheads
            if subsection.blocks:
                # Mark optional blocks
                _mark_optional_blocks(subsection.blocks)

                # Chunk intro blocks
                intro_chunks, intro_metadata = _chunk_blocks_in_scope(
                    blocks=subsection.blocks,
                    chapter_title=chapter.title,
                    subsection_index=subsection.order_index,
                    subsection_title=subsection.title,
                    subhead_path=None,
                    source_path=doc_structure.source_path,
                    chunk_scope="subsection",
                    start_chunk_index=chunk_index
                )

                # Check if we should merge tiny intro with single subhead
                if (MERGE_TINY_INTRO_WITH_SINGLE_SUBHEAD and
                        len(subsection.subheads) == 1 and
                        len(intro_chunks) == 1 and
                        len(intro_chunks[0]) <= TINY_INTRO_MAX_CHARS):
                    # Merge intro into first subhead chunk
                    subhead = subsection.subheads[0]
                    subhead_chunks, subhead_metadata = _chunk_blocks_in_scope(
                        blocks=subhead.blocks,
                        chapter_title=chapter.title,
                        subsection_index=subsection.order_index,
                        subsection_title=subsection.title,
                        subhead_path=subhead.heading_path,
                        source_path=doc_structure.source_path,
                        chunk_scope="subhead",
                        start_chunk_index=chunk_index
                    )
                    if subhead_chunks:
                        # Merge intro into first subhead chunk
                        merged_text = (
                            intro_chunks[0].rstrip() + "\n\n" +
                            subhead_chunks[0]
                        )
                        merged_meta = subhead_metadata[0].copy()
                        merged_meta["chunking_decisions"] = (
                            merged_meta.get("chunking_decisions", []) +
                            ["intro_merge_into_single_subhead"]
                        )
                        merged_meta["block_refs"] = (
                            intro_metadata[0].get("block_refs", []) +
                            merged_meta.get("block_refs", [])
                        )
                        # Re-finalize with merged content
                        # (simplified - would need full block list)
                        chunks.append(merged_text)
                        chunk_metadata.append(merged_meta)
                        chunks.extend(subhead_chunks[1:])
                        chunk_metadata.extend(subhead_metadata[1:])
                        chunk_index += len(subhead_chunks)
                    else:
                        chunks.extend(intro_chunks)
                        chunk_metadata.extend(intro_metadata)
                        chunk_index += len(intro_chunks)
                else:
                    # Keep intro separate
                    chunks.extend(intro_chunks)
                    chunk_metadata.extend(intro_metadata)
                    chunk_index += len(intro_chunks)

            # Chunk per subhead (each subhead is independent)
            for subhead in subsection.subheads:
                subhead_chunks, subhead_metadata = _chunk_blocks_in_scope(
                    blocks=subhead.blocks,
                    chapter_title=chapter.title,
                    subsection_index=subsection.order_index,
                    subsection_title=subsection.title,
                    subhead_path=subhead.heading_path,
                    source_path=doc_structure.source_path,
                    chunk_scope="subhead",
                    start_chunk_index=chunk_index
                )
                chunks.extend(subhead_chunks)
                chunk_metadata.extend(subhead_metadata)
                chunk_index += len(subhead_chunks)
        else:
            # No subheads, chunk subsection blocks directly
            _mark_optional_blocks(subsection.blocks)
            subsection_chunks, subsection_metadata = _chunk_blocks_in_scope(
                blocks=subsection.blocks,
                chapter_title=chapter.title,
                subsection_index=subsection.order_index,
                subsection_title=subsection.title,
                subhead_path=None,
                source_path=doc_structure.source_path,
                chunk_scope="subsection",
                start_chunk_index=chunk_index
            )
            chunks.extend(subsection_chunks)
            chunk_metadata.extend(subsection_metadata)
            chunk_index += len(subsection_chunks)

    # Apply post-pass minimum quality merges (only if reasonable number)
    # Skip for very large documents to avoid performance issues
    if len(chunks) < 500:  # Only merge for documents with < 500 chunks
        chunks, chunk_metadata = _apply_min_quality_merges(
            chunks,
            chunk_metadata
        )
    else:
        logger = logging.getLogger("obsynapse.chunking")
        logger.info(
            f"Skipping min quality merges for large document "
            f"({len(chunks)} chunks)"
        )

    # Update chunk indices after merging
    for i, meta in enumerate(chunk_metadata):
        meta["chunk_index"] = i

    return chunks, chunk_metadata


def _chunk_blocks_in_scope(
    blocks: List[Block],
    chapter_title: str,
    subsection_index: Optional[int],
    subsection_title: Optional[str],
    subhead_path: Optional[List[str]],
    source_path: str,
    chunk_scope: str,
    start_chunk_index: int
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Chunk blocks within a scope (preface, subsection, or subhead).

    Implements semantic chunking rules with hard boundaries, bridge merges,
    and lead-in detection.
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
    current_chunk_decisions = []

    i = 0
    while i < len(blocks):
        block = blocks[i]
        block_text = _format_block_text(block)
        block_size = len(block_text)
        block_ref = block.meta.get("block_index", i)

        # Look ahead for next block (for lead-in detection)
        next_block = blocks[i + 1] if i + 1 < len(blocks) else None

        # Hard boundaries: callout, code, quote, media
        if block.type in ("callout", "code", "quote", "media"):
            # Check for lead-in attachment BEFORE flushing
            should_attach_lead_in = False
            if (current_chunk_text_parts and
                    len(current_chunk_text_parts) == 1 and
                    current_chunk_blocks and
                    current_chunk_blocks[0].type == "paragraph"):
                lead_in_text = current_chunk_text_parts[0]
                if _is_lead_in_paragraph(lead_in_text, block):
                    should_attach_lead_in = True

            if should_attach_lead_in:
                # Merge lead-in paragraph into this chunk
                lead_in = current_chunk_text_parts[0]
                lead_in_ref = current_chunk_block_refs[0]
                lead_in_block = current_chunk_blocks[0]
                current_chunk_size = block_size + len(lead_in) + 2
                current_chunk_text_parts = [lead_in, block_text]
                current_chunk_block_refs = [lead_in_ref, block_ref]
                current_chunk_blocks = [lead_in_block, block]
                current_chunk_decisions.append("lead_in_merge_forward")
            else:
                # Flush current chunk if exists
                if current_chunk_text_parts:
                    embed_text, study_text, metadata = _finalize_chunk(
                        current_chunk_text_parts,
                        current_chunk_block_refs,
                        current_chunk_blocks,
                        chapter_title,
                        subsection_index,
                        subsection_title,
                        subhead_path,
                        source_path,
                        chunk_scope,
                        chunk_index,
                        current_chunk_decisions
                    )
                    chunks.append(embed_text)
                    chunk_metadata.append(metadata)
                    chunk_index += 1
                    current_chunk_blocks = []
                    current_chunk_text_parts = []
                    current_chunk_block_refs = []
                    current_chunk_size = 0
                    current_chunk_decisions = []

                # Start new chunk with this block
                if block_text:  # Skip empty media blocks
                    current_chunk_text_parts = [block_text]
                    current_chunk_block_refs = [block_ref]
                    current_chunk_blocks = [block]
                    current_chunk_size = block_size
                    current_chunk_decisions = ["hard_boundary_flush"]

            # Check for bridge merge: quote/callout + short paragraph
            is_quote_or_callout = block.type in ("quote", "callout")
            if (is_quote_or_callout and
                    next_block and
                    next_block.type == "paragraph"):
                next_text = _format_block_text(next_block)
                if _is_short_paragraph(next_text):
                    # Bridge merge forward
                    i += 1  # Consume next block
                    current_chunk_text_parts.append(next_text)
                    current_chunk_block_refs.append(
                        next_block.meta.get("block_index", i)
                    )
                    current_chunk_blocks.append(next_block)
                    current_chunk_size += len(next_text) + 2
                    current_chunk_decisions.append(
                        f"{block.type}_bridge_merge_forward"
                    )

            # Finalize this chunk
            if current_chunk_text_parts:
                embed_text, study_text, metadata = _finalize_chunk(
                    current_chunk_text_parts,
                    current_chunk_block_refs,
                    current_chunk_blocks,
                    chapter_title,
                    subsection_index,
                    subsection_title,
                    subhead_path,
                    source_path,
                    chunk_scope,
                    chunk_index,
                    current_chunk_decisions
                )
                chunks.append(embed_text)
                chunk_metadata.append(metadata)
                chunk_index += 1
                current_chunk_blocks = []
                current_chunk_text_parts = []
                current_chunk_block_refs = []
                current_chunk_size = 0
                current_chunk_decisions = []

        # Lists
        elif block.type == "list":
            items = block.meta.get("items", [])
            if not items:
                i += 1
                continue

            # Check for lead-in paragraph
            should_attach_lead_in = False
            if (current_chunk_text_parts and
                    len(current_chunk_text_parts) == 1 and
                    current_chunk_blocks and
                    current_chunk_blocks[0].type == "paragraph"):
                lead_in_text = current_chunk_text_parts[0]
                if _is_lead_in_paragraph(lead_in_text, block):
                    should_attach_lead_in = True

            if should_attach_lead_in:
                lead_in = current_chunk_text_parts[0]
                lead_in_ref = current_chunk_block_refs[0]
                lead_in_block = current_chunk_blocks[0]
                current_chunk_text_parts = [lead_in]
                current_chunk_block_refs = [lead_in_ref]
                current_chunk_blocks = [lead_in_block]
                current_chunk_size = len(lead_in)
                current_chunk_decisions.append("lead_in_merge_forward")

            item_groups = _split_list_items(items)
            for group_idx, group in enumerate(item_groups):
                group_text = "\n".join(f"- {item}" for item in group)
                group_size = len(group_text)

                # Check if we can add to current chunk
                max_size = CHUNK_HARD_MAX_CHARS
                if (current_chunk_size + group_size + 2 <= max_size):
                    current_chunk_text_parts.append(group_text)
                    current_chunk_block_refs.append(block_ref)
                    current_chunk_blocks.append(block)
                    current_chunk_size += group_size + 2
                else:
                    # Flush current chunk at safe boundary
                    if current_chunk_text_parts:
                        embed_text, study_text, metadata = _finalize_chunk(
                            current_chunk_text_parts,
                            current_chunk_block_refs,
                            current_chunk_blocks,
                            chapter_title,
                            subsection_index,
                            subsection_title,
                            subhead_path,
                            source_path,
                            chunk_scope,
                            chunk_index,
                            current_chunk_decisions + ["size_flush"],
                            list_item_count=len(items),
                            list_group_index=group_idx
                        )
                        chunks.append(embed_text)
                        chunk_metadata.append(metadata)
                        chunk_index += 1
                        current_chunk_blocks = []
                        current_chunk_text_parts = []
                        current_chunk_block_refs = []
                        current_chunk_size = 0
                        current_chunk_decisions = []

                    # Start new chunk with this list group
                    current_chunk_text_parts = [group_text]
                    current_chunk_block_refs = [block_ref]
                    current_chunk_blocks = [block]
                    current_chunk_size = group_size

        # Paragraphs
        elif block.type == "paragraph":
            # Check if we should flush before adding
            if (current_chunk_size + block_size + 2 > CHUNK_HARD_MAX_CHARS):
                # Must flush at safe boundary
                if current_chunk_text_parts:
                    embed_text, study_text, metadata = _finalize_chunk(
                        current_chunk_text_parts,
                        current_chunk_block_refs,
                        current_chunk_blocks,
                        chapter_title,
                        subsection_index,
                        subsection_title,
                        subhead_path,
                        source_path,
                        chunk_scope,
                        chunk_index,
                        current_chunk_decisions + ["size_flush"]
                    )
                    chunks.append(embed_text)
                    chunk_metadata.append(metadata)
                    chunk_index += 1
                    current_chunk_blocks = []
                    current_chunk_text_parts = []
                    current_chunk_block_refs = []
                    current_chunk_size = 0
                    current_chunk_decisions = []

            # Add paragraph to current chunk
            max_size = CHUNK_HARD_MAX_CHARS
            if (current_chunk_size == 0 or
                    current_chunk_size + block_size + 2 <= max_size):
                current_chunk_text_parts.append(block_text)
                current_chunk_block_refs.append(block_ref)
                current_chunk_blocks.append(block)
                current_chunk_size += block_size + 2
            else:
                # Start new chunk
                current_chunk_text_parts = [block_text]
                current_chunk_block_refs = [block_ref]
                current_chunk_blocks = [block]
                current_chunk_size = block_size

        i += 1

    # Flush remaining chunk
    if current_chunk_text_parts:
        embed_text, study_text, metadata = _finalize_chunk(
            current_chunk_text_parts,
            current_chunk_block_refs,
            current_chunk_blocks,
            chapter_title,
            subsection_index,
            subsection_title,
            subhead_path,
            source_path,
            chunk_scope,
            chunk_index,
            current_chunk_decisions
        )
        chunks.append(embed_text)
        chunk_metadata.append(metadata)

    return chunks, chunk_metadata
