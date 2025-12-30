"""
Structure extractor for Obsidian markdown files.

This module parses Obsidian markdown into a structured intermediate
representation (IR) with chapters, subsections, and blocks for use in
flashcard generation and other downstream tasks.
"""

import re
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any, Tuple
from pydantic import BaseModel
from markdown_it import MarkdownIt
import frontmatter

BlockType = Literal["paragraph", "list", "code", "callout", "quote", "media"]

CALLOUT_RE = re.compile(r"^\s*\[!(?P<kind>[A-Za-z]+)\]\s*(?P<title>.*)$")
NUMBERED_SUBSECTION_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$")
EMBED_RE = re.compile(r"!\[\[([^\]]+)\]\]")


class Block(BaseModel):
    """A content block within a section."""
    type: BlockType
    text: str
    meta: Dict[str, Any] = {}


class Subhead(BaseModel):
    """A subheading (H3+) within a subsection."""
    title: str
    level: int
    heading_path: List[str]
    blocks: List[Block] = []


class Subsection(BaseModel):
    """A numbered subsection (H2) within a chapter."""
    order_index: int
    title: str
    subheads: List[Subhead] = []
    blocks: List[Block] = []


class Chapter(BaseModel):
    """A chapter (first H1) containing subsections."""
    title: str
    blocks_before_subsections: List[Block] = []
    subsections: List[Subsection] = []


class DocumentStructure(BaseModel):
    """Complete document structure with frontmatter and chapter."""
    frontmatter: Dict[str, Any] = {}
    chapter: Optional[Chapter] = None
    source_path: str = ""


def is_numbered_subsection(title: str) -> Optional[Tuple[int, str]]:
    """
    Check if a heading title matches the numbered subsection pattern.

    Args:
        title: Heading title to check

    Returns:
        Tuple of (order_index, clean_title) if match, None otherwise

    Example:
        >>> is_numbered_subsection("1. Introduction")
        (1, "Introduction")
        >>> is_numbered_subsection("10. Advanced Topics")
        (10, "Advanced Topics")
        >>> is_numbered_subsection("Introduction")
        None
    """
    match = NUMBERED_SUBSECTION_RE.match(title.strip())
    if match:
        return (int(match.group(1)), match.group(2).strip())
    return None


def extract_embeds_from_text(text: str) -> Tuple[str, List[str]]:
    """
    Extract Obsidian embeds from text and return cleaned text with targets.

    Args:
        text: Text that may contain ![[...]] embeds

    Returns:
        Tuple of (cleaned_text, list_of_embed_targets)
    """
    embeds = []
    cleaned = text

    for match in EMBED_RE.finditer(text):
        embed_target = match.group(1)
        embeds.append(embed_target)
        # Remove the embed from text
        cleaned = cleaned.replace(match.group(0), "").strip()

    return cleaned, embeds


def compute_line_offsets(text: str) -> List[int]:
    """
    Compute line start offsets for a text string.

    Args:
        text: Text to compute offsets for

    Returns:
        List of character offsets where each line starts
    """
    offsets = [0]
    for i, char in enumerate(text):
        if char == '\n':
            offsets.append(i + 1)
    return offsets


def get_block_position(
    token_start: int,
    token_end: int,
    tokens: List,
    line_offsets: List[int]
) -> Dict[str, int]:
    """
    Compute block position (line range) from token positions.
    Optimized to avoid expensive operations.

    Args:
        token_start: Starting token index
        token_end: Ending token index
        tokens: List of all tokens
        line_offsets: Precomputed line start offsets (unused but kept for API)

    Returns:
        Dict with block_index (simplified for performance)
    """
    # Simplified: just use block index for performance
    # Line number calculation is expensive and not critical
    return {"block_index": token_start}


def parse_chapter_structure(
    md_text: str, source_path: str
) -> Optional[Chapter]:
    """
    Parse Obsidian markdown into a chapter structure with subsections.

    Args:
        md_text: Raw markdown text content (without frontmatter)
        source_path: Path to the source file for metadata

    Returns:
        Chapter object if a chapter (H1) is found, None otherwise
    """
    md = MarkdownIt("commonmark")
    tokens = md.parse(md_text)
    # Line offsets not needed with simplified position tracking
    line_offsets = []

    # Find first H1 heading (chapter)
    chapter_title = None
    chapter_start_idx = None

    for i, token in enumerate(tokens):
        if token.type == "heading_open" and token.tag == "h1":
            # Get the heading text from next inline token
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                chapter_title = tokens[i + 1].content.strip()
                chapter_start_idx = i
                break

    if chapter_title is None or chapter_start_idx is None:
        return None

    # Find first numbered subsection (H2)
    first_subsection_idx = None
    for i in range(chapter_start_idx, len(tokens)):
        token = tokens[i]
        if token.type == "heading_open" and token.tag == "h2":
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                title = tokens[i + 1].content.strip()
                if is_numbered_subsection(title):
                    first_subsection_idx = i
                    break

    # Collect blocks before first subsection
    blocks_before_subsections = []
    if first_subsection_idx is not None:
        blocks_before_subsections = _extract_blocks_in_range(
            tokens,
            chapter_start_idx + 3,  # Skip h1_open, inline, h1_close
            first_subsection_idx,
            source_path,
            chapter_title,
            line_offsets
        )

    # Parse subsections
    subsections = []
    i = first_subsection_idx if first_subsection_idx else chapter_start_idx + 3
    current_subsection = None
    subsection_start_idx = None

    while i < len(tokens):
        token = tokens[i]

        # Check for numbered H2 subsection
        if token.type == "heading_open" and token.tag == "h2":
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                title = tokens[i + 1].content.strip()
                subsection_match = is_numbered_subsection(title)

                if subsection_match:
                    # Extract blocks for previous subsection if exists
                    if current_subsection is not None and subsection_start_idx:
                        _extract_subsection_blocks(
                            tokens,
                            subsection_start_idx,
                            i,
                            current_subsection,
                            source_path,
                            chapter_title,
                            line_offsets
                        )
                        subsections.append(current_subsection)

                    # Start new subsection
                    order_index, clean_title = subsection_match
                    current_subsection = Subsection(
                        order_index=order_index,
                        title=clean_title,
                        subheads=[],
                        blocks=[]
                    )
                    # After h2_open, inline, h2_close
                    subsection_start_idx = i + 3
                    i = subsection_start_idx
                    continue

        i += 1

    # Extract blocks for last subsection
    if current_subsection is not None and subsection_start_idx:
        _extract_subsection_blocks(
            tokens,
            subsection_start_idx,
            len(tokens),
            current_subsection,
            source_path,
            chapter_title,
            line_offsets
        )
        subsections.append(current_subsection)

    return Chapter(
        title=chapter_title,
        blocks_before_subsections=blocks_before_subsections,
        subsections=subsections
    )


def _extract_subsection_blocks(
    tokens: List,
    start_idx: int,
    end_idx: int,
    subsection: Subsection,
    source_path: str,
    chapter_title: str,
    line_offsets: List[int]
) -> None:
    """
    Extract all blocks and subheads from a subsection range.
    Optimized single-pass extraction.

    Args:
        tokens: List of all tokens
        start_idx: Starting token index for subsection
        end_idx: Ending token index for subsection
        subsection: Subsection object to populate
        source_path: Source file path
        chapter_title: Chapter title
        line_offsets: Precomputed line offsets
    """
    i = start_idx
    current_subhead = None
    subhead_start_idx = None
    root_block_start = start_idx  # Track where root blocks start

    while i < end_idx and i < len(tokens):
        token = tokens[i]

        # Check for subhead (H3+)
        if token.type == "heading_open" and token.tag.startswith("h"):
            level = int(token.tag[1])
            if level >= 3:
                # Extract root blocks before this subhead
                if root_block_start < i:
                    root_blocks = _extract_blocks_in_range(
                        tokens,
                        root_block_start,
                        i,
                        source_path,
                        chapter_title,
                        line_offsets,
                        subsection_index=subsection.order_index,
                        subsection_title=subsection.title
                    )
                    subsection.blocks.extend(root_blocks)

                # Extract blocks for previous subhead if exists
                if current_subhead is not None and subhead_start_idx:
                    subhead_blocks = _extract_blocks_in_range(
                        tokens,
                        subhead_start_idx,
                        i,
                        source_path,
                        chapter_title,
                        line_offsets,
                        subsection_index=subsection.order_index,
                        subsection_title=subsection.title,
                        subhead_path=current_subhead.heading_path
                    )
                    current_subhead.blocks = subhead_blocks
                    subsection.subheads.append(current_subhead)

                # Start new subhead
                if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                    subhead_title = tokens[i + 1].content.strip()
                    heading_path = [
                        chapter_title,
                        subsection.title,
                        subhead_title
                    ]

                    current_subhead = Subhead(
                        title=subhead_title,
                        level=level,
                        heading_path=heading_path,
                        blocks=[]
                    )
                    subhead_start_idx = i + 3  # After heading tokens
                    # Next root blocks start after subhead
                    root_block_start = i + 3
                    i = subhead_start_idx
                    continue

        i += 1

    # Extract blocks for last subhead
    if current_subhead is not None and subhead_start_idx:
        subhead_blocks = _extract_blocks_in_range(
            tokens,
            subhead_start_idx,
            end_idx,
            source_path,
            chapter_title,
            line_offsets,
            subsection_index=subsection.order_index,
            subsection_title=subsection.title,
            subhead_path=current_subhead.heading_path
        )
        current_subhead.blocks = subhead_blocks
        subsection.subheads.append(current_subhead)
        root_block_start = end_idx  # No more root blocks

    # Extract remaining root blocks after last subhead
    if root_block_start < end_idx:
        root_blocks = _extract_blocks_in_range(
            tokens,
            root_block_start,
            end_idx,
            source_path,
            chapter_title,
            line_offsets,
            subsection_index=subsection.order_index,
            subsection_title=subsection.title
        )
        subsection.blocks.extend(root_blocks)


def _find_subhead_end(tokens: List, start_idx: int, level: int) -> int:
    """Find where a subhead ends (next heading of level <= subhead level)."""
    for i in range(start_idx + 3, len(tokens)):
        token = tokens[i]
        if token.type == "heading_open":
            token_level = int(token.tag[1])
            if token_level <= level:
                return i
    return len(tokens)


def _find_next_subsection(tokens: List, start_idx: int) -> Optional[int]:
    """Find the start of the next numbered subsection."""
    for i in range(start_idx, len(tokens)):
        token = tokens[i]
        if token.type == "heading_open" and token.tag == "h2":
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                title = tokens[i + 1].content.strip()
                if is_numbered_subsection(title):
                    return i
    return None


def _find_last_subsection_end(tokens: List, last_order_index: int) -> int:
    """Find where the last subsection ends."""
    # Find the last H2 that matches numbered pattern
    last_idx = len(tokens)
    for i in range(len(tokens) - 1, -1, -1):
        token = tokens[i]
        if token.type == "heading_open" and token.tag == "h2":
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                title = tokens[i + 1].content.strip()
                if is_numbered_subsection(title):
                    # Find end of this subsection's content
                    for j in range(i + 3, len(tokens)):
                        if (tokens[j].type == "heading_open" and
                                tokens[j].tag == "h2"):
                            if (j + 1 < len(tokens) and
                                    tokens[j + 1].type == "inline"):
                                next_title = tokens[j + 1].content.strip()
                                if is_numbered_subsection(next_title):
                                    return j
                    return len(tokens)
    return last_idx


def _extract_blocks_in_range(
    tokens: List,
    start_idx: int,
    end_idx: int,
    source_path: str,
    chapter_title: str,
    line_offsets: List[int],
    subsection_index: Optional[int] = None,
    subsection_title: Optional[str] = None,
    subhead_path: Optional[List[str]] = None
) -> List[Block]:
    """
    Extract all blocks from a token range.

    Args:
        tokens: List of all tokens
        start_idx: Starting token index
        end_idx: Ending token index
        source_path: Source file path
        chapter_title: Chapter title for metadata
        line_offsets: Precomputed line offsets
        subsection_index: Optional subsection order index
        subsection_title: Optional subsection title
        subhead_path: Optional subhead path

    Returns:
        List of Block objects
    """
    blocks = []
    i = start_idx

    while i < end_idx and i < len(tokens):
        token = tokens[i]

        # Build base metadata
        meta = {
            "note_path": source_path,
            "chapter_title": chapter_title
        }
        if subsection_index is not None:
            meta["subsection_index"] = subsection_index
        if subsection_title is not None:
            meta["subsection_title"] = subsection_title
        if subhead_path is not None:
            meta["subhead_path"] = subhead_path

        # Paragraph blocks
        if token.type == "paragraph_open":
            j = i + 1
            text_parts = []
            embeds = []
            while j < len(tokens) and tokens[j].type != "paragraph_close":
                if tokens[j].type == "inline":
                    inline_text = tokens[j].content
                    # Extract embeds
                    cleaned_text, embed_targets = extract_embeds_from_text(
                        inline_text
                    )
                    if cleaned_text.strip():
                        text_parts.append(cleaned_text)
                    embeds.extend(embed_targets)
                j += 1

            # Add embeds as media blocks
            for embed_target in embeds:
                position = get_block_position(i, j, tokens, line_offsets)
                blocks.append(Block(
                    type="media",
                    text="",
                    meta={**meta, **position, "target": embed_target}
                ))

            # Add paragraph if it has text
            text = "\n".join([p for p in text_parts if p]).strip()
            if text:
                position = get_block_position(i, j, tokens, line_offsets)
                blocks.append(Block(
                    type="paragraph",
                    text=text,
                    meta={**meta, **position}
                ))
            i = j

        # Fenced code blocks
        elif token.type == "fence":
            position = get_block_position(i, i, tokens, line_offsets)
            blocks.append(Block(
                type="code",
                text=token.content,
                meta={**meta, **position, "lang": (token.info or "").strip()}
            ))

        # Lists
        elif token.type in ("bullet_list_open", "ordered_list_open"):
            if token.type.startswith("ordered"):
                list_kind = "ordered"
            else:
                list_kind = "bullet"
            items = []
            j = i + 1
            depth = 1
            while j < len(tokens) and depth > 0:
                if tokens[j].type in ("bullet_list_open", "ordered_list_open"):
                    depth += 1
                elif tokens[j].type in (
                    "bullet_list_close", "ordered_list_close"
                ):
                    depth -= 1
                elif tokens[j].type == "inline":
                    items.append(tokens[j].content.strip())
                j += 1

            position = get_block_position(i, j, tokens, line_offsets)
            blocks.append(Block(
                type="list",
                text="\n".join(f"- {x}" for x in items if x),
                meta={**meta, **position, "kind": list_kind, "items": items}
            ))
            i = j - 1

        # Blockquotes (callouts or quotes)
        elif token.type == "blockquote_open":
            j = i + 1
            lines = []
            depth = 1
            while j < len(tokens) and depth > 0:
                if tokens[j].type == "blockquote_open":
                    depth += 1
                elif tokens[j].type == "blockquote_close":
                    depth -= 1
                elif tokens[j].type == "inline":
                    lines.append(tokens[j].content)
                j += 1
            text = "\n".join(lines).strip()

            # Check for callout
            first_line = text.splitlines()[0].strip() if text else ""
            m = CALLOUT_RE.match(first_line)
            position = get_block_position(i, j, tokens, line_offsets)

            if m:
                blocks.append(Block(
                    type="callout",
                    text=text,
                    meta={
                        **meta,
                        **position,
                        "kind": m.group("kind").lower(),
                        "title": m.group("title").strip()
                    }
                ))
            else:
                blocks.append(Block(
                    type="quote",
                    text=text,
                    meta={**meta, **position}
                ))
            i = j - 1

        i += 1

    return blocks


def extract_structure(file_path: str) -> DocumentStructure:
    """
    Extract structured representation from an Obsidian markdown file.

    Args:
        file_path: Path to the markdown file

    Returns:
        DocumentStructure with frontmatter and chapter
    """
    path_obj = Path(file_path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Markdown file not found: {file_path}")

    if not path_obj.suffix == ".md":
        raise ValueError(
            f"File must be a markdown file (.md), got: {path_obj.suffix}"
        )

    # Read and parse frontmatter
    with open(path_obj, 'r', encoding='utf-8') as f:
        post = frontmatter.load(f)

    # Extract frontmatter metadata
    frontmatter_data = dict(post.metadata) if post.metadata else {}

    # Parse chapter structure from content
    source_path = str(path_obj.resolve())
    chapter = parse_chapter_structure(post.content, source_path)

    return DocumentStructure(
        frontmatter=frontmatter_data,
        chapter=chapter,
        source_path=source_path
    )
