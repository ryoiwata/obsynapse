"""
Obsidian Markdown → "note AST-lite" JSON converter.

This module parses Obsidian markdown into a simplified JSON structure
with note_id, title, frontmatter, and chunks organized by heading paths.

Example:
    >>> from obsidian_ast_parser import parseObsidianToAstLite
    >>> markdown = '''---
    ... title: My Note
    ... ---
    ... # Introduction
    ... 
    ... This is content.
    ... 
    ... ## Section
    ... 
    ... - Bullet 1
    ... - Bullet 2
    ... '''
    >>> result = parseObsidianToAstLite(markdown, "my_note.md")
    >>> print(result["note_id"])  # "introduction"
    >>> print(result["title"])    # "Introduction"
    >>> print(len(result["chunks"]))  # Number of chunks
"""

import re
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


# Type definitions
Chunk = Union[
    Dict[str, Any],  # { "path": List[str], "text": str }
    Dict[str, Any],  # { "path": List[str], "bullets": List[str] }
]

NoteAstLite = Dict[str, Any]  # { "note_id": str, "title": str, "frontmatter": Dict, "chunks": List[Chunk] }


# Regex patterns
FRONTMATTER_START = re.compile(r'^---\s*$')
FRONTMATTER_END = re.compile(r'^---\s*$')
HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$')
BULLET_RE = re.compile(r'^(\s*)([-*+])\s+(.+)$')
ORDERED_LIST_RE = re.compile(r'^(\s*)(\d+\.)\s+(.+)$')
CODE_FENCE_START = re.compile(r'^```')
BLOCKQUOTE_RE = re.compile(r'^(\s*)>\s*(.*)$')
HORIZONTAL_RULE = re.compile(r'^---+$')


def slugify(text: str) -> str:
    """
    Convert text to kebab-case slug.
    
    Args:
        text: Text to slugify
        
    Returns:
        Kebab-case slug
    """
    # Convert to lowercase
    text = text.lower()
    # Replace underscores with hyphens
    text = text.replace('_', '-')
    # Replace spaces and special chars with hyphens
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    # Remove leading/trailing hyphens
    return text.strip('-')


def parse_frontmatter(markdown: str) -> tuple:
    """
    Parse YAML frontmatter from markdown if present.
    
    Args:
        markdown: Markdown content
        
    Returns:
        Tuple of (frontmatter_dict, remaining_markdown)
    """
    lines = markdown.split('\n')
    
    # Check if starts with frontmatter delimiter
    if not lines or not FRONTMATTER_START.match(lines[0]):
        return {}, markdown
    
    # Find closing delimiter
    end_idx = None
    for i in range(1, len(lines)):
        if FRONTMATTER_END.match(lines[i]):
            end_idx = i
            break
    
    if end_idx is None:
        # No closing delimiter, treat as no frontmatter
        return {}, markdown
    
    # Extract frontmatter lines
    frontmatter_lines = lines[1:end_idx]
    frontmatter_text = '\n'.join(frontmatter_lines)
    
    # Parse YAML frontmatter
    frontmatter_dict = {}
    try:
        # Try using yaml library (PyYAML) if available
        import yaml
        frontmatter_dict = yaml.safe_load(frontmatter_text) or {}
    except ImportError:
        # Fallback: basic key-value parsing
        for line in frontmatter_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                # Try to parse as list if it looks like one
                if value.startswith('[') and value.endswith(']'):
                    # Simple list parsing
                    items = [item.strip().strip('"\'') for item in value[1:-1].split(',')]
                    frontmatter_dict[key] = items
                else:
                    frontmatter_dict[key] = value
    except Exception:
        # If YAML parsing fails, return empty dict
        frontmatter_dict = {}
    
    # Remove frontmatter from markdown
    remaining_lines = lines[end_idx + 1:]
    remaining_markdown = '\n'.join(remaining_lines)
    
    return frontmatter_dict, remaining_markdown


def get_heading_level(line: str) -> Optional[tuple]:
    """
    Check if line is a heading and return level and text.
    
    Args:
        line: Line to check
        
    Returns:
        Tuple of (level, text) if heading, None otherwise
    """
    match = HEADING_RE.match(line)
    if match:
        level = len(match.group(1))
        text = match.group(2).strip()
        return level, text
    return None


def is_bullet_line(line: str) -> bool:
    """Check if line is a bullet list item."""
    return bool(BULLET_RE.match(line) or ORDERED_LIST_RE.match(line))


def is_code_fence(line: str, in_code: bool) -> bool:
    """Check if line starts or ends a code fence."""
    return bool(CODE_FENCE_START.match(line))


def is_blockquote(line: str) -> bool:
    """Check if line is a blockquote."""
    return bool(BLOCKQUOTE_RE.match(line))


def normalize_newlines(text: str) -> str:
    """Normalize Windows newlines to Unix."""
    return text.replace('\r\n', '\n').replace('\r', '\n')


def parseObsidianToAstLite(markdown: str, fileName: Optional[str] = None) -> NoteAstLite:
    """
    Parse Obsidian markdown into AST-lite JSON structure.
    
    Args:
        markdown: Markdown content as string
        fileName: Optional file name for note_id/title fallback
        
    Returns:
        NoteAstLite dictionary with note_id, title, frontmatter, chunks
    """
    # Normalize newlines
    markdown = normalize_newlines(markdown)
    
    # Parse frontmatter
    frontmatter, body = parse_frontmatter(markdown)
    
    # Track heading path
    heading_path: List[str] = []
    heading_levels: List[int] = []  # Track levels for each heading in path
    
    # Chunks accumulator
    chunks: List[Chunk] = []
    
    # Current block state
    current_paragraph_lines: List[str] = []
    current_bullets: List[str] = []
    in_code_block = False
    
    # Track title and note_id
    title: Optional[str] = None
    note_id: Optional[str] = None
    
    lines = body.split('\n')
    
    def flush_paragraph():
        """Flush current paragraph block if non-empty."""
        nonlocal current_paragraph_lines
        if current_paragraph_lines:
            text = '\n'.join(current_paragraph_lines).strip()
            if text:
                chunks.append({
                    "path": heading_path.copy(),
                    "text": text
                })
            current_paragraph_lines = []
    
    def flush_bullets():
        """Flush current bullet block if non-empty."""
        nonlocal current_bullets
        if current_bullets:
            chunks.append({
                "path": heading_path.copy(),
                "bullets": current_bullets.copy()
            })
            current_bullets = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for code fence
        if CODE_FENCE_START.match(line):
            # Flush any pending blocks before entering code
            flush_paragraph()
            flush_bullets()
            
            in_code_block = not in_code_block
            i += 1
            continue
        
        # Skip lines inside code blocks
        if in_code_block:
            i += 1
            continue
        
        # Check for heading
        heading_info = get_heading_level(line)
        if heading_info:
            level, heading_text = heading_info
            
            # Flush any pending blocks before new heading
            flush_paragraph()
            flush_bullets()
            
            # Update heading path
            # Remove headings at same or deeper level
            while heading_levels and heading_levels[-1] >= level:
                heading_levels.pop()
                heading_path.pop()
            
            # Add new heading
            heading_path.append(heading_text)
            heading_levels.append(level)
            
            # Track first H1 as title
            if level == 1 and title is None:
                title = heading_text
                note_id = slugify(heading_text)
            
            i += 1
            continue
        
        # Check for horizontal rule (treat as separator, skip)
        if HORIZONTAL_RULE.match(line.strip()):
            flush_paragraph()
            flush_bullets()
            i += 1
            continue
        
        # Check for blockquote (treat as paragraph text)
        blockquote_match = BLOCKQUOTE_RE.match(line)
        if blockquote_match:
            quote_text = blockquote_match.group(2)
            # Flush bullets if switching to paragraph
            flush_bullets()
            # Remove embed syntax from blockquote if present
            quote_text = re.sub(r'!\[\[([^\]]+)\]\]', '', quote_text).strip()
            if quote_text:
                current_paragraph_lines.append(quote_text)
            i += 1
            continue
        
        # Check for bullet/ordered list
        bullet_match = BULLET_RE.match(line) or ORDERED_LIST_RE.match(line)
        if bullet_match:
            # Flush paragraph BEFORE starting bullet block to preserve label + bullet pattern
            # This ensures the label becomes its own text chunk, and bullets become a separate chunk
            flush_paragraph()
            
            # Extract bullet text (remove marker)
            if BULLET_RE.match(line):
                bullet_text = BULLET_RE.match(line).group(3)
            else:
                bullet_text = ORDERED_LIST_RE.match(line).group(3)
            
            current_bullets.append(bullet_text.strip())
            i += 1
            continue
        
        # Empty line
        if not line.strip():
            # Flush bullets (paragraphs can span blank lines)
            flush_bullets()
            
            # For paragraphs, a single blank line is OK (merge)
            # Multiple blank lines flush the paragraph
            if current_paragraph_lines:
                # Check if next non-empty line is also a paragraph continuation
                next_non_empty = None
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    # Skip code fences
                    if CODE_FENCE_START.match(next_line):
                        break
                    if next_line.strip():
                        next_non_empty = next_line
                        break
                
                # If next line is not a paragraph continuation, flush
                if next_non_empty:
                    next_heading = get_heading_level(next_non_empty)
                    next_bullet = is_bullet_line(next_non_empty)
                    next_blockquote = is_blockquote(next_non_empty)
                    
                    # CRITICAL: Preserve label + bullet pattern
                    # If next line is a bullet, DON'T flush paragraph here
                    # Let the bullet handler flush it, which preserves the label as its own chunk
                    if next_heading or next_blockquote:
                        flush_paragraph()
                    elif next_bullet:
                        # DON'T flush - bullet handler will flush it, preserving label + bullet
                        pass
                    else:
                        # Next is regular paragraph text - check for double blank
                        has_double_blank = False
                        if i + 1 < len(lines) and not lines[i + 1].strip():
                            has_double_blank = True
                        
                        if has_double_blank:
                            flush_paragraph()
                else:
                    # No next non-empty line found - keep paragraph
                    pass
            
            i += 1
            continue
        
        # Regular paragraph line
        # Note: This branch only runs if line is NOT heading, NOT code, NOT empty,
        # NOT bullet, and NOT blockquote
        flush_bullets()
        # Remove embed syntax from line if present (embeds are ignored)
        line_without_embeds = re.sub(r'!\[\[([^\]]+)\]\]', '', line).strip()
        if line_without_embeds:
            current_paragraph_lines.append(line_without_embeds)
        i += 1
    
    # Flush any remaining blocks
    flush_paragraph()
    flush_bullets()
    
    # Merge adjacent chunks of same type under same path
    merged_chunks = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        merged_chunk = chunk.copy()
        
        # Try to merge with following chunks
        j = i + 1
        while j < len(chunks):
            next_chunk = chunks[j]
            
            # Can only merge if same path and same type
            if chunk["path"] != next_chunk["path"]:
                break
            
            # Merge paragraphs
            if "text" in chunk and "text" in next_chunk:
                merged_chunk["text"] = merged_chunk["text"] + "\n\n" + next_chunk["text"]
                j += 1
                continue
            
            # Merge bullets
            if "bullets" in chunk and "bullets" in next_chunk:
                merged_chunk["bullets"].extend(next_chunk["bullets"])
                j += 1
                continue
            
            # Don't merge different types
            break
        
        merged_chunks.append(merged_chunk)
        i = j
    
    # Filter out empty chunks
    filtered_chunks = []
    for chunk in merged_chunks:
        if "text" in chunk and chunk["text"].strip():
            filtered_chunks.append(chunk)
        elif "bullets" in chunk and chunk["bullets"]:
            filtered_chunks.append(chunk)
    
    # Determine title and note_id
    if not title:
        if fileName:
            title = Path(fileName).stem
            note_id = slugify(title)
        else:
            title = ""
            note_id = ""
    
    if not note_id:
        note_id = slugify(title) if title else ""
    
    return {
        "note_id": note_id,
        "title": title or "",
        "frontmatter": frontmatter,
        "chunks": filtered_chunks
    }


if __name__ == "__main__":
    """
    Command-line interface for testing the parser.
    
    Usage:
        python obsidian_ast_parser.py <file_path>
        python obsidian_ast_parser.py <file_path> --pretty
    """
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Parse Obsidian markdown file to AST-lite JSON"
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the markdown file to parse"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print JSON output with indentation"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: print to stdout)"
    )
    
    args = parser.parse_args()
    
    # Read file
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Parse
    try:
        result = parseObsidianToAstLite(markdown_content, str(file_path))
    except Exception as e:
        print(f"Error parsing markdown: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Output
    if args.pretty:
        output_json = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        output_json = json.dumps(result, ensure_ascii=False)
    
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_json)
            print(f"Output written to: {args.output}")
        except Exception as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(output_json)

