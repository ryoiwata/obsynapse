"""
PDF extraction module using PyMuPDF (fitz).

This module provides a PDFExtractor class that extracts text from PDF files
with special handling for monospaced fonts, page numbers, and reading order.
"""

import fitz  # PyMuPDF
from typing import List, Dict, Tuple, Set
import argparse
import logging
import sys
import os
import re
from datetime import datetime


class PDFExtractor:
    """
    Extract text from PDF files using PyMuPDF with markdown formatting.

    Features:
    - Detects monospaced fonts and wraps them in code blocks
    - Sorts text blocks by Y-coordinate to preserve reading order
    - Exports as Markdown-lite with double newlines between paragraphs
    """

    # Monospaced font patterns (case-insensitive)
    MONOSPACED_FONTS = {'courier', 'mono', 'monospace', 'consolas', 'fixed'}

    def __init__(self, pdf_path: str):
        """
        Initialize the PDF extractor.

        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    def _is_monospaced_font(self, font_name: str) -> bool:
        """
        Check if a font name indicates a monospaced font.

        Args:
            font_name: Font name from PDF

        Returns:
            True if the font is monospaced
        """
        if not font_name:
            return False
        font_lower = font_name.lower()
        return any(mono in font_lower for mono in self.MONOSPACED_FONTS)

    def _extract_text_blocks(self, page) -> List[Dict]:
        """
        Extract text blocks from a page using get_text('dict').

        Args:
            page: PyMuPDF page object

        Returns:
            List of text block dictionaries with position and formatting info
        """
        text_dict = page.get_text('dict')
        page_height = getattr(page, 'rect', None).height if page else None
        blocks = []

        for block in text_dict.get('blocks', []):
            if 'lines' not in block:
                continue

            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    text = span.get('text', '').strip()
                    if not text:
                        continue

                    # Get position (bbox Y-coordinate for reading order)
                    bbox = span.get('bbox', [0, 0, 0, 0])
                    y_pos = bbox[1]  # Top Y coordinate

                    # Check if text is bold
                    # PyMuPDF flags: bit 4 (16) indicates bold
                    flags = span.get('flags', 0)
                    is_bold = (
                        (flags & 16) != 0 or
                        'bold' in span.get('font', '').lower()
                    )

                    # Store span info
                    blocks.append({
                        'text': text,
                        'font': span.get('font', ''),
                        'font_size': span.get('size', 0),
                        'y_pos': y_pos,
                        'bbox': bbox,
                        'page_height': page_height,
                        'is_monospaced': self._is_monospaced_font(
                            span.get('font', '')
                        ),
                        'is_bold': is_bold
                    })

        return blocks

    def _sort_blocks_by_position(self, blocks: List[Dict]) -> List[Dict]:
        """
        Sort text blocks using line-based sorting.

        Groups blocks into visual lines based on Y-coordinate tolerance,
        then sorts blocks within each line by X-coordinate. This prevents
        floating text or margin notes with slightly different Y-values from
        appearing out of order.

        Args:
            blocks: List of text block dictionaries

        Returns:
            Sorted list of blocks with line-based ordering
        """
        if not blocks:
            return []

        # Tolerance for grouping blocks into the same line (in pixels)
        LINE_TOLERANCE = 3.0

        # First, sort all blocks by Y-coordinate to process top-to-bottom
        blocks_sorted_by_y = sorted(blocks, key=lambda b: b['y_pos'])

        # Group blocks into lines based on Y-coordinate tolerance
        lines = []
        current_line = []
        current_line_y = None

        for block in blocks_sorted_by_y:
            block_y = block['y_pos']

            # If this is the first block or Y is within tolerance,
            # add to current line
            if (current_line_y is None or
                    abs(block_y - current_line_y) <= LINE_TOLERANCE):
                current_line.append(block)
                # Update current_line_y to the average Y of blocks in this line
                # This helps handle lines with slight variations
                if current_line_y is None:
                    current_line_y = block_y
                else:
                    # Use the minimum Y to keep line position accurate
                    current_line_y = min(current_line_y, block_y)
            else:
                # Y difference exceeds tolerance, start a new line
                if current_line:
                    # Sort current line by X-coordinate (left to right)
                    current_line.sort(key=lambda b: b['bbox'][0])
                    lines.append(current_line)
                current_line = [block]
                current_line_y = block_y

        # Don't forget the last line
        if current_line:
            current_line.sort(key=lambda b: b['bbox'][0])
            lines.append(current_line)

        # Flatten lines back into a single list
        # Blocks are already sorted within each line, and lines are in order
        sorted_blocks = []
        for line in lines:
            sorted_blocks.extend(line)

        return sorted_blocks

    def _group_blocks_into_paragraphs(
        self, blocks: List[Dict]
    ) -> List[List[Dict]]:
        """
        Group consecutive blocks into paragraphs based on proximity.

        Args:
            blocks: Sorted list of text blocks

        Returns:
            List of paragraphs, where each paragraph is a list of blocks
        """
        if not blocks:
            return []

        paragraphs = []
        current_paragraph = [blocks[0]]

        for i in range(1, len(blocks)):
            current_block = blocks[i]
            prev_block = blocks[i - 1]

            # Calculate vertical distance
            y_diff = current_block['y_pos'] - prev_block['y_pos']

            # If blocks are close vertically (within ~1.5 line heights),
            # same paragraph. Otherwise, start a new paragraph.
            avg_font_size = (
                current_block.get('font_size', 12) +
                prev_block.get('font_size', 12)
            ) / 2
            line_height_threshold = avg_font_size * 1.5

            if y_diff <= line_height_threshold:
                current_paragraph.append(current_block)
            else:
                paragraphs.append(current_paragraph)
                current_paragraph = [current_block]

        if current_paragraph:
            paragraphs.append(current_paragraph)

        return paragraphs

    def _get_paragraph_text(self, paragraph: List[Dict]) -> str:
        """
        Get the combined text from a paragraph (list of blocks).

        Args:
            paragraph: List of text block dictionaries

        Returns:
            Combined text string
        """
        return ' '.join(block['text'] for block in paragraph)

    def _ends_with_hyphen(self, text: str) -> bool:
        """
        Check if text ends with a hyphen (indicating word continuation).

        Args:
            text: Text to check

        Returns:
            True if text ends with a hyphen
        """
        text_stripped = text.rstrip()
        return text_stripped.endswith('-')

    def _ends_with_sentence_punctuation(self, text: str) -> bool:
        """
        Check if text ends with sentence-terminal punctuation (. ! ? ").

        Args:
            text: Text to check

        Returns:
            True if text ends with sentence-terminal punctuation
        """
        text_stripped = text.rstrip()
        return text_stripped.endswith(('.', '!', '?', '"'))

    def _should_merge_pages(
        self, last_paragraph: List[Dict], first_paragraph: List[Dict]
    ) -> Tuple[bool, bool]:
        """
        Determine if two paragraphs from consecutive pages should be merged.

        If the previous page ends without terminal punctuation ('.', '!', '?')
        or ends with a hyphen ('-'), paragraphs are merged into a single
        continuous paragraph to handle word-wrapping across page boundaries.

        Args:
            last_paragraph: Last paragraph from previous page
            first_paragraph: First paragraph from next page

        Returns:
            Tuple of (should_merge, remove_hyphen)
            - should_merge: True if paragraphs should be merged
            - remove_hyphen: True if hyphen should be removed from
              last paragraph
        """
        if not last_paragraph or not first_paragraph:
            return False, False

        last_text = self._get_paragraph_text(last_paragraph)

        # Check if last paragraph ends with hyphen (word continuation)
        if self._ends_with_hyphen(last_text):
            return True, True

        # Check if last paragraph doesn't end with sentence punctuation
        # (likely continues on next page)
        if not self._ends_with_sentence_punctuation(last_text):
            return True, False

        return False, False

    def _merge_paragraphs(
        self, last_paragraph: List[Dict], first_paragraph: List[Dict],
        remove_hyphen: bool
    ) -> List[Dict]:
        """
        Merge two paragraphs from consecutive pages.

        Handles hyphenated word fragments (e.g., "engi-" + "neering" ->
        "engineering") by combining the last block of the first paragraph
        with the first block of the second paragraph into a single block.

        Args:
            last_paragraph: Last paragraph from previous page
            first_paragraph: First paragraph from next page
            remove_hyphen: If True, remove trailing hyphen from last paragraph
                          and combine blocks to ensure words join without space

        Returns:
            Merged paragraph (list of blocks)
        """
        merged = last_paragraph.copy()

        if not first_paragraph:
            return merged

        # If we need to remove hyphen, combine the last block with the
        # first block to ensure words join without space
        # This handles cases like "engi-" merging with "neering" ->
        # "engineering"
        if remove_hyphen and merged:
            last_block = merged[-1].copy()
            last_text = last_block['text'].rstrip()
            if last_text.endswith('-'):
                # Remove the hyphen and any trailing whitespace
                last_text_cleaned = last_text[:-1].rstrip()
                # Get first block from next paragraph
                first_block = first_paragraph[0].copy()
                first_text = first_block['text'].lstrip()
                # Combine into single block (no space between)
                last_block['text'] = last_text_cleaned + first_text
                merged[-1] = last_block
                # Append remaining blocks from first paragraph
                merged.extend(first_paragraph[1:])
            else:
                # No hyphen found, just extend normally
                merged.extend(first_paragraph)
        else:
            # Not a hyphen merge, just extend normally
            # Blocks will be joined with space in _paragraphs_to_markdown
            merged.extend(first_paragraph)

        return merged

    def _calculate_baseline_font_size(
        self, all_blocks: List[Dict]
    ) -> float:
        """
        Calculate the baseline (most common) font size in the document.

        This is used to differentiate headers from body text.

        Args:
            all_blocks: List of all text blocks from the document

        Returns:
            Baseline font size (most frequent font size)
        """
        if not all_blocks:
            return 12.0  # Default fallback

        # Count font sizes, rounding to nearest 0.5 for grouping
        font_size_counts = {}
        for block in all_blocks:
            size = block.get('font_size', 0)
            if size > 0:
                # Round to nearest 0.5 for grouping similar sizes
                rounded_size = round(size * 2) / 2
                font_size_counts[rounded_size] = (
                    font_size_counts.get(rounded_size, 0) + 1
                )

        if not font_size_counts:
            return 12.0  # Default fallback

        # Return the most common font size
        baseline = max(font_size_counts.items(), key=lambda x: x[1])[0]
        return baseline

    def _is_header_paragraph(
        self, paragraph: List[Dict], baseline_font_size: float,
        prev_paragraph: List[Dict] = None
    ) -> Tuple[bool, int]:
        """
        Determine if a paragraph is a header and what level it should be.

        Args:
            paragraph: List of blocks in the paragraph
            baseline_font_size: Baseline font size for body text
            prev_paragraph: Previous paragraph (for spacing check)

        Returns:
            Tuple of (is_header, header_level)
            - is_header: True if paragraph is a header
            - header_level: 1, 2, or 3 for #, ##, ### respectively
        """
        if not paragraph:
            return False, 0

        # Extract full text of the paragraph
        paragraph_text = ' '.join(
            block.get('text', '') for block in paragraph
        ).strip()

        # Exclude table and figure captions from being treated as headers
        # Check if text starts with common caption labels (case-insensitive)
        labels_to_exclude = (
            'table', 'figure', 'image', 'chart', 'graph',
            'input', 'output', 'context', 'token', 'description'
        )
        paragraph_text_lower = paragraph_text.lower()
        if any(
            paragraph_text_lower.startswith(label)
            for label in labels_to_exclude
        ):
            return False, 0

        # Heuristic: If it contains parentheses, it's likely a table header,
        # not a section title
        if '(' in paragraph_text or ')' in paragraph_text:
            return False, 0

        # Check for multiple distinct bold spans (table headers often have
        # multiple bold words separated by non-bold text)
        bold_blocks = [
            block for block in paragraph
            if block.get('is_bold', False)
        ]
        non_bold_blocks = [
            block for block in paragraph
            if not block.get('is_bold', False)
        ]
        # If there are multiple bold spans with non-bold text between them,
        # it's likely a table header
        if len(bold_blocks) >= 2 and len(non_bold_blocks) > 0:
            return False, 0

        # Headers are typically short (single line or very few words)
        word_count = len(paragraph_text.split())

        # Headers are usually 1-15 words
        if word_count > 15:
            return False, 0

        # Check if paragraph has significant vertical spacing before it
        # (indicating it's a section break, not inline text)
        has_spacing = False
        if prev_paragraph and paragraph:
            prev_y = prev_paragraph[-1].get('y_pos', 0)
            current_y = paragraph[0].get('y_pos', 0)
            y_diff = current_y - prev_y

            # Check if there's significant vertical spacing
            # (more than 1.5x the average font size)
            avg_font_size = (
                paragraph[0].get('font_size', baseline_font_size) +
                prev_paragraph[-1].get('font_size', baseline_font_size)
            ) / 2
            spacing_threshold = avg_font_size * 1.5

            if y_diff > spacing_threshold:
                has_spacing = True

        # Get font properties of the paragraph
        # Use the maximum font size in the paragraph (headers are consistent)
        max_font_size = max(
            (block.get('font_size', 0) for block in paragraph),
            default=0
        )
        is_bold = any(block.get('is_bold', False) for block in paragraph)

        # Calculate size ratio relative to baseline
        if baseline_font_size > 0:
            size_ratio = max_font_size / baseline_font_size
        else:
            size_ratio = 1.0

        # Header detection criteria:
        # 1. Significantly larger font size OR bold text
        # 2. Short paragraph (already checked above)
        # 3. Has spacing before it (if previous paragraph exists)
        is_header = False
        header_level = 0

        # Very large font (1.5x+ baseline) or bold + large -> Header 1
        if (size_ratio >= 1.5 or (is_bold and size_ratio >= 1.3)):
            is_header = True
            header_level = 1
        # Medium-large font (1.3x+ baseline) or bold -> Header 2
        elif (size_ratio >= 1.3 or (is_bold and size_ratio >= 1.15)):
            is_header = True
            header_level = 2
        # Slightly larger than baseline or bold -> Header 3
        elif (size_ratio >= 1.15 or is_bold):
            is_header = True
            header_level = 3

        # Additional check: if no spacing and not significantly larger,
        # it might be a bold list item or inline text, not a header
        if prev_paragraph and not has_spacing and size_ratio < 1.3:
            # Be more conservative - require both bold AND larger size
            if not (is_bold and size_ratio >= 1.2):
                is_header = False
                header_level = 0

        return is_header, header_level

    def _paragraphs_to_markdown(
        self, paragraphs: List[List[Dict]], all_blocks: List[Dict] = None
    ) -> List[str]:
        """
        Convert paragraphs to markdown format with code block and
        header handling.

        Args:
            paragraphs: List of paragraphs, where each paragraph is a
                        list of blocks
            all_blocks: All blocks from the document (for baseline calculation)

        Returns:
            List of markdown strings
        """
        # Calculate baseline font size for header detection
        if all_blocks:
            baseline_font_size = self._calculate_baseline_font_size(all_blocks)
        else:
            # Fallback: collect all blocks from paragraphs
            all_blocks_flat = []
            for para in paragraphs:
                all_blocks_flat.extend(para)
            baseline_font_size = self._calculate_baseline_font_size(
                all_blocks_flat
            )

        markdown_parts = []
        current_code_block = []
        in_code_block = False

        prev_paragraph = None
        for paragraph in paragraphs:
            # Extract paragraph text
            paragraph_text = ' '.join(
                block.get('text', '') for block in paragraph
            ).strip()

            # Skip if empty
            if not paragraph_text:
                continue

            # Check if paragraph contains monospaced text
            paragraph_is_monospaced = any(
                block.get('is_monospaced', False) for block in paragraph
            )

            # Check if paragraph is a header (but not if it's monospaced)
            is_header = False
            header_level = 0
            if not paragraph_is_monospaced:
                is_header, header_level = self._is_header_paragraph(
                    paragraph, baseline_font_size, prev_paragraph
                )

            if paragraph_is_monospaced:
                # Start or continue code block
                if not in_code_block:
                    if markdown_parts:  # Add newline before code block
                        markdown_parts.append('')
                    in_code_block = True

                # Add all text from paragraph to code block
                paragraph_text = ' '.join(
                    block.get('text', '') for block in paragraph
                )
                current_code_block.append(paragraph_text)
            else:
                # End code block if we were in one
                if in_code_block:
                    code_text = '\n'.join(current_code_block)
                    markdown_parts.append(f'```\n{code_text}\n```')
                    current_code_block = []
                    in_code_block = False

                if paragraph_text:
                    # Emit header or regular paragraph
                    if is_header:
                        header_prefix = '#' * header_level
                        markdown_parts.append(
                            f'{header_prefix} {paragraph_text}'
                        )
                    else:
                        # Add regular paragraph as-is
                        markdown_parts.append(paragraph_text)

            # Update previous paragraph for spacing checks
            prev_paragraph = paragraph

        # Close any remaining code block
        if in_code_block:
            code_text = '\n'.join(current_code_block)
            markdown_parts.append(f'```\n{code_text}\n```')

        return markdown_parts

    def _strip_digits_and_separators(self, text: str) -> str:
        """
        Remove leading/trailing digits and common separator characters.

        Args:
            text: Text to clean

        Returns:
            Cleaned string with surrounding digits/separators removed
        """
        # Strip leading: digits, whitespace, hyphens, em-dashes, colons, pipes
        cleaned = re.sub(r'^[\s\d\-–—:|]+', '', text)
        # Strip trailing: digits, whitespace, hyphens, em-dashes, colons, pipes
        cleaned = re.sub(r'[\s\d\-–—:|]+$', '', cleaned)
        return cleaned.strip()

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison by lowercasing, removing punctuation,
        and collapsing whitespace.

        Args:
            text: Text to normalize

        Returns:
            Normalized text string
        """
        # Remove punctuation and separators, keep alphanumerics and whitespace
        normalized = re.sub(r'[^\w\s]', ' ', text.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    def _is_markdown_header_line(self, line: str) -> bool:
        """
        Check if a line is a markdown header (levels 1-4).
        """
        return bool(re.match(r'^\s*#{1,4}\s+\S', line))

    def _is_pipe_candidate(self, line: str) -> bool:
        """
        Determine if a line is a candidate for pipe-based header stripping.

        Heuristics:
        - Contains '|'
        - Matches patterns with page numbers on the right
        - Matches patterns with numbers on the left
        - Either side has at least 3 alphabetic characters
        - Skips markdown table lines
        """
        if '|' not in line:
            return False

        stripped = line.strip()
        if not stripped:
            return False

        # Avoid markdown tables
        if stripped.startswith('|'):
            return False
        if stripped.count('|') >= 2 and (
            '---' in stripped or re.search(r'\|\s*-{2,}', stripped)
        ):
            return False

        # Common pattern: header | 13
        if re.search(r'\|\s*\d+\b', stripped):
            return True

        # Pattern: 14 | Chapter ...
        if re.match(r'^\s*\d+\s*\|', stripped):
            return True

        # Fallback heuristic: left side has meaningful text
        left = stripped.split('|', 1)[0]
        if sum(1 for c in left if c.isalpha()) >= 3:
            return True

        # Heuristic: right side has meaningful text
        right = stripped.split('|', 1)[1]
        if sum(1 for c in right if c.isalpha()) >= 3:
            return True

        return False

    def _header_matches_seen(
        self, candidate_norm: str, headers_seen: Set[str]
    ) -> bool:
        """
        Check if a normalized candidate matches any seen header.

        Exact match or conservative substring match for longer titles.
        """
        if not candidate_norm:
            return False

        for seen in headers_seen:
            if candidate_norm == seen:
                return True
            min_len = min(len(candidate_norm), len(seen))
            if min_len >= 10 and (
                candidate_norm in seen or seen in candidate_norm
            ):
                return True
        return False

    def _strip_header_prefix(self, text: str, header_norm: str) -> str:
        """
        Remove a header-like prefix from text using a normalized header hint.
        """
        tokens = header_norm.split()
        if not tokens:
            return text

        # Build a loose pattern that tolerates punctuation/separators
        pattern = r'^\s*' + r'[\s\W]*'.join(
            re.escape(token) for token in tokens
        )
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return text[match.end():]
        return text

    def _strip_repeated_pipe_header_fragment(
        self, line: str, headers_seen: Set[str]
    ) -> str | None:
        """
        If line contains a repeated running header/footer with '|', remove
        just the repeated header fragment (and surrounding numbers/separators)
        and return the remaining text. Return '' if nothing remains. Return
        None if the line should be left unchanged (not a match).
        """
        if '|' not in line or not headers_seen:
            return None

        left, right = line.split('|', 1)

        # Clean and normalize the left side (title candidate)
        left_clean = self._strip_digits_and_separators(left)
        left_norm = self._normalize_text(left_clean)

        # Preserve raw right side for reverse pattern heuristics
        right_original = right.strip()

        # Remove leading page numbers and separators from the right side
        right_raw = re.sub(r'^\s*\d+\s*', '', right_original)
        right_raw = re.sub(r'^[\s\-–—:|]+', '', right_raw)

        # Primary case: title on the left, page/content on the right
        if self._header_matches_seen(left_norm, headers_seen):
            if not right_raw:
                return ''
            return right_raw

        # Reverse pattern: number on the left, header on the right
        left_is_number = bool(re.fullmatch(r'\s*\d+\s*', left))
        if left_is_number:
            right_norm_full = self._normalize_text(right_original)
            if self._header_matches_seen(right_norm_full, headers_seen):
                stripped = self._strip_header_prefix(
                    right_original, right_norm_full
                )
                stripped = stripped.strip()
                # Clean any lingering separators after removing the header text
                stripped = re.sub(r'^[\s\-–—:|]+', '', stripped)
                if not stripped:
                    return ''
                return stripped

        return None

    def _remove_repeated_pipe_headers(self, markdown_text: str) -> str:
        """
        Remove repeated running headers/footers that include a pipe character.
        Only removes repeats after the first true markdown header is seen.
        """
        if not markdown_text:
            return markdown_text

        lines = markdown_text.split('\n')
        headers_seen: Set[str] = set()
        cleaned_lines: List[str] = []
        in_code_block = False

        for line in lines:
            stripped_line = line.strip()

            # Track fenced code blocks to avoid altering code content
            if stripped_line.startswith('```'):
                in_code_block = not in_code_block
                cleaned_lines.append(line)
                continue

            if in_code_block:
                cleaned_lines.append(line)
                continue

            # Always keep the first occurrence of markdown headers
            if self._is_markdown_header_line(line):
                header_title = line.lstrip('#').strip()
                header_norm = self._normalize_text(header_title)
                if header_norm:
                    headers_seen.add(header_norm)
                cleaned_lines.append(line)
                continue

            if self._is_pipe_candidate(line):
                stripped = self._strip_repeated_pipe_header_fragment(
                    line, headers_seen
                )
                if stripped is None:
                    cleaned_lines.append(line)
                elif stripped == '':
                    # Drop the line entirely
                    continue
                else:
                    cleaned_lines.append(stripped)
            else:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _remove_leading_page_number_prefixes(self, markdown_text: str) -> str:
        """
        Remove leading page-number tokens that slipped into paragraph starts.
        Skips markdown headers and fenced code blocks.
        """
        if not markdown_text:
            return markdown_text

        lines = markdown_text.split('\n')
        cleaned: List[str] = []
        in_code_block = False
        at_paragraph_start = True
        removed_once = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith('```'):
                in_code_block = not in_code_block
                cleaned.append(line)
                at_paragraph_start = False
                continue

            if in_code_block:
                cleaned.append(line)
                continue

            if stripped == '':
                cleaned.append(line)
                at_paragraph_start = True
                continue

            if self._is_markdown_header_line(line):
                cleaned.append(line)
                at_paragraph_start = False
                continue

            if at_paragraph_start:
                if (not removed_once) and (
                    match := re.match(r'^(\s*)(\d{1,3})\s+([A-Za-z])', line)
                ):
                    num = int(match.group(2))
                    if 1 <= num <= 5000:
                        if not re.match(r'^\s*\d{1,3}\s*[\.\):\]]', line):
                            indent = match.group(1)
                            rest = line[match.end(2):].lstrip()
                            line = indent + rest
                            removed_once = True

            cleaned.append(line)
            at_paragraph_start = False

        return '\n'.join(cleaned)

    def _remove_numbered_paragraphs(self, markdown_text: str) -> str:
        """
        Drop entire paragraphs that begin with a bare numeric prefix like
        '2 For ...' (digits + space + letter). Skips markdown headers and
        fenced code blocks.
        """
        if not markdown_text:
            return markdown_text

        lines = markdown_text.split('\n')
        cleaned: List[str] = []
        in_code_block = False

        for line in lines:
            stripped = line.strip()

            # Preserve and track fenced code blocks
            if stripped.startswith('```'):
                in_code_block = not in_code_block
                cleaned.append(line)
                continue
            if in_code_block:
                cleaned.append(line)
                continue

            # Preserve markdown headers
            if self._is_markdown_header_line(line):
                cleaned.append(line)
                continue

            # Drop paragraphs that start with: 1–2 digits + space + letter
            if re.match(r'^\s*\d{1,2}\s+[A-Za-z]', line):
                continue

            cleaned.append(line)

        return '\n'.join(cleaned)

    def _collect_known_headers(
        self,
        page_paragraphs: List[List[List[Dict]]],
        baseline_font_size: float
    ) -> Set[str]:
        """
        Collect normalized header titles from paragraph structures using
        the existing header detection heuristic.
        """
        headers: Set[str] = set()
        prev_paragraph = None

        for page in page_paragraphs:
            for paragraph in page:
                # Skip monospaced paragraphs from header consideration
                paragraph_is_monospaced = any(
                    block.get('is_monospaced', False) for block in paragraph
                )
                if paragraph_is_monospaced:
                    prev_paragraph = paragraph
                    continue

                is_header, _ = self._is_header_paragraph(
                    paragraph, baseline_font_size, prev_paragraph
                )
                if is_header:
                    title = self._get_paragraph_text(paragraph)
                    title_norm = self._normalize_text(title)
                    if title_norm:
                        headers.add(title_norm)

                prev_paragraph = paragraph

        return headers

    def _paragraph_in_header_footer_zone(self, paragraph: List[Dict]) -> bool:
        """
        Determine if a paragraph sits within the top/bottom 8% of the page.
        """
        if not paragraph:
            return False

        first_block = paragraph[0]
        page_height = first_block.get('page_height')
        if not page_height:
            return False

        y_top = min(
            block.get('bbox', [0, 0, 0, 0])[1]
            for block in paragraph
        )

        top_zone = page_height * 0.08
        bottom_zone = page_height * 0.92

        return y_top <= top_zone or y_top >= bottom_zone

    def _remove_running_headers_linguistic(
        self,
        page_paragraphs: List[List[List[Dict]]],
        headers_seen: Set[str]
    ) -> List[List[List[Dict]]]:
        """
        Remove running headers/footers using linguistic heuristics
        (no Y-coords).

        Protections:
        - Lowercase start = absolute shield (sentence continuation)
        - Sentence continuity: if previous page ended mid-sentence, the first
          paragraph on the next page has immunity unless it is a pure header.
        - Header scalpel: regex-based removal of [Number] | [Known Title],
          stripping only the matched header and preserving remainder.
        - Strict known title check: only attempts removal when matching known
          headers (headers_seen).
        """
        if not page_paragraphs or not headers_seen:
            return page_paragraphs

        filtered_pages: List[List[List[Dict]]] = []
        is_sentence_open = False

        for page in page_paragraphs:
            filtered_paragraphs: List[List[Dict]] = []
            is_first_paragraph = True

            for paragraph in page:
                paragraph_text = self._get_paragraph_text(paragraph)

                # Lowercase shield
                leading_alpha_match = re.search(r'[A-Za-z]', paragraph_text)
                if leading_alpha_match:
                    first_letter = paragraph_text[leading_alpha_match.start()]
                    if first_letter.islower():
                        filtered_paragraphs.append(paragraph)
                        is_first_paragraph = False
                        continue

                # Continuity guard: first paragraph after open sentence
                if is_first_paragraph and is_sentence_open:
                    if '|' in paragraph_text:
                        # Only drop if it's a pure header; otherwise keep
                        if not self._is_pure_footer_pattern(
                            paragraph_text, headers_seen
                        ):
                            filtered_paragraphs.append(paragraph)
                            is_first_paragraph = False
                            continue
                    else:
                        filtered_paragraphs.append(paragraph)
                        is_first_paragraph = False
                        continue

                # If no pipe, keep
                if '|' not in paragraph_text:
                    filtered_paragraphs.append(paragraph)
                    is_first_paragraph = False
                    continue

                # Apply header scalpel
                scalpel_result = self._header_scalpel(
                    paragraph_text, headers_seen
                )

                if scalpel_result is None:
                    filtered_paragraphs.append(paragraph)
                elif scalpel_result == '':
                    # Pure header - drop
                    pass
                else:
                    # Replace first block text with remainder
                    modified_paragraph = paragraph.copy()
                    if modified_paragraph:
                        modified_paragraph[0] = modified_paragraph[0].copy()
                        modified_paragraph[0]['text'] = scalpel_result
                    filtered_paragraphs.append(modified_paragraph)

                is_first_paragraph = False

            # Update sentence-open state using last kept paragraph
            if filtered_paragraphs:
                last_text = self._get_paragraph_text(filtered_paragraphs[-1])
                # Open if hyphen-continued or missing terminal punctuation
                is_sentence_open = (
                    self._ends_with_hyphen(last_text) or
                    not self._ends_with_sentence_punctuation(last_text)
                )
            else:
                is_sentence_open = False

            filtered_pages.append(filtered_paragraphs)

        return filtered_pages

    def _header_scalpel(
        self, text: str, headers_seen: Set[str]
    ) -> str | None:
        """
        Regex-based scalpel to remove [Number] | [Known Title] patterns while
        preserving any trailing content.

        Returns:
            ''   if pure header (should be dropped)
            str  if header removed and content preserved
            None if no header match (keep original)
        """
        if '|' not in text:
            return None

        # Try each known header to build a scalpel pattern
        for known_header in headers_seen:
            header_tokens = known_header.split()
            if not header_tokens:
                continue

            header_pattern = r'[\s\W]*'.join(
                re.escape(token) for token in header_tokens
            )

            # Pattern 1: "42 | Chapter Title" or "42 | Chapter Title content"
            pattern1 = (
                r'^\s*(\d{1,4})\s*\|\s*'
                r'(?:\d+\s*)?'        # Optional leading number
                r'[\s\-–—:|]*'        # Optional separators
                r'(' + header_pattern + r')'
                r'[\s\-–—:|]*'        # Optional trailing separators
            )

            match = re.search(pattern1, text, re.IGNORECASE)
            if match:
                header_end = match.end()
                remainder = text[header_end:].strip()
                remainder = re.sub(r'^[\s\-–—:|]+', '', remainder)

                matched_text = text[match.start():match.end()]
                matched_clean = self._strip_digits_and_separators(matched_text)
                matched_norm = self._normalize_text(matched_clean)
                if matched_norm.startswith(known_header):
                    if remainder:
                        return remainder
                    return ''

            # Pattern 2: "Chapter Title | 42"
            pattern2 = (
                r'^(' + header_pattern + r')'
                r'[\s\-–—:|]*'
                r'\s*\|\s*(\d{1,4})\s*$'
            )

            match = re.search(pattern2, text, re.IGNORECASE)
            if match:
                return ''

        return None

    def _is_pure_footer_pattern(
        self, text: str, headers_seen: Set[str]
    ) -> bool:
        """
        Check if text matches a pure footer pattern: [Number] | [Title]
        or [Title] | [Number] where Title exactly matches a known header.

        Returns True only if the entire text is just the footer pattern
        with no additional body content.
        """
        if '|' not in text:
            return False

        parts = text.split('|', 1)
        if len(parts) != 2:
            return False

        left = parts[0].strip()
        right = parts[1].strip()

        # Check if one side is just a number
        left_is_num = bool(re.fullmatch(r'\d{1,4}', left))
        right_is_num = bool(re.fullmatch(r'\d{1,4}', right))

        if left_is_num:
            # Pattern: "42 | Title"
            right_clean = self._strip_digits_and_separators(right)
            right_norm = self._normalize_text(right_clean)
            # Must be exact match (not substring) for pure footer
            return right_norm in headers_seen

        if right_is_num:
            # Pattern: "Title | 42"
            left_clean = self._strip_digits_and_separators(left)
            left_norm = self._normalize_text(left_clean)
            # Must be exact match (not substring) for pure footer
            return left_norm in headers_seen

        return False

    def _strip_footer_if_pure_match(
        self,
        paragraph: List[Dict],
        paragraph_text: str,
        headers_seen: Set[str]
    ) -> str | None:
        """
        If paragraph is a pure footer match, return '' to drop it.
        If it's a partial match with extra content, use regex to strip just the
        header portion and return the remainder. If no match, return None.

        Uses regex-based pattern matching to handle cases like:
        "14 | Chapter 1: Introduction... Figure 1-6..." where we strip
        the header and keep "Figure 1-6...".

        Returns:
            '' if pure footer (should be dropped)
            str if partial match (stripped remainder to keep)
            None if no match (keep original)
        """
        if '|' not in paragraph_text:
            return None

        parts = paragraph_text.split('|', 1)
        if len(parts) != 2:
            return None

        left = parts[0].strip()
        right = parts[1].strip()

        left_is_num = bool(re.fullmatch(r'\d{1,4}', left))
        right_is_num = bool(re.fullmatch(r'\d{1,4}', right))

        # Pattern: "42 | Title" or "42 | Title extra content"
        if left_is_num:
            # Remove leading page number from right side
            right_after_num = re.sub(r'^\s*\d+\s*', '', right)
            right_after_num = re.sub(r'^[\s\-–—:|]+', '', right_after_num)
            right_clean = self._strip_digits_and_separators(right_after_num)
            right_norm = self._normalize_text(right_clean)

            # Check for exact match with known header
            if right_norm in headers_seen:
                # Pure footer - drop it
                return ''

            # Check if normalized text starts with a known header
            for known_header in headers_seen:
                if right_norm.startswith(known_header):
                    # Partial match - use regex to strip header pattern
                    # Build regex pattern to match: [number] | [header text]
                    # The header text may have punctuation, so we need to
                    # match it flexibly
                    header_tokens = known_header.split()
                    if header_tokens:
                        # Create a pattern that matches the header with
                        # flexible punctuation/whitespace
                        header_pattern = r'[\s\W]*'.join(
                            re.escape(token) for token in header_tokens
                        )
                        # Match: [number] | [optional spaces] [header pattern]
                        full_pattern = (
                            r'^\s*\d{1,4}\s*\|\s*'
                            r'(?:\d+\s*)?'  # Optional leading number
                            r'[\s\-–—:|]*'  # Optional separators
                            r'(' + header_pattern + r')'
                            r'[\s\-–—:|]*'  # Optional trailing separators
                        )
                        match = re.search(
                            full_pattern, paragraph_text, re.IGNORECASE
                        )
                        if match:
                            # Found the header pattern - extract everything
                            # after it
                            header_end = match.end()
                            remainder = paragraph_text[header_end:].strip()
                            remainder = re.sub(r'^[\s\-–—:|]+', '', remainder)
                            if remainder:
                                return remainder
                    # Fallback: try to strip header prefix from right side
                    if len(right_norm) > len(known_header) + 5:
                        remainder = self._strip_header_prefix(
                            right_after_num, known_header
                        )
                        remainder = remainder.strip()
                        remainder = re.sub(r'^[\s\-–—:|]+', '', remainder)
                        if remainder:
                            return remainder
                    # No substantial remainder - treat as pure footer
                    return ''

        # Pattern: "Title | 42" or "Title extra | 42"
        if right_is_num:
            left_clean = self._strip_digits_and_separators(left)
            left_norm = self._normalize_text(left_clean)

            # Check for exact match
            if left_norm in headers_seen:
                # Pure footer - drop it
                return ''

            # Check if left starts with a known header
            for known_header in headers_seen:
                if left_norm.startswith(known_header):
                    # Partial match - use regex to strip header pattern
                    header_tokens = known_header.split()
                    if header_tokens:
                        header_pattern = r'[\s\W]*'.join(
                            re.escape(token) for token in header_tokens
                        )
                        # Match: [header pattern] [optional content] | [number]
                        full_pattern = (
                            r'^(' + header_pattern + r')'
                            r'[\s\-–—:|]*'  # Optional separators
                            r'.*?'  # Any content between header and pipe
                            r'\s*\|\s*\d{1,4}\s*$'
                        )
                        match = re.search(
                            full_pattern, paragraph_text, re.IGNORECASE
                        )
                        if match:
                            # Extract content between header and pipe
                            header_end = match.end(1)
                            pipe_start = paragraph_text.rfind('|')
                            if pipe_start > header_end:
                                remainder = paragraph_text[
                                    header_end:pipe_start
                                ].strip()
                                remainder = re.sub(
                                    r'^[\s\-–—:|]+', '', remainder
                                )
                                if remainder:
                                    return remainder
                    # Fallback: try to extract remainder
                    if len(left_norm) > len(known_header) + 5:
                        remainder = self._strip_header_prefix(
                            left, known_header
                        )
                        remainder = remainder.strip()
                        remainder = re.sub(r'^[\s\-–—:|]+', '', remainder)
                        if remainder:
                            return remainder
                    # No substantial remainder - treat as pure footer
                    return ''

        return None

    def extract(self) -> str:
        """
        Extract text from the PDF and return as Markdown-lite string.

        Processes pages individually to preserve correct reading order.
        Each page is fully processed (extract, sort, group) before
        moving to the next page. This prevents text from different pages
        from intermingling during sorting.

        After all pages are processed, paragraphs are evaluated for merging
        across page boundaries:
        - If a page ends with a hyphen, it's merged with the next page
          (hyphen removed)
        - If a page doesn't end with sentence punctuation, it's merged
          with the next page (single space)

        Returns:
            Markdown-lite formatted string with double newlines between
            paragraphs
        """
        page_paragraphs = []
        # Collect all blocks for baseline font size calculation
        all_blocks = []

        # Process each page individually to preserve reading order
        # This ensures Page 1 content is fully processed before Page 2
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]

            # Extract blocks from this page
            blocks = self._extract_text_blocks(page)
            all_blocks.extend(blocks)  # Collect for baseline calculation

            # Sort blocks by position (within this page only)
            # This ensures correct reading order within each page
            sorted_blocks = self._sort_blocks_by_position(blocks)

            # Group into paragraphs (within this page only)
            # Paragraphs are grouped based on vertical proximity
            paragraphs = self._group_blocks_into_paragraphs(sorted_blocks)

            # Store paragraphs for this page (not yet converted to markdown)
            if paragraphs:
                page_paragraphs.append(paragraphs)

        # Calculate baseline font size for header detection
        baseline_font_size = self._calculate_baseline_font_size(all_blocks)

        # Collect known headers from the document for positional filtering
        known_headers = self._collect_known_headers(
            page_paragraphs, baseline_font_size
        )

        # Remove running headers/footers using linguistic heuristics
        page_paragraphs = self._remove_running_headers_linguistic(
            page_paragraphs, known_headers
        )

        # Merge paragraphs across page boundaries where appropriate
        merged_paragraphs = self._merge_pages_paragraphs(page_paragraphs)

        # Convert all merged paragraphs to markdown
        # Pass all_blocks for baseline font size calculation
        markdown_parts = self._paragraphs_to_markdown(
            merged_paragraphs, all_blocks
        )

        # Join with double newlines between paragraphs
        markdown_text = '\n\n'.join(markdown_parts)

        # Remove repeated running headers/footers that include pipes
        markdown_text = self._remove_repeated_pipe_headers(markdown_text)

        # Remove stray page numbers merged into paragraph starts
        markdown_text = self._remove_leading_page_number_prefixes(
            markdown_text
        )

        # Drop paragraphs starting with bare numeric prefixes
        markdown_text = self._remove_numbered_paragraphs(markdown_text)

        return markdown_text

    def _merge_pages_paragraphs(
        self, page_paragraphs: List[List[List[Dict]]]
    ) -> List[List[Dict]]:
        """
        Merge paragraphs across page boundaries where text continues.

        Evaluates the last paragraph of each page with the first paragraph
        of the next page to determine if they should be merged.

        Args:
            page_paragraphs: List of pages, where each page is a list of
                            paragraphs (each paragraph is a list of blocks)

        Returns:
            Flattened list of paragraphs with page boundaries merged
            where appropriate
        """
        if not page_paragraphs:
            return []

        merged = []
        current_page_paragraphs = page_paragraphs[0]

        for page_idx in range(len(page_paragraphs) - 1):
            next_page_paragraphs = page_paragraphs[page_idx + 1]

            # If current page has no paragraphs, just add next page's
            # paragraphs
            if not current_page_paragraphs:
                current_page_paragraphs = next_page_paragraphs
                continue

            # If next page has no paragraphs, add current page's paragraphs
            if not next_page_paragraphs:
                merged.extend(current_page_paragraphs)
                current_page_paragraphs = []
                continue

            # Get last paragraph of current page
            last_para = (
                current_page_paragraphs[-1]
                if current_page_paragraphs else None
            )
            last_para_idx = len(current_page_paragraphs) - 1

            # Get first paragraph of next page
            first_para = (
                next_page_paragraphs[0]
                if next_page_paragraphs else None
            )
            first_para_idx = 0

            # If we found valid paragraphs to potentially merge
            if last_para and first_para:
                # Check if they should be merged
                should_merge, remove_hyphen = self._should_merge_pages(
                    last_para, first_para
                )

                if should_merge:
                    # Merge the last paragraph of current page with first
                    # paragraph of next page
                    merged_para = self._merge_paragraphs(
                        last_para, first_para, remove_hyphen
                    )
                    # Add all paragraphs up to (but not including) last_para
                    merged.extend(current_page_paragraphs[:last_para_idx])
                    # Add the merged paragraph
                    merged.append(merged_para)
                    # Set remaining paragraphs from next page
                    # (excluding first_para)
                    current_page_paragraphs = (
                        next_page_paragraphs[first_para_idx + 1:]
                    )
                else:
                    # No merging needed - add all paragraphs from current page
                    merged.extend(current_page_paragraphs)
                    # Start fresh with next page's paragraphs
                    current_page_paragraphs = next_page_paragraphs
            else:
                # No valid paragraphs to merge - add current and move to next
                merged.extend(current_page_paragraphs)
                current_page_paragraphs = next_page_paragraphs

        # Add any remaining paragraphs from the last page
        if current_page_paragraphs:
            merged.extend(current_page_paragraphs)

        return merged

    def close(self):
        """
        Close the PDF document.

        Defensive cleanup that checks if document is already closed
        using PyMuPDF's is_closed attribute and logs any errors
        without raising exceptions.
        """
        if self.doc and not self.doc.is_closed:
            try:
                self.doc.close()
            except Exception as e:
                # Only log if it's not a ValueError about document being closed
                if not (
                    isinstance(e, ValueError) and
                    'document closed' in str(e).lower()
                ):
                    # Try to log legitimate errors if logger is available
                    try:
                        logger = logging.getLogger(__name__)
                        if logger.handlers:
                            logger.error(
                                f'Error closing PDF document: {str(e)}',
                                exc_info=True
                            )
                    except Exception:
                        # Logger not available or error logging failed
                        # Silently continue to prevent cascade failures
                        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """
        Silent cleanup on deletion.

        This is a fallback cleanup method. Since the with statement
        handles primary cleanup via close(), this should be silent.
        Only logs errors that are NOT related to the document already
        being closed.
        """
        try:
            # Check if document exists and is not already closed
            if hasattr(self, 'doc') and self.doc and not self.doc.is_closed:
                self.doc.close()
            # If document is already closed, do nothing (silent)
        except ValueError as e:
            # Silently ignore ValueError about document being closed
            # This is expected during garbage collection after with statement
            if 'document closed' not in str(e).lower():
                # Different ValueError - try to log it
                try:
                    logger = logging.getLogger(__name__)
                    if logger.handlers:
                        logger.error(
                            f'ValueError during PDFExtractor cleanup: '
                            f'{str(e)}',
                            exc_info=True
                        )
                except Exception:
                    pass
        except Exception as e:
            # Log legitimate errors (not document-closed related)
            try:
                logger = logging.getLogger(__name__)
                if logger.handlers:
                    logger.error(
                        f'Error during PDFExtractor cleanup: {str(e)}',
                        exc_info=True
                    )
            except Exception:
                # Logger not available or error during logging
                # Silently ignore to prevent "Exception ignored" warnings
                pass


def get_timestamp() -> str:
    """
    Generate a timestamp string in YYYYMMDD_HHMMSS format.

    Returns:
        Timestamp string
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def setup_log_directories(base_dir: str = 'logs') -> Tuple[str, str]:
    """
    Create organized log directory structure.

    Args:
        base_dir: Base directory for logs (default: 'logs')

    Returns:
        Tuple of (processing_dir, errors_dir) paths
    """
    processing_dir = os.path.join(base_dir, 'processing')
    errors_dir = os.path.join(base_dir, 'errors')

    os.makedirs(processing_dir, exist_ok=True)
    os.makedirs(errors_dir, exist_ok=True)

    return processing_dir, errors_dir


def setup_logging(
    base_log_dir: str = 'logs',
    log_filename: str = None
) -> logging.Logger:
    """
    Set up triple-logging system with console and file handlers.

    Args:
        base_log_dir: Base directory for log files
        log_filename: Optional custom log filename (without timestamp)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler - INFO level, clean format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Set up directory structure
    processing_dir, errors_dir = setup_log_directories(base_log_dir)

    # Processing log handler - DEBUG level, detailed format
    timestamp = get_timestamp()
    if log_filename:
        # Use custom filename with timestamp
        log_basename = log_filename
        if not log_basename.endswith('.log'):
            log_basename += '.log'
        processing_log_file = os.path.join(
            processing_dir, f'{timestamp}_{log_basename}'
        )
    else:
        processing_log_file = os.path.join(
            processing_dir, f'{timestamp}_extraction.log'
        )

    processing_handler = logging.FileHandler(
        processing_log_file, mode='w'
    )
    processing_handler.setLevel(logging.DEBUG)
    processing_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - '
        '%(filename)s:%(lineno)d - %(message)s'
    )
    processing_handler.setFormatter(processing_formatter)
    logger.addHandler(processing_handler)

    # Error log handler - ERROR and CRITICAL levels only, detailed format
    # with full stack traces
    error_log_file = os.path.join(
        errors_dir, f'{timestamp}_errors.log'
    )
    error_handler = logging.FileHandler(error_log_file, mode='w')
    error_handler.setLevel(logging.ERROR)  # Captures ERROR and CRITICAL
    # Formatter that includes full exception tracebacks
    # The standard Formatter automatically includes tracebacks when
    # logger.exception() is used (which sets exc_info=True)
    error_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - '
        '%(filename)s:%(lineno)d - %(message)s',
        style='%'
    )
    error_handler.setFormatter(error_formatter)
    logger.addHandler(error_handler)

    logger.debug(f'Processing log: {processing_log_file}')
    logger.debug(f'Error log: {error_log_file}')

    return logger


def generate_output_path(input_path: str, output_arg: str = None) -> str:
    """
    Generate the output path for extracted markdown following a standardized
    convention.

    Priority:
    1) If output_arg is provided, return it unchanged.
    2) Otherwise, create/use an 'out' directory alongside the input PDF and
       name the file as: INPUT_FILENAME__extracted__YYYYMMDD_HHMMSS.md
    """
    if output_arg:
        return output_arg

    timestamp = get_timestamp()
    input_dir = os.path.dirname(input_path) or '.'
    input_basename = os.path.basename(input_path)
    stem, _ = os.path.splitext(input_basename)
    # Replace whitespace with underscores to avoid spaces
    stem_safe = re.sub(r'\s+', '_', stem)

    out_dir = os.path.join(input_dir, 'out')
    os.makedirs(out_dir, exist_ok=True)

    filename = f'{stem_safe}__extracted__{timestamp}.md'
    return os.path.join(out_dir, filename)


def main():
    """Main entry point for PDF extraction CLI."""
    parser = argparse.ArgumentParser(
        description='Extract text from PDF files with markdown formatting'
    )
    parser.add_argument(
        'pdf_path',
        type=str,
        help='Path to the input PDF file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save extracted text. If omitted, a path will be '
             'generated as out/<pdf_name>__extracted__YYYYMMDD_HHMMSS.md '
             'next to the PDF.'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Custom log filename (without extension). '
             'Will be timestamped and placed in logs/processing/ '
             '(default: extraction.log)'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Base directory for log files (default: logs)'
    )
    parser.add_argument(
        '--no-timestamp-output',
        action='store_true',
        help='Do not add timestamp to output filename'
    )

    args = parser.parse_args()

    # Set up logging with organized directory structure
    logger = setup_logging(args.log_dir, args.log_file)

    try:
        logger.info(f'Starting PDF extraction from: {args.pdf_path}')
        logger.debug(f'Log directory: {args.log_dir}')
        if args.output:
            logger.debug(f'Output file (user provided): {args.output}')
        else:
            logger.debug('Output file: auto-generated in out/ next to PDF')

        # Extract text from PDF
        with PDFExtractor(args.pdf_path) as extractor:
            extracted_text = extractor.extract()

        logger.info('PDF extraction completed successfully')

        # Verification: get character count
        char_count = len(extracted_text)
        logger.info(f'Extracted text character count: {char_count:,}')

        # Write output to file or stdout
        try:
            output_path = generate_output_path(args.pdf_path, args.output)

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                logger.debug(f'Created output directory: {output_dir}')

            # Write file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            logger.info(
                f'Extraction complete. '
                f'Character count: {char_count:,}'
            )
            logger.info(f'Output saved to: {output_path}')
        except Exception as write_error:
            logger.exception(
                f'Error writing output file: {str(write_error)}'
            )
            sys.exit(1)

    except Exception as e:
        logger.exception(
            f'Error during PDF extraction: {str(e)}'
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
