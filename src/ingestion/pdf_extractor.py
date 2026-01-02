"""
PDF extraction module using PyMuPDF (fitz).

This module provides a PDFExtractor class that extracts text from PDF files
with special handling for monospaced fonts, page numbers, and reading order.
"""

import fitz  # PyMuPDF
from typing import List, Dict, Tuple
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
        self.known_titles = set()
        self.seen_titles = set()

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
        blocks = []

        # Get page dimensions for header/footer detection
        page_rect = page.rect
        page_height = page_rect.height
        header_threshold = page_height * 0.10  # Top 10%
        footer_threshold = page_height * 0.90  # Bottom 10%

        for block in text_dict.get('blocks', []):
            if 'lines' not in block:
                continue

            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    text = span.get('text', '').strip()
                    if not text:
                        continue

                    # Get position (use bbox for Y-coordinate)
                    bbox = span.get('bbox', [0, 0, 0, 0])
                    y_pos = bbox[1]  # Top Y coordinate
                    y_bottom = bbox[3]  # Bottom Y coordinate

                    # Check if text is bold
                    # PyMuPDF flags: bit 4 (16) indicates bold
                    flags = span.get('flags', 0)
                    is_bold = (
                        (flags & 16) != 0 or
                        'bold' in span.get('font', '').lower()
                    )

                    # Determine if block is in header or footer region
                    is_in_header_region = y_pos <= header_threshold
                    is_in_footer_region = y_bottom >= footer_threshold
                    is_header_footer = (
                        is_in_header_region or is_in_footer_region
                    )

                    # Store span info
                    blocks.append({
                        'text': text,
                        'font': span.get('font', ''),
                        'font_size': span.get('size', 0),
                        'y_pos': y_pos,
                        'bbox': bbox,
                        'is_monospaced': self._is_monospaced_font(
                            span.get('font', '')
                        ),
                        'is_bold': is_bold,
                        'is_header_footer': is_header_footer,
                        'page_height': page_height
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
        Check if text ends with sentence-terminal punctuation (. ! ?).

        Args:
            text: Text to check

        Returns:
            True if text ends with sentence-terminal punctuation
        """
        text_stripped = text.rstrip()
        return text_stripped.endswith(('.', '!', '?'))

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

    def _normalize_for_comparison(self, text: str) -> str:
        """
        Normalize text for fuzzy comparison by removing non-alphanumeric
        characters.

        Converts text to lowercase and removes all non-alphanumeric characters
        (including colons, dashes, pipes, and extra spaces) for fuzzy matching
        of headers that may have punctuation differences.

        Args:
            text: Text to normalize

        Returns:
            Normalized string with only lowercase alphanumeric characters
        """
        # Remove all non-alphanumeric characters and convert to lowercase
        return re.sub(r'[^a-z0-9]', '', text.lower())

    def _find_title_end_position(self, text: str, title: str) -> int:
        """
        Find the end position of a normalized title in the original text.

        Uses normalized comparison to match the title, then finds where it
        actually ends in the original text (accounting for punctuation).

        Args:
            text: Text that may contain the title at the start
            title: Title to find (will be normalized for comparison)

        Returns:
            Character position where the title ends, or 0 if not found
        """
        if not text or not title:
            return 0

        text_normalized = self._normalize_for_comparison(text)
        title_normalized = self._normalize_for_comparison(title)

        if not text_normalized.startswith(title_normalized):
            return 0

        # Find where the title ends by matching words
        words = text.split()
        title_words = title.split()

        # Try to find the longest matching prefix
        for n in range(min(len(words), len(title_words) + 10), 0, -1):
            candidate = ' '.join(words[:n])
            if self._normalize_for_comparison(candidate) == title_normalized:
                # Found exact match - return the end position
                candidate_text = ' '.join(words[:n])
                # Find this in the original text (accounting for whitespace)
                pos = text.find(candidate_text)
                if pos != -1:
                    return pos + len(candidate_text)
                # Fallback: calculate approximate position
                return len(candidate_text)

        return 0

    def _is_running_header(
        self, text: str, paragraph: List[Dict] = None
    ) -> bool:
        """
        Check if a text is a running header/footer.

        Running headers can be:
        1. Pipe patterns: "Chapter 1: Introduction | 15" or "2 | Chapter 1"
        2. Known titles that appear standalone (not as actual headers)
        3. Page numbers alone
        4. Text in header/footer regions that matches known titles

        Args:
            text: The text to check
            paragraph: Optional paragraph blocks for position-based detection

        Returns:
            True if the text is a running header, False otherwise
        """
        text_stripped = text.strip()

        # Check if it's just a page number (standalone digits)
        if text_stripped.isdigit():
            return True

        # Check for pipe patterns with advanced normalization
        if '|' in text:
            parts = [p.strip() for p in text.split('|')]
            if len(parts) >= 2:
                normalized_parts = [
                    self._normalize_for_comparison(p) for p in parts
                ]

                for i, norm_p in enumerate(normalized_parts):
                    # Check if this part matches a known title (fuzzy)
                    is_title = any(
                        norm_p == self._normalize_for_comparison(t)
                        for t in self.known_titles
                    )
                    # Check if the OTHER part is just a number
                    other_part_is_num = parts[1 - i].strip().isdigit()

                    if is_title and other_part_is_num:
                        return True

                # Also check if both parts are just digits (page numbers)
                if all(p.strip().isdigit() for p in parts):
                    return True

        # Check if text is in header/footer region and matches known title
        # (using normalized comparison)
        if paragraph:
            is_header_footer = any(
                block.get('is_header_footer', False) for block in paragraph
            )
            if is_header_footer:
                # Check if it matches a known title (normalized)
                text_normalized = self._normalize_for_comparison(text_stripped)
                for title in self.known_titles:
                    if text_normalized == self._normalize_for_comparison(
                        title
                    ):
                        return True

        # Check if text matches a known title (normalized comparison)
        # (but not if it's an actual header)
        # This catches running headers that appear in the middle of content
        text_normalized = self._normalize_for_comparison(text_stripped)
        for title in self.known_titles:
            if text_normalized == self._normalize_for_comparison(title):
                # If it's in header/footer region, it's definitely
                # running header
                if paragraph and any(
                    block.get('is_header_footer', False)
                    for block in paragraph
                ):
                    return True
                # If it's a short standalone text that matches a known title,
                # it's likely a running header
                # (actual headers are detected separately)
                # We'll be conservative and only mark it if it's very short
                if len(text_stripped.split()) <= 20:
                    return True

        return False

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

        # Extract text for running header check
        paragraph_text = ' '.join(
            block['text'] for block in paragraph
        ).strip()

        # Simple check: if it contains pipe and digit, it's likely a running
        # header (we don't have known_titles here, so use simple heuristic)
        if '|' in paragraph_text:
            parts = [p.strip() for p in paragraph_text.split('|')]
            if any(part.isdigit() for part in parts):
                return False, 0

        # Extract full text of the paragraph
        paragraph_text = ' '.join(block['text'] for block in paragraph).strip()

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

        # If this is a header, add original text to known_titles
        # for fuzzy matching of running headers (we normalize when comparing)
        if is_header and paragraph_text:
            self.known_titles.add(paragraph_text.strip())

        return is_header, header_level

    def _strip_running_header(self, text: str) -> str:
        """
        Strip running header from the beginning of text if present.

        Checks if text starts with a known title followed by optional
        pipe and page number, and removes that portion.

        Args:
            text: Text that may start with a running header

        Returns:
            Text with running header stripped from the beginning, or
            original text if no running header found
        """
        cleaned_text = text.strip()

        # Check for known titles at the start of the merged block
        for title in self.known_titles:
            if cleaned_text.lower().startswith(title.lower()):
                # Remove the title
                remaining = cleaned_text[len(title):].strip()

                # Also remove a leading pipe or page number if they exist
                # e.g., "| 14 " or " 14 " or "|14"
                # Regex to catch: optional pipe, then numbers, then optional
                # trailing space/pipe
                remaining = re.sub(r'^[\s|]*\d+[\s|]*', '', remaining).strip()

                return remaining

        return cleaned_text

    def _remove_running_header_from_text(self, text: str) -> str:
        """
        Remove running header portion from text if it contains one.

        Removes:
        1. Pipe patterns with digits and known titles
        2. Standalone occurrences of known titles anywhere in the text
        3. Page numbers (standalone digits)
        4. Combinations like "Chapter 1 | 15" or "15 | Chapter 1"

        Args:
            text: Text that may contain a running header

        Returns:
            Text with running header removed, or original text if no
            running header found
        """
        if not text or not text.strip():
            return text

        text = text.strip()

        # First, handle pipe patterns using normalized comparison
        if '|' in text:
            parts = [p.strip() for p in text.split('|')]
            if len(parts) >= 2:
                cleaned_parts = []
                for part in parts:
                    part_stripped = part.strip()
                    # Skip if it's just a digit (page number)
                    if part_stripped.isdigit():
                        continue
                    # Skip if it matches a known title (using normalized comparison)
                    part_normalized = self._normalize_for_comparison(part_stripped)
                    is_title = any(
                        part_normalized == self._normalize_for_comparison(t)
                        for t in self.known_titles
                    )
                    if is_title:
                        continue
                    # Otherwise, keep it
                    cleaned_parts.append(part)

                # If we removed parts, reconstruct
                if len(cleaned_parts) < len(parts):
                    result = ' '.join(cleaned_parts).strip()
                    if result:
                        text = result
                    else:
                        return ''  # Entire text was a running header

        # Now check for standalone known titles anywhere in the text
        # In merged content, any occurrence of a known title is likely
        # a running header (the actual header would have been processed
        # separately)
        text_lower = text.lower()

        # Check each known title
        for title in self.known_titles:
            title_lower = title.lower()

            # Find all occurrences of this title in the text
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(title_lower) + r'\b'
            matches = list(re.finditer(pattern, text_lower))

            # Process matches from end to start to preserve indices
            for match in reversed(matches):
                start, end = match.span()
                before = text[:start].rstrip()
                after = text[end:].lstrip()

                # Check if this occurrence is a running header
                # It's a running header if:
                # 1. It's at a sentence boundary
                #    (after . ! ? or before sentence start)
                # 2. It's at the start/end of the text
                # 3. It's surrounded by whitespace and appears isolated

                is_at_sentence_end = (
                    not before or
                    before.endswith(('.', '!', '?', ':', ';'))
                )
                is_at_sentence_start = (
                    not after or
                    after[0] in '.!?'
                )
                is_at_text_boundary = (start == 0 or end == len(text))

                # Check if it's isolated
                # (surrounded by sentence boundaries or whitespace)
                is_isolated = (
                    is_at_sentence_end and
                    (is_at_sentence_start or is_at_text_boundary)
                )

                # Also check if there's significant content before and after
                # (if there is, it might be part of a sentence, not a
                # running header)
                has_content_before = (
                    len(before.split()) > 0 if before else False
                )
                has_content_after = (
                    len(after.split()) > 0 if after else False
                )

                # If it's isolated and either at a boundary or between
                # sentences, it's likely a running header
                if (is_isolated or
                        (is_at_text_boundary and len(text.split()) <= 30)):
                    # Additional check: if it's in the middle with content
                    # on both sides, only remove if it's clearly separated
                    # (sentence boundaries)
                    if has_content_before and has_content_after:
                        if not (is_at_sentence_end and is_at_sentence_start):
                            continue  # Probably part of content, skip

                    # Remove the title and clean up spacing
                    text = before + (' ' if before and after else '') + after
                    text = text.strip()

        # Remove standalone page numbers (just digits) but be conservative
        # Only remove if they're clearly page numbers
        # (1-3 digits, standalone)
        # Pattern: standalone digits at boundaries or with pipes
        # Match: start of text + digits + end, or
        # space/pipe + digits + space/pipe/end
        text = re.sub(
            r'(?:^|[\s|])(\d{1,3})(?:[\s|]|$)',
            lambda m: ' ' if m.group(1) else m.group(0),
            text
        )
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text if text else ''

    def _remove_running_headers(self, text: str) -> str:
        """
        Remove running headers from text by checking against known titles.

        This method checks if the text starts with or contains any string
        from self.known_titles. If a match is found, it removes the title,
        any associated pipe |, and any adjacent page numbers.

        Args:
            text: Text that may contain running headers

        Returns:
            Text with running headers removed
        """
        cleaned = text.strip()
        if not cleaned or not self.known_titles:
            return cleaned

        # Sort titles by length descending to catch longest matches first
        for title in sorted(self.known_titles, key=len, reverse=True):
            # Pattern to catch: Title, optional pipe, optional page number,
            # and surrounding whitespace
            # Example matches: "Chapter 1 | 14", "14 | Chapter 1",
            # "Chapter 1 Figure..."
            pattern = re.compile(
                rf"({re.escape(title)})|"
                rf"(\b\d+\b\s*\|\s*{re.escape(title)})|"
                rf"({re.escape(title)}\s*\|\s*\b\d+\b)",
                re.IGNORECASE
            )
            if pattern.search(cleaned):
                cleaned = pattern.sub("", cleaned).strip()

        # Clean up leftover leading pipes or orphan numbers at start/end
        cleaned = re.sub(r"^[|\s\d]+|[|\s\d]+$", "", cleaned).strip()
        return cleaned

    def _strip_merged_header(self, text: str) -> str:
        """
        Strip a known header from the beginning of text if present.

        If the text starts with a known title, strips the header prefix
        (including any associated pipe | and page numbers) but keeps
        any substantive content that follows.

        Handles patterns like:
        - "14 | Chapter 1: Introduction..." -> "Figure 1-6..."
        - "Chapter 1: Introduction | 14 Figure..." -> "Figure..."
        - "Chapter 1: Introduction Figure..." -> "Figure..."

        Uses normalized comparison to match titles, ignoring punctuation.

        Args:
            text: Text that may start with a running header

        Returns:
            Text with header prefix stripped, or original text if no header
            found at the start
        """
        if not text or not self.known_titles:
            return text

        text_stripped = text.strip()
        if not text_stripped:
            return text

        # First, handle pipe patterns at the start:
        # "14 | Title..." or "Title | 14 ..."
        if '|' in text_stripped:
            # Split by first pipe only
            pipe_pos = text_stripped.find('|')
            if pipe_pos != -1:
                before_pipe = text_stripped[:pipe_pos].strip()
                after_pipe = text_stripped[pipe_pos + 1:].strip()

                # Pattern 1: "14 | Title..."
                if before_pipe.isdigit():
                    # Check if after_pipe starts with known title (normalized)
                    for title in sorted(
                        self.known_titles, key=len, reverse=True
                    ):
                        after_norm = self._normalize_for_comparison(after_pipe)
                        title_norm = self._normalize_for_comparison(title)

                        if after_norm.startswith(title_norm):
                            # Find where title ends in after_pipe
                            words = after_pipe.split()
                            title_words = title.split()

                            # Try to find matching prefix
                            max_n = min(len(words), len(title_words) + 10)
                            for n in range(max_n, 0, -1):
                                candidate = ' '.join(words[:n])
                                cand_norm = self._normalize_for_comparison(
                                    candidate
                                )
                                if cand_norm == title_norm:
                                    # Found match - return remaining words
                                    remaining = ' '.join(words[n:]).strip()
                                    # Remove any leading page numbers
                                    remaining = re.sub(
                                        r'^[\s|]*\d+[\s|]*', '', remaining
                                    ).strip()
                                    if remaining:
                                        return remaining
                                    break

                # Pattern 2: "Title | 14 ..."
                # Check if before_pipe starts with known title
                # and after_pipe starts with a number
                if after_pipe:
                    after_words = after_pipe.split()
                    if after_words and after_words[0].isdigit():
                        for title in sorted(
                            self.known_titles, key=len, reverse=True
                        ):
                            before_norm = self._normalize_for_comparison(
                                before_pipe
                            )
                            title_norm = self._normalize_for_comparison(title)

                            if before_norm.startswith(title_norm):
                                # Find where title ends in before_pipe
                                words = before_pipe.split()
                                title_words = title.split()

                                max_n = min(len(words), len(title_words) + 10)
                                for n in range(max_n, 0, -1):
                                    candidate = ' '.join(words[:n])
                                    cand_norm = self._normalize_for_comparison(
                                        candidate
                                    )
                                    if cand_norm == title_norm:
                                        # Title matches - remove pipe pattern
                                        # Return what comes after pipe/number
                                        after_clean = re.sub(
                                            r'^\d+[\s|]*', '', after_pipe
                                        ).strip()
                                        if after_clean:
                                            return after_clean
                                        break

        # Handle non-pipe patterns: text starting with a known title
        # Sort titles by length descending to catch longest matches first
        text_normalized = self._normalize_for_comparison(text_stripped)
        for title in sorted(self.known_titles, key=len, reverse=True):
            title_normalized = self._normalize_for_comparison(title)

            if text_normalized.startswith(title_normalized):
                # Find where the title actually ends in the original text
                words = text_stripped.split()
                title_words = title.split()

                # Try to find the longest matching prefix
                max_n = min(len(words), len(title_words) + 10)
                for n in range(max_n, 0, -1):
                    candidate = ' '.join(words[:n])
                    if (self._normalize_for_comparison(candidate) ==
                            title_normalized):
                        # Found exact match - remove these words
                        remaining = ' '.join(words[n:]).strip()
                        # Also remove any leading pipe or page numbers
                        remaining = re.sub(
                            r'^[\s|]*\d+[\s|]*', '', remaining
                        ).strip()
                        if remaining and len(remaining.split()) > 0:
                            return remaining
                        break

        return text

    def _clean_paragraph_text(self, text: str) -> str:
        """
        Clean paragraph text by removing duplicate headers.

        Removes:
        1. Exact matches of seen titles (standalone duplicates) using normalized comparison
        2. Merged headers at the start of paragraphs (including pipe patterns)

        Args:
            text: Text that may contain duplicate headers

        Returns:
            Cleaned text, or empty string if entire text was a duplicate
        """
        if not text or not text.strip():
            return text.strip()

        text_stripped = text.strip()

        # 1. Check for exact matches (standalone duplicates) using normalized comparison
        text_normalized = self._normalize_for_comparison(text_stripped)
        for seen_title in self.seen_titles:
            seen_normalized = self._normalize_for_comparison(seen_title)
            if text_normalized == seen_normalized:
                return ""

        # 2. Use _strip_merged_header to remove merged headers at start
        # This handles both pipe patterns and direct title matches
        cleaned = self._strip_merged_header(text_stripped)

        # If _strip_merged_header found and removed a header, return cleaned
        if cleaned != text_stripped:
            return cleaned.strip()

        # 3. Fallback: Check for merged headers using normalized comparison
        # Sort titles by length descending to avoid partial matches
        sorted_titles = sorted(self.seen_titles, key=len, reverse=True)
        for title in sorted_titles:
            # Use normalized comparison to match titles
            title_normalized = self._normalize_for_comparison(title)
            if text_normalized.startswith(title_normalized):
                # Find where title ends and remove it
                words = text_stripped.split()
                title_words = title.split()
                max_n = min(len(words), len(title_words) + 10)
                for n in range(max_n, 0, -1):
                    candidate = ' '.join(words[:n])
                    if (self._normalize_for_comparison(candidate) ==
                            title_normalized):
                        remaining = ' '.join(words[n:]).strip()
                        # Remove any leading pipe or page numbers
                        remaining = re.sub(
                            r'^[\s|]*\d+[\s|]*', '', remaining
                        ).strip()
                        if remaining:
                            return remaining
                        break

        return text_stripped

    def _is_numeric_prefix_line(self, text: str) -> bool:
        """
        Check if text matches the strict numeric prefix pattern.

        Matches lines that start with one or more digits, followed by whitespace,
        followed by any non-empty text (e.g., "2 For non-English languages...").

        This method does NOT match numbered list items (e.g., "1. Step one")
        by ensuring there is NO period directly after the number.

        Args:
            text: The text to check

        Returns:
            True if the text matches the numeric prefix pattern, False otherwise

        Examples:
            >>> _is_numeric_prefix_line("2 For non-English languages...")
            True
            >>> _is_numeric_prefix_line("3 Autoregressive language models...")
            True
            >>> _is_numeric_prefix_line("1. Step one")
            False
            >>> _is_numeric_prefix_line("14 Chapter title continues...")
            True
        """
        if not text or not text.strip():
            return False

        text_stripped = text.strip()

        # Pattern: starts with digit(s), whitespace, then any non-whitespace char
        # This matches: "2 For non-English...", "3 Autoregressive...", etc.
        numeric_prefix_pattern = re.compile(r"^\d+\s+\S")
        if not numeric_prefix_pattern.match(text_stripped):
            return False

        # Crucial: exclude numbered list items by checking for period after number
        # Pattern: digit(s) followed by period and space (e.g., "1. Step one")
        numbered_list_pattern = re.compile(r"^\d+\.\s+")
        if numbered_list_pattern.match(text_stripped):
            return False

        return True

    def _split_paragraph_into_lines(self, paragraph: List[Dict]) -> List[List[Dict]]:
        """
        Split a paragraph into visual lines based on Y-coordinate tolerance.

        Groups blocks within a paragraph into lines using the same LINE_TOLERANCE
        concept from _sort_blocks_by_position. This allows line-level filtering
        of numeric prefix footnotes without removing entire paragraphs.

        Args:
            paragraph: List of text block dictionaries representing a paragraph

        Returns:
            List of lines, where each line is a list of blocks
        """
        if not paragraph:
            return []

        # Tolerance for grouping blocks into the same line (in pixels)
        # Reuse the same value from _sort_blocks_by_position
        LINE_TOLERANCE = 3.0

        # Sort blocks by Y-coordinate
        blocks_sorted_by_y = sorted(paragraph, key=lambda b: b.get('y_pos', 0))

        # Group blocks into lines based on Y-coordinate tolerance
        lines = []
        current_line = []
        current_line_y = None

        for block in blocks_sorted_by_y:
            block_y = block.get('y_pos', 0)

            # If this is the first block or Y is within tolerance, add to current line
            if (current_line_y is None or
                    abs(block_y - current_line_y) <= LINE_TOLERANCE):
                current_line.append(block)
                if current_line_y is None:
                    current_line_y = block_y
                else:
                    current_line_y = min(current_line_y, block_y)
            else:
                # Y difference exceeds tolerance, start a new line
                if current_line:
                    # Sort current line by X-coordinate (left to right)
                    current_line.sort(key=lambda b: b.get('bbox', [0, 0, 0, 0])[0])
                    lines.append(current_line)
                current_line = [block]
                current_line_y = block_y

        # Don't forget the last line
        if current_line:
            current_line.sort(key=lambda b: b.get('bbox', [0, 0, 0, 0])[0])
            lines.append(current_line)

        return lines

    def _filter_numeric_prefix_lines(
        self, lines: List[List[Dict]]
    ) -> List[List[Dict]]:
        """
        Filter out lines that match the numeric prefix pattern.

        For each line, checks if the combined text matches the numeric prefix
        pattern. If it does, the line is removed. Otherwise, it is kept.

        Args:
            lines: List of lines, where each line is a list of blocks

        Returns:
            List of lines with numeric prefix lines removed

        Example:
            Input paragraph lines:
            - "1 In this book, I use traditional ML..."
            - "The statistical nature of languages was discovered centuries ago..."

            After filtering:
            - "The statistical nature of languages was discovered centuries ago..."
            (first line removed, second line preserved)
        """
        filtered_lines = []
        for line in lines:
            # Compute line text from all blocks in the line
            line_text = ' '.join(
                block.get('text', '') for block in line
            ).strip()

            # Skip lines that match the numeric prefix pattern
            if self._is_numeric_prefix_line(line_text):
                continue

            # Keep all other lines
            filtered_lines.append(line)

        return filtered_lines

    def _clean_footnotes(self, text: str) -> str:
        """
        Strict final pass to remove numeric prefix lines from markdown text.

        Removes any entire paragraph line that matches the numeric prefix pattern:
        - Starts with one or more digits
        - Followed by whitespace
        - Followed by any non-empty text
        - Does NOT match numbered lists (e.g., "1. Step one")

        This is a defensive cleanup pass that catches any numeric prefix lines
        that may have slipped through earlier processing.

        Args:
            text: Markdown text that may contain numeric prefix lines

        Returns:
            Text with numeric prefix lines removed and normalized blank lines
        """
        if not text:
            return text

        # Pattern to match numeric prefix lines: ^\d+\s+\S.*$
        # ^: Start of line
        # \d+: One or more digits
        # \s+: One or more whitespace characters
        # \S: Any non-whitespace character (ensures there's content after space)
        # .*$: Rest of the line
        numeric_prefix_pattern = re.compile(
            r"^\d+\s+\S.*$", re.MULTILINE
        )

        footnote_count = 0

        def should_remove_line(match: re.Match) -> str:
            """
            Determine if a matched line should be removed.

            Args:
                match: Regex match object

            Returns:
                Empty string if should be removed, original match if not
            """
            nonlocal footnote_count
            line = match.group(0).strip()

            # Exclude numbered list items (e.g., "1. Step one")
            # Pattern: digit(s) followed by period and space
            numbered_list_pattern = re.compile(r"^\d+\.\s+")
            if numbered_list_pattern.match(line):
                return match.group(0)  # Keep numbered lists

            # Remove this line (matches numeric prefix pattern)
            footnote_count += 1
            return ""

        # Replace matched lines with empty string
        cleaned_text = numeric_prefix_pattern.sub(should_remove_line, text)

        # Normalize excessive blank lines (3+ newlines -> 2 newlines)
        cleaned_text = re.sub(r'\n\n\n+', '\n\n', cleaned_text)

        return cleaned_text

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

        # Track numeric footnotes removed for logging
        numeric_footnote_count = 0

        prev_paragraph = None
        for paragraph in paragraphs:
            # Split paragraph into visual lines for line-level filtering
            # This allows us to remove numeric prefix footnote lines without
            # deleting entire paragraphs that contain real content
            lines = self._split_paragraph_into_lines(paragraph)

            # Filter out numeric prefix lines at the line level
            filtered_lines = self._filter_numeric_prefix_lines(lines)
            removed_count = len(lines) - len(filtered_lines)
            if removed_count > 0:
                numeric_footnote_count += removed_count

            # If all lines were removed, skip this paragraph
            if not filtered_lines:
                continue

            # Reconstruct paragraph from remaining lines
            # Join blocks within each line, then join lines with spaces
            paragraph_blocks = []
            for line in filtered_lines:
                paragraph_blocks.extend(line)

            # If no blocks remain after filtering, skip
            if not paragraph_blocks:
                continue

            # Extract paragraph text from filtered blocks
            paragraph_text = ' '.join(
                block.get('text', '') for block in paragraph_blocks
            ).strip()

            if not paragraph_text:
                continue

            # Update paragraph to use filtered blocks for subsequent processing
            paragraph = paragraph_blocks

            # Check if paragraph contains monospaced text
            paragraph_is_monospaced = any(
                block['is_monospaced'] for block in paragraph
            )

            # Check if paragraph is a header (but not if it's monospaced)
            # This check happens BEFORE removing running headers to ensure
            # we can identify the first occurrence
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
                    block['text'] for block in paragraph
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
                    # Case A: Standalone Header
                    if is_header:
                        # This is a structural header
                        # Use normalized comparison for seen_titles check
                        normalized_title = self._normalize_for_comparison(
                            paragraph_text
                        )
                        # Check if we've seen this normalized title before
                        seen_before = any(
                            normalized_title ==
                            self._normalize_for_comparison(seen)
                            for seen in self.seen_titles
                        )
                        if seen_before:
                            # Already seen - skip duplicate standalone header
                            continue
                        else:
                            # First occurrence - add original text to
                            # seen_titles and keep it
                            self.seen_titles.add(paragraph_text.strip())
                            # Normalized version already added to known_titles
                            # in _is_header_paragraph
                            header_prefix = '#' * header_level
                            markdown_parts.append(
                                f'{header_prefix} {paragraph_text}'
                            )
                    else:
                        # Case B: Not a standalone header - check for merged
                        # headers at the start
                        # Use _strip_merged_header first to handle pipe patterns
                        # This will strip headers while preserving content after
                        paragraph_text = self._strip_merged_header(
                            paragraph_text
                        )

                        # If entire paragraph was stripped, skip it
                        if not paragraph_text or not paragraph_text.strip():
                            continue

                        # Clean merged headers at start (handles extra cases)
                        cleaned_text = self._clean_paragraph_text(paragraph_text)

                        # Skip if paragraph is empty after cleaning
                        if not cleaned_text:
                            continue

                        # Skip if it's clearly a running header
                        if self._is_running_header(cleaned_text, paragraph):
                            continue

                        # Add regular paragraph
                        markdown_parts.append(cleaned_text)

            # Update previous paragraph for spacing checks
            prev_paragraph = paragraph

        # Close any remaining code block
        if in_code_block:
            code_text = '\n'.join(current_code_block)
            markdown_parts.append(f'```\n{code_text}\n```')

        # Log numeric footnotes removed
        if numeric_footnote_count > 0:
            logger = logging.getLogger(__name__)
            logger.info(
                f"Removed {numeric_footnote_count} numeric footnote "
                f"paragraph(s) during markdown conversion"
            )

        return markdown_parts

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

        # Merge paragraphs across page boundaries where appropriate
        merged_paragraphs = self._merge_pages_paragraphs(page_paragraphs)

        # Note: We no longer do a first pass to build known_titles.
        # Instead, we use first-seen logic in _paragraphs_to_markdown
        # to preserve the first occurrence of each header.

        # Convert all merged paragraphs to markdown
        # Pass all_blocks for baseline font size calculation
        markdown_parts = self._paragraphs_to_markdown(
            merged_paragraphs, all_blocks
        )

        # Join with double newlines between paragraphs
        markdown_text = '\n\n'.join(markdown_parts)

        # Clean footnotes from the final markdown
        markdown_text = self._clean_footnotes(markdown_text)

        return markdown_text

    def _merge_pages_paragraphs(
        self, page_paragraphs: List[List[List[Dict]]]
    ) -> List[List[Dict]]:
        """
        Merge paragraphs across page boundaries where text continues.

        Evaluates the last paragraph of each page with the first paragraph
        of the next page to determine if they should be merged.

        Filters out running headers before merging to prevent them from
        being included in merged content.

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
            # paragraphs (after filtering running headers)
            if not current_page_paragraphs:
                # Filter running headers from next page
                filtered_next = [
                    para for para in next_page_paragraphs
                    if not self._is_running_header(
                        ' '.join(
                            block['text'] for block in para
                        ).strip(),
                        para
                    )
                ]
                current_page_paragraphs = filtered_next
                continue

            # If next page has no paragraphs, add current page's paragraphs
            # (after filtering running headers)
            if not next_page_paragraphs:
                filtered_current = [
                    para for para in current_page_paragraphs
                    if not self._is_running_header(
                        ' '.join(
                            block['text'] for block in para
                        ).strip(),
                        para
                    )
                ]
                merged.extend(filtered_current)
                current_page_paragraphs = []
                continue

            # Filter running headers from boundaries before merging
            # Get last paragraph of current page
            # (skip if it's a running header)
            last_para = None
            last_para_idx = len(current_page_paragraphs) - 1
            for i in range(len(current_page_paragraphs) - 1, -1, -1):
                para = current_page_paragraphs[i]
                para_text = ' '.join(
                    block['text'] for block in para
                ).strip()
                if not self._is_running_header(para_text, para):
                    last_para = para
                    last_para_idx = i
                    break

            # Get first paragraph of next page
            # (skip if it's a running header)
            first_para = None
            first_para_idx = 0
            for i in range(len(next_page_paragraphs)):
                para = next_page_paragraphs[i]
                para_text = ' '.join(
                    block['text'] for block in para
                ).strip()
                if not self._is_running_header(para_text, para):
                    first_para = para
                    first_para_idx = i
                    break

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
                    # (including last_para)
                    merged.extend(
                        current_page_paragraphs[:last_para_idx + 1]
                    )
                    # Start fresh with next page's paragraphs
                    # (from first_para onwards)
                    current_page_paragraphs = (
                        next_page_paragraphs[first_para_idx:]
                    )
            elif last_para:
                # No valid first para on next page
                # - add current page's paragraphs
                merged.extend(current_page_paragraphs[:last_para_idx + 1])
                current_page_paragraphs = []
            elif first_para:
                # No valid last para on current page
                # - add what we have and move to next
                merged.extend([
                    para for para in current_page_paragraphs
                    if not self._is_running_header(
                        ' '.join(
                            block['text'] for block in para
                        ).strip(),
                        para
                    )
                ])
                current_page_paragraphs = (
                    next_page_paragraphs[first_para_idx:]
                )
            else:
                # Both are running headers - skip both and continue
                merged.extend([
                    para for para in current_page_paragraphs[:last_para_idx]
                    if not self._is_running_header(
                        ' '.join(
                            block['text'] for block in para
                        ).strip(),
                        para
                    )
                ])
                current_page_paragraphs = (
                    next_page_paragraphs[first_para_idx + 1:]
                )

        # Add any remaining paragraphs from the last page
        # (filtering running headers)
        if current_page_paragraphs:
            filtered_remaining = [
                para for para in current_page_paragraphs
                if not self._is_running_header(
                    ' '.join(
                        block['text'] for block in para
                    ).strip(),
                    para
                )
            ]
            merged.extend(filtered_remaining)

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
        help='Path to save extracted text (default: print to stdout). '
             'If directory does not exist, it will be created. '
             'Timestamp will be prepended to filename if not specified.'
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

    # Generate timestamp for this run
    timestamp = get_timestamp()

    # Set up logging with organized directory structure
    logger = setup_logging(args.log_dir, args.log_file)

    try:
        logger.info(f'Starting PDF extraction from: {args.pdf_path}')
        logger.debug(f'Log directory: {args.log_dir}')
        if args.output:
            logger.debug(f'Output file: {args.output}')
        else:
            logger.debug('Output: stdout (piping mode)')

        # Extract text from PDF
        with PDFExtractor(args.pdf_path) as extractor:
            extracted_text = extractor.extract()

        logger.info('PDF extraction completed successfully')

        # Verification: get character count
        char_count = len(extracted_text)
        logger.info(f'Extracted text character count: {char_count:,}')

        # Write output to file or stdout
        try:
            if args.output:
                # Determine output path with timestamp if needed
                output_path = args.output
                if not args.no_timestamp_output:
                    # Add timestamp to filename
                    dirname = os.path.dirname(output_path) or '.'
                    basename = os.path.basename(output_path)
                    name, ext = os.path.splitext(basename)
                    if not ext:
                        ext = '.md'  # Default to .md if no extension
                    timestamped_basename = f'{timestamp}_{name}{ext}'
                    output_path = os.path.join(dirname, timestamped_basename)

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
            else:
                # Print to stdout for piping (only when --output is None)
                sys.stdout.write(extracted_text)
                logger.debug('Extracted text written to stdout')
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
