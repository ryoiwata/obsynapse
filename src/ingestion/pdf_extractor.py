"""
PDF extraction module using PyMuPDF (fitz).

This module provides a PDFExtractor class that extracts text from PDF files
with special handling for monospaced fonts, page numbers, and reading order.
"""

import fitz  # PyMuPDF
from typing import List, Dict
import re


class PDFExtractor:
    """
    Extract text from PDF files using PyMuPDF with markdown formatting.

    Features:
    - Detects monospaced fonts and wraps them in code blocks
    - Filters out page numbers (short numeric strings at page edges)
    - Sorts text blocks by Y-coordinate to preserve reading order
    - Exports as Markdown-lite with double newlines between paragraphs
    """

    # Monospaced font patterns (case-insensitive)
    MONOSPACED_FONTS = {'courier', 'mono', 'monospace', 'consolas', 'fixed'}

    # Thresholds for page number detection
    PAGE_NUMBER_MAX_LENGTH = 4  # Maximum digits in a page number
    PAGE_EDGE_THRESHOLD = 0.05  # 5% of page height from top/bottom

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

    def _is_page_number(
        self, text: str, y_pos: float, page_height: float
    ) -> bool:
        """
        Determine if a text block is likely a page number.

        Page numbers are typically:
        - Short numeric strings (1-4 digits)
        - Located at the top or bottom of the page

        Args:
            text: Text content to check
            y_pos: Y-coordinate of the text block
            page_height: Height of the page

        Returns:
            True if the text is likely a page number
        """
        # Check if text is purely numeric and short
        text_stripped = text.strip()
        if not text_stripped:
            return False

        # Check if it's a short numeric string
        pattern = (
            r'^\d{1,' + str(self.PAGE_NUMBER_MAX_LENGTH) + r'}$'
        )
        if not re.match(pattern, text_stripped):
            return False

        # Check if it's near the top or bottom of the page
        relative_y = y_pos / page_height
        is_near_top = relative_y < self.PAGE_EDGE_THRESHOLD
        is_near_bottom = relative_y > (1 - self.PAGE_EDGE_THRESHOLD)

        return is_near_top or is_near_bottom

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

        for block in text_dict.get('blocks', []):
            if 'lines' not in block:
                continue

            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    text = span.get('text', '').strip()
                    if not text:
                        continue

                    # Get font information
                    font_name = span.get('font', '')
                    font_size = span.get('size', 0)

                    # Get position (use bbox for Y-coordinate)
                    bbox = span.get('bbox', [0, 0, 0, 0])
                    y_pos = bbox[1]  # Top Y coordinate

                    blocks.append({
                        'text': text,
                        'font': font_name,
                        'font_size': font_size,
                        'y_pos': y_pos,
                        'bbox': bbox,
                        'is_monospaced': self._is_monospaced_font(font_name)
                    })

        return blocks

    def _sort_blocks_by_position(self, blocks: List[Dict]) -> List[Dict]:
        """
        Sort text blocks by Y-coordinate (top to bottom).

        For blocks on the same line, sort by X-coordinate (left to right).

        Args:
            blocks: List of text block dictionaries

        Returns:
            Sorted list of blocks
        """
        # Sort by Y, then X
        return sorted(blocks, key=lambda b: (b['y_pos'], b['bbox'][0]))

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

    def extract(self) -> str:
        """
        Extract text from the PDF and return as Markdown-lite string.

        Returns:
            Markdown-lite formatted string with double newlines between
            paragraphs
        """
        all_blocks = []

        # Extract blocks from all pages
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            page_height = page.rect.height
            blocks = self._extract_text_blocks(page)

            # Filter out page numbers
            filtered_blocks = [
                block for block in blocks
                if not self._is_page_number(
                    block['text'], block['y_pos'], page_height
                )
            ]

            all_blocks.extend(filtered_blocks)

        # Sort all blocks by Y-coordinate (across all pages)
        sorted_blocks = self._sort_blocks_by_position(all_blocks)

        # Group into paragraphs
        paragraphs = self._group_blocks_into_paragraphs(sorted_blocks)

        # Convert to markdown
        markdown_parts = []
        current_code_block = []
        in_code_block = False

        for paragraph in paragraphs:
            # Check if paragraph contains monospaced text
            paragraph_is_monospaced = any(
                block['is_monospaced'] for block in paragraph
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

                # Add regular paragraph
                paragraph_text = ' '.join(
                    block['text'] for block in paragraph
                )
                if paragraph_text.strip():
                    markdown_parts.append(paragraph_text)

        # Close any remaining code block
        if in_code_block:
            code_text = '\n'.join(current_code_block)
            markdown_parts.append(f'```\n{code_text}\n```')

        # Join with double newlines between paragraphs
        return '\n\n'.join(markdown_parts)

    def close(self):
        """Close the PDF document."""
        if self.doc:
            self.doc.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
