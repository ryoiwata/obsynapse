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
    - Filters out headers and footers (short text in page edge zones)
    - Sorts text blocks by Y-coordinate to preserve reading order
    - Exports as Markdown-lite with double newlines between paragraphs
    """

    # Monospaced font patterns (case-insensitive)
    MONOSPACED_FONTS = {'courier', 'mono', 'monospace', 'consolas', 'fixed'}

    # Thresholds for header/footer detection
    # PAGE_EDGE_THRESHOLD: 10% of page height from top/bottom
    # Increased to catch headers/footers that sit deeper in the margins
    PAGE_EDGE_THRESHOLD = 0.10
    # MAX_HEADER_FOOTER_LENGTH: Maximum characters for header/footer
    # Short text in edge zones is filtered to avoid deleting real paragraphs
    # near margins
    MAX_HEADER_FOOTER_LENGTH = 150

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

    def _is_header_or_footer(
        self, text: str, y_pos: float, page_height: float
    ) -> bool:
        """
        Determine if a text block is likely a header or footer.

        Uses positional filtering: checks if block is in the edge zone
        (top/bottom PAGE_EDGE_THRESHOLD) AND matches specific header/footer
        patterns or has short text length.

        Strictly enforces removal of lines matching these patterns when in
        edge zones:
        - "Number | Text" (e.g., "10 | Chapter 1...")
        - "Text | Number" (e.g., "The Rise of AI Engineering | 15")
        - Solo page numbers (e.g., "1", "2", "123")

        Args:
            text: Text content to check
            y_pos: Y-coordinate of the text block (top of block)
            page_height: Height of the page

        Returns:
            True if the text is likely a header or footer and should be
            filtered out. Returns True immediately if a regex pattern matches.
        """
        # Positional filtering: check if in edge zone first
        relative_y = y_pos / page_height
        is_near_top = relative_y < self.PAGE_EDGE_THRESHOLD
        is_near_bottom = relative_y > (1 - self.PAGE_EDGE_THRESHOLD)

        # Must be in edge zone (top 10% or bottom 10%) to be considered
        if not (is_near_top or is_near_bottom):
            return False

        # If in edge zone, check for specific header/footer patterns
        text_stripped = text.strip()
        if not text_stripped:
            return False

        # Regex patterns for specific header/footer formats
        # These patterns are strictly enforced - if matched, return True
        # immediately to filter out the block
        # More resilient patterns to handle fragmenting and spacing
        # Pattern 1: "Number | Text" (e.g., "10 | Chapter 1...")
        pattern_number_text = re.compile(r'^\d+\s*[|]\s*.*', re.IGNORECASE)
        # Pattern 2: "Text | Number" (e.g., "The Rise of AI Engineering | 15")
        pattern_text_number = re.compile(r'.*\s*[|]\s*\d+$', re.IGNORECASE)
        # Pattern 3: Solo page numbers (e.g., "1", "2", "123")
        pattern_solo_number = re.compile(r'^\d+$')

        # Strict enforcement: if text matches any pattern, return True
        # immediately to filter it out
        if pattern_number_text.match(text_stripped):
            return True
        if pattern_text_number.match(text_stripped):
            return True
        if pattern_solo_number.match(text_stripped):
            return True

        # Fallback: short text in edge zones is filtered
        # (e.g., < 150 characters)
        return len(text_stripped) <= self.MAX_HEADER_FOOTER_LENGTH

    def _extract_text_blocks(self, page) -> List[Dict]:
        """
        Extract text blocks from a page using get_text('dict').

        Checks entire lines (concatenated spans) for headers/footers before
        adding individual spans. This ensures fragmented spans like "14" and
        "| Chapter 1" are reunited before the regex check runs.

        Args:
            page: PyMuPDF page object

        Returns:
            List of text block dictionaries with position and formatting info
        """
        text_dict = page.get_text('dict')
        blocks = []
        page_height = page.rect.height

        for block in text_dict.get('blocks', []):
            if 'lines' not in block:
                continue

            for line in block.get('lines', []):
                # Collect all spans in this line
                line_spans = []
                line_text_parts = []
                line_y_pos = None

                for span in line.get('spans', []):
                    text = span.get('text', '').strip()
                    if not text:
                        continue

                    # Get position (use bbox for Y-coordinate)
                    bbox = span.get('bbox', [0, 0, 0, 0])
                    y_pos = bbox[1]  # Top Y coordinate

                    # Store span info for later use
                    line_spans.append({
                        'text': text,
                        'font': span.get('font', ''),
                        'font_size': span.get('size', 0),
                        'y_pos': y_pos,
                        'bbox': bbox,
                        'is_monospaced': self._is_monospaced_font(
                            span.get('font', '')
                        )
                    })

                    # Collect text parts for concatenation
                    line_text_parts.append(text)
                    # Use first span's Y position for line position
                    if line_y_pos is None:
                        line_y_pos = y_pos

                # Concatenate all spans in the line to form full line text
                if line_text_parts and line_y_pos is not None:
                    full_line_text = ' '.join(line_text_parts)

                    # Check if the entire line is a header/footer
                    # Only add spans if the line is NOT a header/footer
                    if not self._is_header_or_footer(
                        full_line_text, line_y_pos, page_height
                    ):
                        # Line is not a header/footer, add all spans
                        blocks.extend(line_spans)

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

    def _paragraphs_to_markdown(
        self, paragraphs: List[List[Dict]]
    ) -> List[str]:
        """
        Convert paragraphs to markdown format with code block handling.

        Args:
            paragraphs: List of paragraphs, where each paragraph is a
                        list of blocks

        Returns:
            List of markdown strings
        """
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

        return markdown_parts

    def extract(self) -> str:
        """
        Extract text from the PDF and return as Markdown-lite string.

        Processes pages individually to preserve correct reading order.
        Headers and footers are automatically removed based on positional
        filtering (top/bottom 5% of page) and regex pattern matching.

        Each page is fully processed (extract, filter, sort, group) before
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

        # Process each page individually to preserve reading order
        # This ensures Page 1 content is fully processed before Page 2
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]

            # Extract blocks from this page
            # Headers/footers are already filtered at the line level in
            # _extract_text_blocks by checking concatenated line text
            blocks = self._extract_text_blocks(page)

            # Sort blocks by position (within this page only)
            # This prevents headers from Page 2 appearing before footers
            # from Page 1
            sorted_blocks = self._sort_blocks_by_position(blocks)

            # Group into paragraphs (within this page only)
            # Paragraphs are grouped based on vertical proximity
            paragraphs = self._group_blocks_into_paragraphs(sorted_blocks)

            # Store paragraphs for this page (not yet converted to markdown)
            if paragraphs:
                page_paragraphs.append(paragraphs)

        # Merge paragraphs across page boundaries where appropriate
        merged_paragraphs = self._merge_pages_paragraphs(page_paragraphs)

        # Convert all merged paragraphs to markdown
        markdown_parts = self._paragraphs_to_markdown(merged_paragraphs)

        # Join with double newlines between paragraphs
        return '\n\n'.join(markdown_parts)

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

            # Get last paragraph of current page and first paragraph of
            # next page
            last_para = current_page_paragraphs[-1]
            first_para = next_page_paragraphs[0]

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
                # Add all paragraphs except the last from current page
                merged.extend(current_page_paragraphs[:-1])
                # Add the merged paragraph
                merged.append(merged_para)
                # Set remaining paragraphs from next page (excluding first)
                current_page_paragraphs = next_page_paragraphs[1:]
            else:
                # No merging needed - add all paragraphs from current page
                merged.extend(current_page_paragraphs)
                # Start fresh with next page's paragraphs
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
