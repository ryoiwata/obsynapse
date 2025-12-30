"""
PDF extraction module using PyMuPDF (fitz).

This module provides a PDFExtractor class that extracts text from PDF files
with special handling for monospaced fonts, page numbers, and reading order.
"""

import fitz  # PyMuPDF
from typing import List, Dict, Tuple
import re
import argparse
import logging
import sys
import os
from datetime import datetime


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
