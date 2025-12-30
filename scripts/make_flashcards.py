#!/usr/bin/env python3
"""
Generate flashcards from parsed Obsidian notes using OpenAI structured outputs.

Reads system/user prompts, injects note JSON, calls OpenAI API, and writes results.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI


# JSON Schema for structured outputs
FLASHCARD_SCHEMA = {
    "type": "object",
    "properties": {
        "note_id": {"type": "string"},
        "title": {"type": "string"},
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "card_type": {"type": "string", "enum": ["qa", "cloze"]},
                    "front": {"type": "string"},
                    "back": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "source_path": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": [
                    "id", "card_type", "front", "back", "tags", "source_path"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": ["note_id", "title", "cards"],
    "additionalProperties": False
}


def setup_logging(log_dir: Path) -> tuple[logging.Logger, Path, Path]:
    """
    Set up logging to both console and files.
    
    Returns:
        Tuple of (logger, log_file_path, error_log_path)
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file names with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"make_flashcards_{timestamp}.log"
    error_log_file = log_dir / f"make_flashcards_errors_{timestamp}.log"
    
    # Configure root logger
    logger = logging.getLogger("make_flashcards")
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (DEBUG and above)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Error file handler (ERROR and above)
    error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_format)
    logger.addHandler(error_handler)
    
    return logger, log_file, error_log_file


def read_file_or_die(file_path: Path, description: str, logger: logging.Logger) -> str:
    """Read file content or exit with error."""
    if not file_path.exists():
        error_msg = f"{description} not found: {file_path}"
        logger.error(error_msg)
        print(f"Error: {error_msg}", file=sys.stderr)
        sys.exit(1)
    
    try:
        logger.debug(f"Reading {description}: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Successfully read {description}")
        return content
    except Exception as e:
        error_msg = f"Error reading {description}: {e}"
        logger.error(error_msg, exc_info=True)
        print(f"Error: {error_msg}", file=sys.stderr)
        sys.exit(1)


def read_json_or_die(file_path: Path, description: str, logger: logging.Logger) -> Dict[str, Any]:
    """Read and parse JSON file or exit with error."""
    content = read_file_or_die(file_path, description, logger)
    
    try:
        logger.debug(f"Parsing JSON from {description}")
        data = json.loads(content)
        logger.info(f"Successfully parsed JSON from {description}")
        return data
    except json.JSONDecodeError as e:
        error_msg = f"{description} is not valid JSON: {e}"
        logger.error(error_msg, exc_info=True)
        print(f"Error: {error_msg}", file=sys.stderr)
        sys.exit(1)


def inject_json_into_template(template: str, json_data: Dict[str, Any]) -> str:
    """Replace placeholder in template with JSON string."""
    json_string = json.dumps(json_data, ensure_ascii=False)
    return template.replace('"{parsed_json}"', json_string)


def save_raw_api_response(response, output_path: Path, logger: logging.Logger) -> None:
    """Save the raw API response to a JSON file."""
    try:
        # Convert response object to dict for serialization
        response_dict = {}
        
        # Try to extract all relevant fields
        if hasattr(response, '__dict__'):
            response_dict = response.__dict__.copy()
        else:
            # Try to serialize common response attributes
            response_dict = {
                'id': getattr(response, 'id', None),
                'object': getattr(response, 'object', None),
                'created': getattr(response, 'created', None),
                'model': getattr(response, 'model', None),
            }
            
            # Extract choices if available
            if hasattr(response, 'choices') and response.choices:
                response_dict['choices'] = []
                for choice in response.choices:
                    choice_dict = {}
                    if hasattr(choice, 'message'):
                        msg = choice.message
                        choice_dict['message'] = {
                            'role': getattr(msg, 'role', None),
                            'content': getattr(msg, 'content', None),
                            'refusal': getattr(msg, 'refusal', None),
                        }
                    if hasattr(choice, 'finish_reason'):
                        choice_dict['finish_reason'] = choice.finish_reason
                    if hasattr(choice, 'index'):
                        choice_dict['index'] = choice.index
                    response_dict['choices'].append(choice_dict)
            
            # Extract usage if available
            if hasattr(response, 'usage'):
                usage = response.usage
                response_dict['usage'] = {
                    'prompt_tokens': getattr(usage, 'prompt_tokens', None),
                    'completion_tokens': getattr(usage, 'completion_tokens', None),
                    'total_tokens': getattr(usage, 'total_tokens', None),
                }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(response_dict, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Saved raw API response to: {output_path}")
    except Exception as e:
        logger.warning(f"Failed to save raw API response: {e}", exc_info=True)


def extract_json_from_response(response, logger: logging.Logger) -> Dict[str, Any]:
    """
    Extract JSON from OpenAI API response.
    
    Handles different response formats from the SDK.
    """
    logger.debug("Extracting JSON from API response")
    
    # Try different response formats
    text = None
    
    # Check for structured output format
    if hasattr(response, 'choices') and response.choices:
        choice = response.choices[0]
        if hasattr(choice, 'message'):
            if hasattr(choice.message, 'content'):
                text = choice.message.content
                logger.debug("Extracted text from response.choices[0].message.content")
            elif hasattr(choice.message, 'refusal'):
                # Handle refusal case
                error_msg = "API returned refusal"
                logger.error(error_msg)
                print(f"Warning: {error_msg}", file=sys.stderr)
                text = None
    
    # Fallback to output_text if available
    if not text and hasattr(response, 'output_text'):
        text = response.output_text
        logger.debug("Extracted text from response.output_text")
    
    # Fallback to output field
    if not text and hasattr(response, 'output'):
        if isinstance(response.output, list) and response.output:
            if hasattr(response.output[0], 'content'):
                if isinstance(response.output[0].content, list):
                    if hasattr(response.output[0].content[0], 'text'):
                        text = response.output[0].content[0].text
                        logger.debug("Extracted text from response.output[0].content[0].text")
    
    if not text:
        error_msg = "Could not extract text from API response"
        logger.error(error_msg)
        logger.debug(f"Response type: {type(response)}")
        if hasattr(response, '__dict__'):
            logger.debug(f"Response attributes: {list(response.__dict__.keys())}")
        print(f"Error: {error_msg}", file=sys.stderr)
        sys.exit(1)
    
    # Parse JSON
    try:
        logger.debug("Parsing extracted JSON text")
        data = json.loads(text)
        logger.info("Successfully parsed JSON from API response")
        return data
    except json.JSONDecodeError as e:
        error_msg = f"API response is not valid JSON: {e}"
        logger.error(error_msg)
        logger.debug(f"Response text (first 500 chars): {text[:500]}")
        print(f"Error: {error_msg}", file=sys.stderr)
        print(f"Response text (first 500 chars): {text[:500]}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate flashcards from parsed Obsidian notes"
    )
    parser.add_argument(
        "--system",
        type=str,
        default="prompts/flashcards_system_v1_0.txt",
        help="Path to system prompt file"
    )
    parser.add_argument(
        "--user",
        type=str,
        default="prompts/flashcards_user_v1_0.txt",
        help="Path to user prompt template file"
    )
    parser.add_argument(
        "--note",
        type=str,
        required=True,
        help="Path to parsed note JSON file"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out/flashcards.json",
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenAI model to use (default: OPENAI_MODEL from .env or gpt-4o)"
    )
    
    args = parser.parse_args()
    
    # Resolve project root first
    project_root = Path(__file__).parent.parent
    
    # Set up logging
    log_dir = project_root / "logs"
    logger, log_file, error_log_file = setup_logging(log_dir)
    
    logger.info("=" * 80)
    logger.info("Starting flashcard generation")
    logger.info("=" * 80)
    logger.debug(f"Project root: {project_root}")
    logger.debug(f"Arguments: {vars(args)}")
    
    # Load environment variables from .env file
    env_path = project_root / ".env"
    if env_path.exists():
        logger.debug(f"Loading .env from: {env_path}")
        load_dotenv(env_path)
    else:
        logger.debug("No .env file found in project root, trying current directory")
        load_dotenv()
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        error_msg = "OPENAI_API_KEY not found in environment or .env file"
        logger.error(error_msg)
        print(f"Error: {error_msg}", file=sys.stderr)
        sys.exit(1)
    logger.info("API key loaded successfully")
    
    # Get model from args, .env, or default
    model = args.model or os.getenv("OPENAI_MODEL", "gpt-4o")
    logger.info(f"Using model: {model}")
    
    # Resolve paths
    system_path = project_root / args.system
    user_path = project_root / args.user
    note_path = project_root / args.note if not Path(args.note).is_absolute() else Path(args.note)
    out_path = project_root / args.out if not Path(args.out).is_absolute() else Path(args.out)
    
    logger.debug(f"System prompt path: {system_path}")
    logger.debug(f"User prompt path: {user_path}")
    logger.debug(f"Note JSON path: {note_path}")
    logger.debug(f"Output path: {out_path}")
    
    # Read prompts
    logger.info("Reading prompts...")
    system_prompt = read_file_or_die(system_path, "System prompt", logger)
    user_template = read_file_or_die(user_path, "User prompt template", logger)
    logger.debug(f"System prompt length: {len(system_prompt)} chars")
    logger.debug(f"User template length: {len(user_template)} chars")
    
    # Read note JSON
    logger.info(f"Reading note JSON from: {note_path}")
    note_data = read_json_or_die(note_path, "Note JSON", logger)
    logger.info(f"Note ID: {note_data.get('note_id', 'unknown')}")
    logger.info(f"Note title: {note_data.get('title', 'unknown')}")
    logger.info(f"Number of chunks: {len(note_data.get('chunks', []))}")
    
    # Inject JSON into user template
    logger.info("Injecting note JSON into user prompt template...")
    user_prompt = inject_json_into_template(user_template, note_data)
    logger.debug(f"Final user prompt length: {len(user_prompt)} chars")
    
    # Verify placeholder was replaced
    if '"{parsed_json}"' in user_prompt:
        warning_msg = "Placeholder '{parsed_json}' not found in template"
        logger.warning(warning_msg)
        print(f"Warning: {warning_msg}", file=sys.stderr)
    else:
        logger.debug("Placeholder successfully replaced in template")
    
    # Initialize OpenAI client
    logger.info("Initializing OpenAI client...")
    client = OpenAI(api_key=api_key)
    
    # Call OpenAI API with structured outputs
    logger.info("Calling OpenAI API...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "flashcards",
                    "schema": FLASHCARD_SCHEMA,
                    "strict": True
                }
            }
        )
        logger.info("OpenAI API call completed successfully")
        
        # Log usage if available
        if hasattr(response, 'usage'):
            usage = response.usage
            logger.info(
                f"Token usage - Prompt: {getattr(usage, 'prompt_tokens', 'N/A')}, "
                f"Completion: {getattr(usage, 'completion_tokens', 'N/A')}, "
                f"Total: {getattr(usage, 'total_tokens', 'N/A')}"
            )
    except Exception as e:
        error_msg = f"Error calling OpenAI API: {e}"
        logger.error(error_msg, exc_info=True)
        print(f"Error: {error_msg}", file=sys.stderr)
        sys.exit(1)
    
    # Save raw API response
    raw_response_path = out_path.parent / f"{out_path.stem}_raw_api_response.json"
    logger.info(f"Saving raw API response to: {raw_response_path}")
    save_raw_api_response(response, raw_response_path, logger)
    
    # Extract JSON from response
    logger.info("Extracting JSON from API response...")
    flashcards_data = extract_json_from_response(response, logger)
    
    # Validate required fields
    logger.debug("Validating API response structure...")
    if "note_id" not in flashcards_data:
        error_msg = "API response missing 'note_id'"
        logger.error(error_msg)
        print(f"Error: {error_msg}", file=sys.stderr)
        sys.exit(1)
    if "title" not in flashcards_data:
        error_msg = "API response missing 'title'"
        logger.error(error_msg)
        print(f"Error: {error_msg}", file=sys.stderr)
        sys.exit(1)
    if "cards" not in flashcards_data:
        error_msg = "API response missing 'cards'"
        logger.error(error_msg)
        print(f"Error: {error_msg}", file=sys.stderr)
        sys.exit(1)
    logger.debug("API response structure validated")
    
    # Create output directory if needed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory ensured: {out_path.parent}")
    
    # Write output
    logger.info(f"Writing flashcards to: {out_path}")
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(flashcards_data, f, indent=2, ensure_ascii=False)
        logger.info("Flashcards written successfully")
    except Exception as e:
        error_msg = f"Error writing output file: {e}"
        logger.error(error_msg, exc_info=True)
        print(f"Error: {error_msg}", file=sys.stderr)
        sys.exit(1)
    
    # Print summary
    num_cards = len(flashcards_data.get("cards", []))
    note_id = flashcards_data.get("note_id", "unknown")
    title = flashcards_data.get("title", "unknown")
    
    logger.info("=" * 80)
    logger.info("Flashcard generation completed successfully")
    logger.info(f"Generated {num_cards} flashcards")
    logger.info(f"Note: {note_id} - {title}")
    logger.info(f"Output: {out_path}")
    logger.info(f"Raw API response: {raw_response_path}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Error log file: {error_log_file}")
    logger.info("=" * 80)
    
    print(f"Generated {num_cards} flashcards")
    print(f"Note: {note_id} - {title}")
    print(f"Output: {out_path}")
    print(f"Raw API response: {raw_response_path}")
    print(f"Logs: {log_file}")


if __name__ == "__main__":
    main()

