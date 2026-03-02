"""File I/O helpers for the arbitration pipeline.

Simple, dependency-free functions for reading OCR text, discovering
segments, and writing JSON output atomically.
"""

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Regex to match OCR page delimiters like "--- Page 1 ---"
_PAGE_DELIMITER_RE = re.compile(r"^-{2,}\s*Page\s+\d+\s*-{2,}\s*$", re.MULTILINE)


def load_ocr_text(segment_dir: Path) -> str:
    """Read ocr_text.txt from a segment directory and clean it.

    Strips page delimiter lines (``--- Page N ---``) and normalizes
    excessive whitespace while preserving paragraph structure.

    Args:
        segment_dir: Path to the segment folder containing ocr_text.txt.

    Returns:
        Cleaned OCR text as a single string.

    Raises:
        FileNotFoundError: If ocr_text.txt does not exist in the directory.
    """
    ocr_path = segment_dir / "ocr_text.txt"
    raw = ocr_path.read_text(encoding="utf-8")

    # Remove page delimiter lines
    cleaned = _PAGE_DELIMITER_RE.sub("", raw)

    # Collapse runs of 3+ newlines into 2 (preserve paragraph breaks)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def discover_segments(segments_dir: Path) -> list[str]:
    """Find all segment directories that contain an ocr_text.txt file.

    Args:
        segments_dir: Root segments/ directory.

    Returns:
        Sorted list of segment folder names (used as segment IDs).
    """
    segment_ids: list[str] = []
    for entry in sorted(segments_dir.iterdir()):
        if entry.is_dir() and (entry / "ocr_text.txt").exists():
            segment_ids.append(entry.name)

    logger.info("Discovered %d segments in %s", len(segment_ids), segments_dir)
    return segment_ids


def write_json_output(data: list[dict], output_path: Path) -> None:
    """Atomically write JSON data to a file.

    Writes to a temporary file first, then renames. This prevents
    corrupted output if the process is killed mid-write.

    Args:
        data: List of dicts to serialize.
        output_path: Final destination path for the JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = output_path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp_path, output_path)

    logger.info("Wrote output to %s (%d entries)", output_path, len(data))
