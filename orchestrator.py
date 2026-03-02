"""Orchestrator — runs the arbitration analysis pipeline.

Usage:
    uv run python orchestrator.py            # Run all phases (skips completed)
    uv run python orchestrator.py --force    # Force re-run Phase 1
"""

import argparse
import logging
import sys
from pathlib import Path

from pipeline.phase1_index import run_phase1

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SEGMENTS_DIR = Path("segments")
OUTPUTS_DIR = Path("outputs")


def main() -> None:
    """Entry point for the pipeline orchestrator."""
    parser = argparse.ArgumentParser(description="Construction Arbitration Pipeline")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run phases even if output already exists",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ------------------------------------------------------------------
    # Ensure output directory exists
    # ------------------------------------------------------------------
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1 — Index & Structured Extraction
    # ------------------------------------------------------------------
    logging.getLogger(__name__).info("=== Starting Pipeline ===")

    phase1_results = run_phase1(
        segments_dir=SEGMENTS_DIR,
        output_path=OUTPUTS_DIR / "index.json",
        force=args.force,
    )

    logging.getLogger(__name__).info(
        "Phase 1 produced %d segment entries", len(phase1_results)
    )

    # ------------------------------------------------------------------
    # Phase 2 — Document Type Taxonomy & Classification (TODO)
    # ------------------------------------------------------------------
    # run_phase2(...)

    # ------------------------------------------------------------------
    # Phase 3 — Claim Head Discovery & Mapping (TODO)
    # ------------------------------------------------------------------
    # run_phase3(...)

    # ------------------------------------------------------------------
    # Phase 4 — Case-Level Reasoning & Report (TODO)
    # ------------------------------------------------------------------
    # run_phase4(...)

    logging.getLogger(__name__).info("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
