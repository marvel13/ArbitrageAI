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
from pipeline.phase2_types import run_phase2_step1, run_phase2_step2
from pipeline.phase3_claims import run_phase3_step1, run_phase3_step2

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
    # Phase 2 — Document Type Taxonomy & Classification
    # ------------------------------------------------------------------
    phase2_taxonomy = run_phase2_step1(outputs_dir=OUTPUTS_DIR, force=args.force)

    logging.getLogger(__name__).info(
        "Phase 2 Step 1 produced %d clusters", len(phase2_taxonomy)
    )

    # Phase 2 Step 2 — Classification
    phase2_classifications = run_phase2_step2(outputs_dir=OUTPUTS_DIR, force=args.force)

    logging.getLogger(__name__).info(
        "Phase 2 Step 2 produced %d classifications", len(phase2_classifications)
    )

    # ------------------------------------------------------------------
    # Phase 3 — Claim Head Discovery & Mapping
    # ------------------------------------------------------------------
    phase3_claims = run_phase3_step1(outputs_dir=OUTPUTS_DIR, force=args.force)

    logging.getLogger(__name__).info(
        "Phase 3 Step 1 produced %d claim heads", len(phase3_claims)
    )

    # Phase 3 Step 2 — Claim Mapping
    phase3_mappings = run_phase3_step2(outputs_dir=OUTPUTS_DIR, force=args.force)

    logging.getLogger(__name__).info(
        "Phase 3 Step 2 produced %d claim mappings", len(phase3_mappings)
    )

    # ------------------------------------------------------------------
    # Phase 4 — Case-Level Reasoning & Report (TODO)
    # ------------------------------------------------------------------
    # run_phase4(...)

    logging.getLogger(__name__).info("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
