"""Orchestrator — runs the arbitration analysis pipeline.

Usage:
    uv run python orchestrator.py            # Run all phases (parallel, skips completed)
    uv run python orchestrator.py --force    # Force re-run all phases
    uv run python orchestrator.py --no-parallel  # Run sequentially
"""

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline.phase1_index import run_phase1
from pipeline.phase2_types import run_phase2_full, run_phase2_step1, run_phase2_step2
from pipeline.phase3_claims import run_phase3_full, run_phase3_step1, run_phase3_step2

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
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        dest="parallel",
        help="Run Phase 2 and Phase 3 in parallel (default: True)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_false",
        dest="parallel",
        help="Run Phase 2 and Phase 3 sequentially",
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
    # Phase 2 & 3 — Parallel or Sequential execution
    # ------------------------------------------------------------------
    logger = logging.getLogger(__name__)

    if args.parallel:
        logger.info("=== Running Phase 2 and Phase 3 in parallel ===")
        parallel_start = time.perf_counter()

        # Shared LLM worker pool (max 5 concurrent API calls)
        with ThreadPoolExecutor(max_workers=5) as llm_executor:
            # Phase-level executor (2 workers: one per phase)
            with ThreadPoolExecutor(max_workers=2) as phase_executor:
                phase2_future = phase_executor.submit(
                    run_phase2_full,
                    outputs_dir=OUTPUTS_DIR,
                    force=args.force,
                    executor=llm_executor,
                )
                phase3_future = phase_executor.submit(
                    run_phase3_full,
                    outputs_dir=OUTPUTS_DIR,
                    force=args.force,
                    executor=llm_executor,
                )

                # Wait for both phases to complete
                futures = {phase2_future: "Phase 2", phase3_future: "Phase 3"}
                for future in as_completed(futures):
                    phase_name = futures[future]
                    try:
                        result = future.result()
                        if phase_name == "Phase 2":
                            phase2_taxonomy, phase2_classifications = result
                            logger.info(
                                "%s complete: %d clusters, %d classifications",
                                phase_name,
                                len(phase2_taxonomy),
                                len(phase2_classifications),
                            )
                        else:
                            phase3_claims, phase3_mappings = result
                            logger.info(
                                "%s complete: %d claim heads, %d mappings",
                                phase_name,
                                len(phase3_claims),
                                len(phase3_mappings),
                            )
                    except Exception as exc:
                        logger.error("%s failed: %s", phase_name, exc)
                        raise

        parallel_elapsed = time.perf_counter() - parallel_start
        logger.info("Parallel phases completed in %.1fs", parallel_elapsed)

    else:
        logger.info("=== Running Phase 2 and Phase 3 sequentially ===")

        # Phase 2 — Document Type Taxonomy & Classification
        phase2_taxonomy = run_phase2_step1(outputs_dir=OUTPUTS_DIR, force=args.force)
        logger.info("Phase 2 Step 1 produced %d clusters", len(phase2_taxonomy))

        phase2_classifications = run_phase2_step2(
            outputs_dir=OUTPUTS_DIR, force=args.force
        )
        logger.info(
            "Phase 2 Step 2 produced %d classifications", len(phase2_classifications)
        )

        # Phase 3 — Claim Head Discovery & Mapping
        phase3_claims = run_phase3_step1(outputs_dir=OUTPUTS_DIR, force=args.force)
        logger.info("Phase 3 Step 1 produced %d claim heads", len(phase3_claims))

        phase3_mappings = run_phase3_step2(outputs_dir=OUTPUTS_DIR, force=args.force)
        logger.info("Phase 3 Step 2 produced %d claim mappings", len(phase3_mappings))

    # ------------------------------------------------------------------
    # Phase 4 — Case-Level Reasoning & Report (TODO)
    # ------------------------------------------------------------------
    # run_phase4(...)

    logging.getLogger(__name__).info("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
