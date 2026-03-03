"""Orchestrator — runs the arbitration analysis pipeline.

Usage:
    uv run python orchestrator.py                # Run all phases (parallel, skips completed)
    uv run python orchestrator.py --force        # Force re-run all phases
    uv run python orchestrator.py --phase 1 4 --force  # Re-run only Phase 1 and Phase 4
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
from pipeline.phase4_analysis import run_phase4

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
    parser.add_argument(
        "--phase",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4],
        default=None,
        help="Run only specific phase(s), e.g. --phase 1 4",
    )
    args = parser.parse_args()

    # Which phases to run (None = all)
    run_phases: set[int] = set(args.phase) if args.phase else {1, 2, 3, 4}

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
    logger = logging.getLogger(__name__)
    logger.info("=== Starting Pipeline (phases: %s) ===",
               ", ".join(str(p) for p in sorted(run_phases)))

    # ------------------------------------------------------------------
    # Phase 1 — Index & Structured Extraction
    # ------------------------------------------------------------------
    if 1 in run_phases:
        phase1_results = run_phase1(
            segments_dir=SEGMENTS_DIR,
            output_path=OUTPUTS_DIR / "index.json",
            force=args.force,
        )
        logger.info("Phase 1 produced %d segment entries", len(phase1_results))
    else:
        logger.info("Skipping Phase 1")

    # ------------------------------------------------------------------
    # Phase 2 & 3 — Parallel or Sequential execution
    # ------------------------------------------------------------------
    run_middle = bool(run_phases & {2, 3})

    if run_middle and args.parallel:
        logger.info("=== Running Phase 2 and Phase 3 in parallel ===")
        parallel_start = time.perf_counter()

        # Shared LLM worker pool (max 5 concurrent API calls)
        with ThreadPoolExecutor(max_workers=5) as llm_executor:
            # Phase-level executor (2 workers: one per phase)
            with ThreadPoolExecutor(max_workers=2) as phase_executor:
                futures: dict = {}
                if 2 in run_phases:
                    futures[phase_executor.submit(
                        run_phase2_full,
                        outputs_dir=OUTPUTS_DIR,
                        force=args.force,
                        executor=llm_executor,
                    )] = "Phase 2"
                if 3 in run_phases:
                    futures[phase_executor.submit(
                        run_phase3_full,
                        outputs_dir=OUTPUTS_DIR,
                        force=args.force,
                        executor=llm_executor,
                    )] = "Phase 3"

                # Wait for submitted phases to complete
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

    elif run_middle:
        logger.info("=== Running Phase 2 and Phase 3 sequentially ===")

        if 2 in run_phases:
            phase2_taxonomy = run_phase2_step1(outputs_dir=OUTPUTS_DIR, force=args.force)
            logger.info("Phase 2 Step 1 produced %d clusters", len(phase2_taxonomy))

            phase2_classifications = run_phase2_step2(
                outputs_dir=OUTPUTS_DIR, force=args.force
            )
            logger.info(
                "Phase 2 Step 2 produced %d classifications", len(phase2_classifications)
            )

        if 3 in run_phases:
            phase3_claims = run_phase3_step1(outputs_dir=OUTPUTS_DIR, force=args.force)
            logger.info("Phase 3 Step 1 produced %d claim heads", len(phase3_claims))

            phase3_mappings = run_phase3_step2(outputs_dir=OUTPUTS_DIR, force=args.force)
            logger.info("Phase 3 Step 2 produced %d claim mappings", len(phase3_mappings))
    else:
        logger.info("Skipping Phase 2 and Phase 3")

    # ------------------------------------------------------------------
    # Phase 4 — Case-Level Reasoning & Report
    # ------------------------------------------------------------------
    if 4 in run_phases:
        logger.info("=== Starting Phase 4 ===")
        report_path = run_phase4(outputs_dir=OUTPUTS_DIR, force=args.force)
        logger.info("Phase 4 produced report at %s", report_path)
    else:
        logger.info("Skipping Phase 4")

    logger.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
