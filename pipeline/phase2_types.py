"""Phase 2 — Document Type Taxonomy & Classification.

Step 1: Generate a corpus-level taxonomy from index.json metadata.
Step 2: Classify each segment into taxonomy clusters.

All Phase 2-specific LLM logic (prompts, post-processing) lives here.
The generic GeminiClient is imported from utils.llm.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from models.schemas import (
    LLMClassificationResponse,
    LLMTaxonomyResponse,
    SegmentClassification,
)
from utils.file_io import write_json_output
from utils.llm import GeminiClient

logger = logging.getLogger(__name__)

# Model used for Phase 2 taxonomy generation
PHASE2_MODEL = "gemini-2.5-flash"

# Hard cap on parallel Gemini calls
MAX_WORKERS = 5


# ---------------------------------------------------------------------------
# Phase 2 prompts
# ---------------------------------------------------------------------------

TAXONOMY_SYSTEM_PROMPT = """\
You are a senior legal taxonomy architect specializing in construction arbitration disputes.

Your task is to design a coherent, analytically useful document taxonomy
that would help a legal team understand the structure of a dispute.

The taxonomy should reflect distinct legal and contractual functions
within the dispute lifecycle, not generic project phases.
"""

TAXONOMY_USER_PROMPT_TEMPLATE = """\
Analyze this corpus of construction arbitration documents and generate a
document-type taxonomy.

CONSTRAINTS:

1. Create 4 to 7 top-level document clusters.

2. Each cluster must represent a DISTINCT legal or contractual function.
   Clusters must be mutually conceptually separable with minimal thematic overlap.

3. Clusters should reflect structural layers of a dispute such as:
   - contract formation and scope
   - project execution and performance
   - financial administration and claims
   - delay and time-related issues
   - dispute escalation and legal proceedings
   - contractual securities
   (Use these as conceptual guidance only — do not force them if not supported.)

4. Sub-types must reflect document intent or legal function —
   do NOT simply mirror the document_type labels from the input.

5. Each cluster should have 1 to 5 sub-types.
   Prefer 2-4 where meaningful. Do not invent artificial categories.

6. The taxonomy must logically cover every document in the corpus.
   No document should be unclassifiable.

7. Avoid overly broad clusters such as "Project Management" or
   "General Correspondence" unless they are legally meaningful.

8. Do NOT create a "Miscellaneous" or "Other" catch-all cluster
   unless absolutely unavoidable.

9. cluster_key must be snake_case (e.g., "contract_documents").

10. sub_type_key must follow cluster_key/sub_type_slug format
    (e.g., "contract_documents/work_orders").

Output strict JSON matching the provided schema.

CORPUS METADATA:
{corpus_json}"""


# ---------------------------------------------------------------------------
# Taxonomy generation helper
# ---------------------------------------------------------------------------


def _generate_taxonomy(index_data: list[dict]) -> list[dict]:
    """Generate a document-type taxonomy from corpus metadata.

    Args:
        index_data: List of segment dicts from index.json.

    Returns:
        List of cluster dicts matching TaxonomyCluster schema.
    """
    # Condense input — keep only fields needed for taxonomy analysis
    condensed = [
        {
            "segment_id": seg["segment_id"],
            "document_type": seg["document_type"],
            "document_stage": seg["document_stage"],
            "summary": seg["summary"],
            "dispute_signals": seg.get("dispute_signals", []),
        }
        for seg in index_data
    ]

    user_prompt = TAXONOMY_USER_PROMPT_TEMPLATE.format(
        corpus_json=json.dumps(condensed, indent=2)
    )

    client = GeminiClient()
    raw_json = client.generate_json(
        system_prompt=TAXONOMY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=LLMTaxonomyResponse.model_json_schema(),
        model=PHASE2_MODEL,
        temperature=0.2,
    )

    result = LLMTaxonomyResponse.model_validate(raw_json)
    return [cluster.model_dump() for cluster in result.clusters]


# --------------------------------------------------------------------------- #
# Step 1 — Taxonomy Generation                                                #
# --------------------------------------------------------------------------- #


def run_phase2_step1(outputs_dir: Path, force: bool = False) -> list[dict]:
    """Generate document-type taxonomy from corpus metadata.

    Reads index.json, calls the LLM to produce a taxonomy with 4-8 clusters
    and their sub-types, writes result to type_taxonomy.json.

    Args:
        outputs_dir: Directory containing index.json and for writing output.
        force: If True, regenerate even if output exists.

    Returns:
        List of cluster dicts (the taxonomy).

    Raises:
        FileNotFoundError: If index.json does not exist.
    """
    output_path = outputs_dir / "type_taxonomy.json"

    # Checkpoint: skip if already exists
    if output_path.exists() and not force:
        logger.info("Phase 2 Step 1 output exists, loading: %s", output_path)
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        logger.info("Loaded %d clusters from checkpoint", len(existing))
        return existing

    # Dependency check
    index_path = outputs_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Phase 2 requires index.json from Phase 1. Not found: {index_path}"
        )

    logger.info("=== Phase 2 Step 1 — Taxonomy Generation ===")
    start = time.perf_counter()

    # Load corpus index
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    logger.info("Loaded %d segments from index.json", len(index_data))

    # Generate taxonomy via LLM
    taxonomy = _generate_taxonomy(index_data)

    # Write output atomically
    write_json_output(taxonomy, output_path)

    elapsed = time.perf_counter() - start
    logger.info(
        "Phase 2 Step 1 complete: %d clusters generated in %.1fs",
        len(taxonomy),
        elapsed,
    )

    return taxonomy


# --------------------------------------------------------------------------- #
# Step 2 — Classification                                                     #
# --------------------------------------------------------------------------- #

CLASSIFICATION_SYSTEM_PROMPT = """\
You are a document classifier for construction arbitration disputes.

Select 1 to 3 clusters from the provided list that best describe the document.
You MUST choose exactly one primary_cluster.
Do NOT invent new cluster keys — only use keys from the provided list.
"""

CLASSIFICATION_USER_PROMPT_TEMPLATE = """\
Classify this document into the most appropriate cluster(s).

DOCUMENT METADATA:
- Document Type: {document_type}
- Document Stage: {document_stage}
- Summary: {summary}
- Dispute Signals: {dispute_signals}

VALID CLUSTER KEYS (choose only from this list):
{cluster_list}

RULES:
1. Select exactly ONE primary_cluster from the list above.
2. Optionally select up to TWO secondary_clusters if the document spans multiple themes.
3. secondary_clusters must NOT include the primary_cluster.
4. Total clusters assigned must be between 1 and 3.
5. Confidence (0.0 to 1.0) applies to the primary_cluster selection.
6. Do NOT invent new cluster keys.

Output strict JSON matching the provided schema."""


def _classify_one(
    client: GeminiClient,
    segment: dict,
    valid_clusters: set[str],
) -> SegmentClassification | None:
    """Classify a single segment. Returns None on failure."""
    segment_id = segment["segment_id"]
    start = time.perf_counter()

    try:
        # Format dispute signals for prompt
        signals = segment.get("dispute_signals", [])
        signal_str = ", ".join(
            s.get("signal_type", "") for s in signals if s.get("signal_type")
        ) or "none"

        user_prompt = CLASSIFICATION_USER_PROMPT_TEMPLATE.format(
            document_type=segment.get("document_type", "Unknown"),
            document_stage=segment.get("document_stage", "Unknown"),
            summary=segment.get("summary", "No summary available"),
            dispute_signals=signal_str,
            cluster_list="\n".join(f"- {k}" for k in sorted(valid_clusters)),
        )

        raw_json = client.generate_json(
            system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=LLMClassificationResponse.model_json_schema(),
            model=PHASE2_MODEL,
            temperature=0.1,
        )

        # Validate response
        llm_result = LLMClassificationResponse.model_validate(raw_json)

        # Post-LLM validation: check cluster keys exist
        if llm_result.primary_cluster not in valid_clusters:
            raise ValueError(
                f"Invalid primary_cluster: {llm_result.primary_cluster}"
            )

        for sec in llm_result.secondary_clusters:
            if sec not in valid_clusters:
                raise ValueError(f"Invalid secondary_cluster: {sec}")

        if llm_result.primary_cluster in llm_result.secondary_clusters:
            raise ValueError("primary_cluster cannot appear in secondary_clusters")

        # Limit secondary_clusters to 2
        secondary = llm_result.secondary_clusters[:2]

        # Total clusters must be 1-3
        total = 1 + len(secondary)
        if total > 3:
            raise ValueError(f"Too many clusters assigned: {total}")

        # Round confidence to 2 decimal places
        confidence = round(llm_result.confidence, 2)

        result = SegmentClassification(
            segment_id=segment_id,
            primary_cluster=llm_result.primary_cluster,
            secondary_clusters=secondary,
            confidence=confidence,
        )

        elapsed = time.perf_counter() - start
        logger.info("Done: %s → %s (%.1fs)", segment_id, result.primary_cluster, elapsed)
        return result

    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.error("Failed: %s (%.1fs): %s", segment_id, elapsed, exc)
        return None


def run_phase2_step2(
    outputs_dir: Path,
    force: bool = False,
    executor: ThreadPoolExecutor | None = None,
) -> list[dict]:
    """Classify each segment into taxonomy clusters.

    Reads index.json and type_taxonomy.json, classifies each segment
    into 1-3 clusters, writes result to type_classifications.json.

    Args:
        outputs_dir: Directory containing index.json and type_taxonomy.json.
        force: If True, reclassify even if output exists.
        executor: Optional shared ThreadPoolExecutor for LLM calls.
            If None, creates an internal pool.

    Returns:
        List of classification dicts.

    Raises:
        FileNotFoundError: If required input files do not exist.
    """
    output_path = outputs_dir / "type_classifications.json"

    # Checkpoint: skip if already exists
    if output_path.exists() and not force:
        logger.info("Phase 2 Step 2 output exists, loading: %s", output_path)
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        logger.info("Loaded %d classifications from checkpoint", len(existing))
        return existing

    # Dependency checks
    index_path = outputs_dir / "index.json"
    taxonomy_path = outputs_dir / "type_taxonomy.json"

    if not index_path.exists():
        raise FileNotFoundError(f"index.json not found: {index_path}")
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"type_taxonomy.json not found: {taxonomy_path}")

    logger.info("=== Phase 2 Step 2 — Classification ===")
    phase_start = time.perf_counter()

    # Load inputs
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    taxonomy_data = json.loads(taxonomy_path.read_text(encoding="utf-8"))

    # Extract valid cluster keys (top-level only, no sub-types)
    valid_clusters: set[str] = {c["cluster_key"] for c in taxonomy_data}

    logger.info(
        "Loaded %d segments, %d valid clusters",
        len(index_data),
        len(valid_clusters),
    )

    # Process segments in parallel
    client = GeminiClient()
    results: list[SegmentClassification] = []
    failures: list[str] = []

    def _run_classification(pool: ThreadPoolExecutor) -> None:
        """Execute classification with the given pool."""
        future_to_seg = {
            pool.submit(_classify_one, client, seg, valid_clusters): seg["segment_id"]
            for seg in index_data
        }

        for i, future in enumerate(as_completed(future_to_seg), 1):
            segment_id = future_to_seg[future]
            result = future.result()

            if result is not None:
                results.append(result)
            else:
                failures.append(segment_id)

            logger.info("Phase 2 Step 2 progress: %d/%d complete", i, len(index_data))

    if executor is not None:
        _run_classification(executor)
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            _run_classification(pool)

    # Sort by segment_id for deterministic output
    results.sort(key=lambda r: r.segment_id)

    # Write output atomically
    data = [r.model_dump() for r in results]
    write_json_output(data, output_path)

    elapsed = time.perf_counter() - phase_start
    logger.info(
        "Phase 2 Step 2 complete: %d succeeded, %d failed, %.1fs",
        len(results),
        len(failures),
        elapsed,
    )

    if failures:
        logger.warning("Failed segments: %s", ", ".join(sorted(failures)))

    return data


# --------------------------------------------------------------------------- #
# Full Phase 2 Runner (for parallel orchestration)                            #
# --------------------------------------------------------------------------- #


def run_phase2_full(
    outputs_dir: Path,
    force: bool = False,
    executor: ThreadPoolExecutor | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run Phase 2 Step 1 and Step 2 sequentially.

    Designed for parallel orchestration where Phase 2 and Phase 3
    run concurrently but internally execute their steps sequentially.

    Args:
        outputs_dir: Directory containing index.json and for output.
        force: If True, re-run even if outputs exist.
        executor: Optional shared ThreadPoolExecutor for LLM calls in Step 2.

    Returns:
        Tuple of (taxonomy, classifications).
    """
    taxonomy = run_phase2_step1(outputs_dir=outputs_dir, force=force)
    classifications = run_phase2_step2(
        outputs_dir=outputs_dir, force=force, executor=executor
    )
    return taxonomy, classifications

