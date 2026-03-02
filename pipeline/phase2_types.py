"""Phase 2 — Document Type Taxonomy & Classification.

Step 1: Generate a corpus-level taxonomy from index.json metadata.
Step 2: Classify each segment into taxonomy sub-types. (TODO)

All Phase 2-specific LLM logic (prompts, post-processing) lives here.
The generic GeminiClient is imported from utils.llm.
"""

import json
import logging
import time
from pathlib import Path

from models.schemas import LLMTaxonomyResponse
from utils.file_io import write_json_output
from utils.llm import GeminiClient

logger = logging.getLogger(__name__)

# Model used for Phase 2 taxonomy generation
PHASE2_MODEL = "gemini-2.5-flash"

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
# Step 2 — Classification (TODO)                                              #
# --------------------------------------------------------------------------- #


def run_phase2_step2(outputs_dir: Path, force: bool = False) -> list[dict]:
    """Classify each segment into taxonomy sub-types.

    TODO: Implement classification of each segment against the taxonomy.

    Args:
        outputs_dir: Directory containing type_taxonomy.json and index.json.
        force: If True, reclassify even if output exists.

    Returns:
        List of classification dicts.
    """
    raise NotImplementedError("Phase 2 Step 2 not yet implemented")
