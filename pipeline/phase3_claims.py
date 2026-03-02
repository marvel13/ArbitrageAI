"""Phase 3 — Claim Head Discovery.

Step 1: Discover financially distinct claim heads from the corpus.
This phase performs dispute abstraction, not clustering.

All Phase 3-specific LLM logic (prompts, validation) lives here.
The generic GeminiClient is imported from utils.llm.
"""

import json
import logging
import time
from pathlib import Path

from models.schemas import ClaimHead, LLMClaimHeadsResponse
from utils.file_io import write_json_output
from utils.llm import GeminiClient

logger = logging.getLogger(__name__)

# Model used for Phase 3 claim discovery — requires stronger reasoning
PHASE3_MODEL = "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# Phase 3 prompts
# ---------------------------------------------------------------------------

CLAIM_DISCOVERY_SYSTEM_PROMPT = """\
You are a senior arbitration analyst performing dispute abstraction.

Your task is to identify the financially distinct claim heads that define
what this arbitration dispute is fundamentally about.

This is NOT document clustering. You must reason about:
- What categories of financial recovery are being pursued
- Who is pursuing each category (contractor or employer)
- What documentary evidence supports each claim head

CRITICAL RULES:

1. Identify BOTH claims (contractor pursuing recovery) AND counterclaims
   (employer pursuing recovery, e.g., liquidated damages) if present.

2. Merge related financial themes into coherent claim heads:
   - Multiple escalation documents → ONE escalation claim head
   - Multiple delay/EOT documents → ONE delay/prolongation claim head
   - Do NOT micro-fragment into trivial themes

3. Ground every claim head in at least TWO documents from the corpus.
   Do NOT invent claims not supported by documentary evidence.

4. Do NOT create vague themes like "Miscellaneous Issues" or "Other Claims".

5. Sub-heads are OPTIONAL. Use them only when a claim head has clear
   structural components (e.g., prolongation → site overheads, idle machinery).
   Sub-heads are descriptive only — no supporting IDs or amounts.

6. approximate_claimed_amount must be a narrative estimate grounded in
   document language (e.g., "Approx. INR 6-7 crore based on claim statements").
   Do NOT compute numeric totals.
7. Completeness is more important than excessive specificity. However, you must strike the right balance:
   - Do NOT merge financially distinct recovery categories into one broad head.
   - Do NOT fragment one financial theme into multiple trivial heads.
   - Each claim head must represent a coherent, legally meaningful category of recovery.
"""

CLAIM_DISCOVERY_USER_PROMPT_TEMPLATE = """\
Analyze this corpus of construction arbitration documents and identify
the financially distinct claim heads that define this dispute.

CONSTRAINTS:

1. Each claim head must be supported by at least 2 distinct segment IDs.
   Do NOT create single-document claim heads.

2. Prefer 3-7 major claim heads unless the corpus clearly supports more.
   Avoid micro-fragmentation.

3. supporting_segment_ids must reference ACTUAL segment IDs from the input.
   Do not invent segment IDs.

4. claimant must be exactly "contractor" or "employer".

5. approximate_claimed_amount must be:
   - A narrative estimate (e.g., "Approx. INR 6-7 crore based on claim statements")
   - Grounded in document language
   - NOT a computed numeric total
   - Can be null if no amount is determinable

6. claim_key must be snake_case (e.g., "escalation_claim", "delay_damages").

7. Sub-heads are optional. Only include them if they represent clear
   structural components of a larger claim head.

8. Do NOT use predefined claim categories. Discover from the corpus.

9. Do NOT create overlapping claim heads for the same financial theme.

Output strict JSON matching the provided schema.

CORPUS METADATA:
{corpus_json}"""


# ---------------------------------------------------------------------------
# Input condensation
# ---------------------------------------------------------------------------


def _condense_index_for_claims(index_data: list[dict]) -> list[dict]:
    """Extract only claim-relevant fields from index data.

    Phase 3 operates independently of Phase 2 — no taxonomy or
    classification data is included.
    """
    condensed = []
    for segment in index_data:
        condensed.append({
            "segment_id": segment["segment_id"],
            "document_type": segment.get("document_type", ""),
            "summary": segment.get("summary", ""),
            "monetary_amounts": segment.get("monetary_amounts", []),
            "dispute_signals": segment.get("dispute_signals", []),
            "parties": segment.get("parties", []),
        })
    return condensed


# ---------------------------------------------------------------------------
# Post-LLM validation
# ---------------------------------------------------------------------------


def _validate_claim_heads(
    claim_heads: list[ClaimHead],
    valid_segment_ids: set[str],
) -> None:
    """Validate claim heads against corpus constraints.

    Raises ValueError if validation fails, triggering retry.
    """
    if not claim_heads:
        raise ValueError("No claim heads returned")

    seen_keys: set[str] = set()

    for head in claim_heads:
        # Check for duplicate claim_key
        if head.claim_key in seen_keys:
            raise ValueError(f"Duplicate claim_key: {head.claim_key}")
        seen_keys.add(head.claim_key)

        # Check minimum supporting segments
        if len(head.supporting_segment_ids) < 2:
            raise ValueError(
                f"Claim head '{head.claim_key}' has fewer than 2 supporting segments"
            )

        # Check for empty supporting_segment_ids
        if not head.supporting_segment_ids:
            raise ValueError(
                f"Claim head '{head.claim_key}' has no supporting segments"
            )

        # Check all segment IDs exist in corpus
        for seg_id in head.supporting_segment_ids:
            if seg_id not in valid_segment_ids:
                raise ValueError(
                    f"Claim head '{head.claim_key}' references invalid segment: {seg_id}"
                )

        # Check claimant is valid
        if head.claimant not in {"contractor", "employer"}:
            raise ValueError(
                f"Claim head '{head.claim_key}' has invalid claimant: {head.claimant}"
            )

    # Warn if count is outside expected range
    count = len(claim_heads)
    if count < 3:
        logger.warning("Only %d claim heads discovered (expected 3-7)", count)
    elif count > 10:
        logger.warning("Many claim heads discovered (%d) — possible fragmentation", count)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_phase3_step1(outputs_dir: Path, force: bool = False) -> list[dict]:
    """Discover claim heads from corpus metadata.

    Performs dispute-level reasoning to identify financially distinct
    claim heads, grounded in documentary evidence.

    Args:
        outputs_dir: Directory containing index.json and for output.
        force: If True, re-run even if output exists.

    Returns:
        List of claim head dicts written to claim_heads.json.

    Raises:
        FileNotFoundError: If index.json does not exist.
        RuntimeError: If LLM call fails after retries.
    """
    output_path = outputs_dir / "claim_heads.json"

    # Checkpoint: skip if already exists
    if output_path.exists() and not force:
        logger.info("Phase 3 Step 1 output exists, loading: %s", output_path)
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        return existing

    logger.info("=== Phase 3 Step 1 — Claim Head Discovery ===")
    start = time.perf_counter()

    # Load index.json dependency
    index_path = outputs_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Phase 3 requires index.json but not found: {index_path}"
        )

    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    valid_segment_ids = {seg["segment_id"] for seg in index_data}

    logger.info("Loaded %d segments from index.json", len(index_data))

    # Condense input for LLM
    condensed = _condense_index_for_claims(index_data)
    corpus_json = json.dumps(condensed, indent=2, ensure_ascii=False)

    # Build user prompt
    user_prompt = CLAIM_DISCOVERY_USER_PROMPT_TEMPLATE.format(corpus_json=corpus_json)

    # Call LLM with schema enforcement
    client = GeminiClient()
    raw_json = client.generate_json(
        system_prompt=CLAIM_DISCOVERY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=LLMClaimHeadsResponse.model_json_schema(),
        model=PHASE3_MODEL,
        temperature=0.2,
    )

    # Parse and validate
    response = LLMClaimHeadsResponse.model_validate(raw_json)
    _validate_claim_heads(response.claim_heads, valid_segment_ids)

    # Serialize and write
    claim_heads = [head.model_dump() for head in response.claim_heads]
    write_json_output(claim_heads, output_path)

    elapsed = time.perf_counter() - start
    logger.info(
        "Phase 3 Step 1 complete: %d claim heads discovered in %.1fs",
        len(claim_heads),
        elapsed,
    )

    return claim_heads
