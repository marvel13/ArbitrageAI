"""Phase 3 — Claim Head Discovery & Mapping.

Step 1: Discover financially distinct claim heads from the corpus.
Step 2: Map each segment to relevant claim heads.

This phase performs dispute abstraction, not clustering.

All Phase 3-specific LLM logic (prompts, validation) lives here.
The generic GeminiClient is imported from utils.llm.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from models.schemas import (
    ClaimHead,
    ClaimMapping,
    LLMClaimHeadsResponse,
    LLMClaimMappingsResponse,
    SegmentClaimMapping,
)
from utils.file_io import write_json_output
from utils.llm import GeminiClient

logger = logging.getLogger(__name__)

# Model used for Phase 3 claim discovery — requires stronger reasoning
PHASE3_MODEL = "gemini-2.5-pro"

# Model used for Phase 3 Step 2 — classification is simpler
PHASE3_STEP2_MODEL = "gemini-2.5-flash"

# Hard cap on parallel Gemini calls
MAX_WORKERS = 5


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


# ---------------------------------------------------------------------------
# Phase 3 Step 2 — Claim Mapping
# ---------------------------------------------------------------------------

CLAIM_MAPPING_SYSTEM_PROMPT = """\
You are an arbitration document analyst determining how documents relate
to identified claim heads in a construction dispute.

Your task is to assess how a single document relates to one or more
previously identified claim heads.

For each document, determine:

1. Which claim heads (if any) the document relates to.
2. The type of relevance:
   - "direct": The document explicitly evidences, quantifies, asserts, or forms
     part of the legal basis of the claim.
   - "contextual": The document provides background, contractual framework,
     timeline context, or procedural setting relevant to the claim.
   - "rebuttal": The document disputes, counters, rejects, or challenges
     the opposing party’s position regarding the claim.
3. Which party the document supports:
   - "supports_contractor"
   - "supports_employer"
   - "neutral"

------------------------------------------------------------
RULES
------------------------------------------------------------

- A document may relate to ZERO claims if it is purely procedural or not
  materially relevant.
- A document may relate to MULTIPLE claims if there is genuine
  multi-claim relevance.
- Only return mappings with confidence >= 0.5.
- Do NOT force mappings. If relevance is weak or speculative,
  return an empty list.
- Reasoning must be concise (1–2 sentences) and document-specific.
  Do not restate the claim title. Refer to what the document actually does.

------------------------------------------------------------
GUARDRAILS
------------------------------------------------------------

- Do NOT assign a claim mapping unless the document has a clear,
  meaningful connection to the claim head.
- Avoid over-assigning "contextual" relevance.
- party_role must reflect which side the document substantively supports
  in relation to the specific claim head — NOT merely who authored it.
- Most documents should map to 1–2 claim heads.
  Mapping to more than 3 claims should be rare and justified by genuine
  multi-claim relevance.

------------------------------------------------------------
CONFIDENCE CALIBRATION RULES
------------------------------------------------------------

Confidence must reflect the evidentiary strength of the document in
relation to the claim head — NOT rhetorical clarity or narrative strength.

Use this scale:

- 0.95-1.0  
  The document explicitly quantifies, asserts, or directly forms part of
  the legal or financial basis of the claim (e.g., invoice, levy order,
  claim statement section, quantified demand).

- 0.80-0.94  
  The document clearly supports, disputes, or materially advances the
  claim but does not independently quantify or conclusively establish it.

- 0.65-0.79  
  The document is materially relevant but indirect, inferential,
  or partially supportive.

- 0.50-0.64  
  The document provides limited but genuine contextual relevance.

IMPORTANT:
- Do NOT default to 0.95 or 1.0.
- Use 1.0 only when the document is explicit, unambiguous,
  and central to the claim.
- Contextual mappings should rarely exceed 0.80.
- Rebuttal mappings should reflect how strongly the document
  substantively counters the claim.
- Confidence must meaningfully discriminate between strong,
  moderate, and weak relevance.

Your goal is to produce disciplined, legally reasoned mappings
that reflect evidentiary weight — not over-inclusive tagging.
  """

CLAIM_MAPPING_USER_PROMPT_TEMPLATE = """\
Determine which claim heads this document relates to.

AVAILABLE CLAIM HEADS:
{claim_heads_json}

DOCUMENT TO ANALYZE:
{segment_json}

CONSTRAINTS:
1. claim_key must EXACTLY match one from the available claim heads.
2. Only include mappings with confidence >= 0.5.
3. Return empty mappings list if document is not claim-relevant.
4. Provide concise reasoning (1-2 sentences) for each mapping.

Output strict JSON matching the provided schema."""


def _build_claim_heads_summary(claim_heads: list[dict]) -> list[dict]:
    """Build condensed claim heads summary for mapping prompt."""
    return [
        {
            "claim_key": h["claim_key"],
            "title": h["title"],
            "claimant": h["claimant"],
        }
        for h in claim_heads
    ]


def _condense_segment_for_mapping(segment: dict) -> dict:
    """Extract mapping-relevant fields from a segment."""
    return {
        "segment_id": segment["segment_id"],
        "document_type": segment.get("document_type", ""),
        "summary": segment.get("summary", ""),
        "dispute_signals": segment.get("dispute_signals", []),
    }


def _map_segment_to_claims(
    client: GeminiClient,
    segment: dict,
    claim_heads_summary: list[dict],
    valid_claim_keys: set[str],
) -> list[SegmentClaimMapping]:
    """Map a single segment to relevant claim heads.

    Returns list of SegmentClaimMapping (0-N per segment).
    """
    segment_id = segment["segment_id"]
    condensed = _condense_segment_for_mapping(segment)

    user_prompt = CLAIM_MAPPING_USER_PROMPT_TEMPLATE.format(
        claim_heads_json=json.dumps(claim_heads_summary, indent=2),
        segment_json=json.dumps(condensed, indent=2),
    )

    raw_json = client.generate_json(
        system_prompt=CLAIM_MAPPING_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=LLMClaimMappingsResponse.model_json_schema(),
        model=PHASE3_STEP2_MODEL,
        temperature=0.1,
    )

    response = LLMClaimMappingsResponse.model_validate(raw_json)

    # Filter and validate mappings
    results: list[SegmentClaimMapping] = []
    for mapping in response.mappings:
        # Skip invalid claim keys
        if mapping.claim_key not in valid_claim_keys:
            logger.warning(
                "Segment %s: invalid claim_key '%s', skipping",
                segment_id,
                mapping.claim_key,
            )
            continue

        # Skip low confidence
        if mapping.confidence < 0.5:
            continue

        results.append(
            SegmentClaimMapping(
                segment_id=segment_id,
                claim_key=mapping.claim_key,
                relevance_type=mapping.relevance_type,
                party_role=mapping.party_role,
                confidence=mapping.confidence,
                reasoning=mapping.reasoning,
            )
        )

    return results


def run_phase3_step2(
    outputs_dir: Path,
    force: bool = False,
    executor: ThreadPoolExecutor | None = None,
) -> list[dict]:
    """Map segments to claim heads.

    For each segment, determines which claim heads (if any) it relates to,
    how it relates (direct/contextual/rebuttal), and which party it supports.

    Args:
        outputs_dir: Directory containing index.json, claim_heads.json.
        force: If True, re-run even if output exists.
        executor: Optional shared ThreadPoolExecutor for LLM calls.
            If None, creates an internal pool.

    Returns:
        List of segment-claim mapping dicts written to claim_classifications.json.

    Raises:
        FileNotFoundError: If dependencies do not exist.
        RuntimeError: If LLM calls fail after retries.
    """
    output_path = outputs_dir / "claim_classifications.json"

    # Checkpoint: skip if already exists
    if output_path.exists() and not force:
        logger.info("Phase 3 Step 2 output exists, loading: %s", output_path)
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        return existing

    logger.info("=== Phase 3 Step 2 — Claim Mapping ===")
    start = time.perf_counter()

    # Load dependencies
    index_path = outputs_dir / "index.json"
    claims_path = outputs_dir / "claim_heads.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Phase 3 Step 2 requires index.json: {index_path}")
    if not claims_path.exists():
        raise FileNotFoundError(f"Phase 3 Step 2 requires claim_heads.json: {claims_path}")

    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    claim_heads = json.loads(claims_path.read_text(encoding="utf-8"))

    valid_claim_keys = {h["claim_key"] for h in claim_heads}
    claim_heads_summary = _build_claim_heads_summary(claim_heads)

    logger.info(
        "Mapping %d segments to %d claim heads",
        len(index_data),
        len(claim_heads),
    )

    # Parallel classification
    client = GeminiClient()
    all_mappings: list[SegmentClaimMapping] = []
    failures: list[str] = []

    def _run_mapping(pool: ThreadPoolExecutor) -> None:
        """Execute claim mapping with the given pool."""
        future_to_seg = {
            pool.submit(
                _map_segment_to_claims,
                client,
                segment,
                claim_heads_summary,
                valid_claim_keys,
            ): segment["segment_id"]
            for segment in index_data
        }

        for future in as_completed(future_to_seg):
            segment_id = future_to_seg[future]
            try:
                mappings = future.result()
                all_mappings.extend(mappings)
                logger.info(
                    "Phase 3 Step 2: Mapped segment %s → %d claims",
                    segment_id,
                    len(mappings),
                )
            except Exception as exc:
                logger.error("Failed to map segment %s: %s", segment_id, exc)
                failures.append(segment_id)

    if executor is not None:
        _run_mapping(executor)
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            _run_mapping(pool)

    if failures:
        logger.warning("Failed to map %d segments: %s", len(failures), failures)

    # Sort for determinism
    all_mappings.sort(key=lambda m: (m.segment_id, m.claim_key))

    # Serialize and write
    mappings_data = [m.model_dump() for m in all_mappings]
    write_json_output(mappings_data, output_path)

    elapsed = time.perf_counter() - start
    logger.info(
        "Phase 3 Step 2 complete: %d mappings from %d segments in %.1fs",
        len(mappings_data),
        len(index_data),
        elapsed,
    )

    return mappings_data


# --------------------------------------------------------------------------- #
# Full Phase 3 Runner (for parallel orchestration)                            #
# --------------------------------------------------------------------------- #


def run_phase3_full(
    outputs_dir: Path,
    force: bool = False,
    executor: ThreadPoolExecutor | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run Phase 3 Step 1 and Step 2 sequentially.

    Designed for parallel orchestration where Phase 2 and Phase 3
    run concurrently but internally execute their steps sequentially.

    Args:
        outputs_dir: Directory containing index.json and for output.
        force: If True, re-run even if outputs exist.
        executor: Optional shared ThreadPoolExecutor for LLM calls in Step 2.

    Returns:
        Tuple of (claim_heads, claim_mappings).
    """
    claim_heads = run_phase3_step1(outputs_dir=outputs_dir, force=force)
    claim_mappings = run_phase3_step2(
        outputs_dir=outputs_dir, force=force, executor=executor
    )
    return claim_heads, claim_mappings