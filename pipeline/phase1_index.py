"""Phase 1 — Index & Structured Extraction.

Processes each segment's OCR text through Gemini to extract structured
metadata (parties, dates, amounts, dispute signals, etc.) and writes
the results to outputs/index.json.

Supports incremental resume: if index.json already exists, only missing
segments are processed. Uses ThreadPoolExecutor for parallel extraction.

All Phase 1-specific LLM logic (prompts, post-processing) lives here.
The generic GeminiClient is imported from utils.llm.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from models.schemas import DOCUMENT_TYPE_ALIASES, LLMSegmentResponse, SegmentIndex
from utils.file_io import discover_segments, load_ocr_text, write_json_output
from utils.llm import GeminiClient

logger = logging.getLogger(__name__)

# Hard cap on parallel Gemini calls. Keep low to avoid rate limits.
MAX_WORKERS = 5

# Model used for Phase 1 extraction
PHASE1_MODEL = "gemini-2.0-flash"

# ---------------------------------------------------------------------------
# Phase 1 prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a legal document metadata extraction agent for construction arbitration cases.

You receive raw OCR text from scanned documents. The OCR may contain noise, \
artifacts, misspellings, and formatting errors — work around them.

Extract structured metadata strictly from what is present in the text.
Do NOT infer, guess, or hallucinate any information not supported by the document text.
Do NOT include a segment_id field in your response. It will be assigned programmatically.

IMPORTANT: Extract ALL organizations and individuals mentioned in the document, \
including the employer, government authority, issuing department, and any other party. \
Do not extract only the sender or the primary party.\
"""

USER_PROMPT_TEMPLATE = """\
Extract structured metadata from this construction arbitration document.

Rules:

1. document_type: Use the closest match from this preferred set:
   "Invoice", "Letter", "Claim Statement", "Government Order",
   "Contract Agreement", "Work Order", "Court Filing", "Petition",
   "Board Resolution", "Notice", "Application", "Certificate",
   "Office Memorandum", "Supplementary Work Order", "Report",
   "Abstract Statement", "Annexure", "Completion Certificate".
   Use "Invoice" not "Bill". Use the closest standard label.

2. document_stage: Must be EXACTLY one of:
   "Contract Formation", "Execution", "Delay/EOT",
   "Financial/Billing", "Legal Escalation", "Security/BG"

3. parties: Extract ALL organizations and individuals mentioned,
   including employer, contractor, government authority, issuing
   department, legal counsel, and sub-contractors.
   Assign roles like: "contractor", "employer", "government authority",
   "legal counsel", "sub-contractor", "project authority".
   Do NOT extract only the sender — include all parties referenced.

4. dates:
Extract only legally significant dates relevant to the dispute, such as:
- Work order date
- Commencement date
- Completion date
- Suspension date
- Notice date
- Claim date
- Termination date
- Award/order date

Do NOT extract every referenced correspondence date or repeated historical references.

Limit to a maximum of 10 dates.

Normalize to DD-MM-YYYY format. Handle these input formats:
  DD.MM.YYYY, DD-MM-YYYY, DD/MM/YY, DD/MM/YYYY,
  Month DD YYYY, D.M.YYYY, abbreviated months.

If the year is missing, unreadable, or ambiguous, set normalized = null.
Do NOT guess missing digits.

5. monetary_amounts:
   Extract only legally significant monetary amounts such as:
   - Total contract value
   - Total claim amount
   - Total escalation amount
   - Released amount
   - Balance amount
   - LD amount
   - Compensation amount
   - Interest amounts
   - Total deviation/extra item value
   Do NOT extract repetitive line-item breakdowns, per-quarter breakdowns,
   material/labour splits, or individual BOQ rows unless they are
   explicitly the final total or legally significant.
   Limit monetary_amounts to a maximum of 25 entries.
   - Preserve raw_text EXACTLY as it appears in the OCR.
   - Convert the numeric value conservatively to a float.
   - If the OCR number formatting is corrupted, ambiguous, or has
     missing/extra digits (e.g. "3,34,59,11500", "1.3.00"),
     set amount = null. Do NOT guess or reconstruct the number.
   - Do NOT infer missing decimal points or reconstruct commas.
   - If the context of a monetary value is unclear, use:
     "Context not specified in document" as the description.
     Do NOT use vague labels like "Unclear amount".

6. summary: Provide a concise factual summary of the document’s purpose and key claims. Maximum 100 words. Do not exceed 3 sentences. Do not include quotations, detailed allegations, or procedural history. Only describe the document’s main intent and primary financial or legal issue.

7. dispute_signals:
Identify only the most relevant dispute themes present in the document.

signal_type must be one of:
delay, escalation, payment_dispute, EOT, LD, BG,
covid, force_majeure, extra_item, interest,
arbitration, compensation.

Limit to a maximum of 5 dispute signals.
Do not include redundant or overlapping themes.
Include a brief one-line explanation of why the signal was identified.

OCR TEXT:
{ocr_text}"""

# ---------------------------------------------------------------------------
# Phase 1 post-processing
# ---------------------------------------------------------------------------

# Values above this are almost always OCR mis-parses (Indian numbering
# corruption), e.g. "3,34,59,11500" → 3_345_911_500.
_SUSPICIOUS_AMOUNT_THRESHOLD = 1_000_000_000


def _post_process(segment: SegmentIndex) -> SegmentIndex:
    """Apply lightweight fixes after LLM extraction.

    1. Normalize document_type via alias mapping.
    2. Warn on suspiciously large monetary amounts.

    Modifies the segment in place and returns it.
    """
    canonical = DOCUMENT_TYPE_ALIASES.get(segment.document_type)
    if canonical:
        logger.debug(
            "Normalized document_type '%s' → '%s' for %s",
            segment.document_type,
            canonical,
            segment.segment_id,
        )
        segment.document_type = canonical

    for ma in segment.monetary_amounts:
        if ma.amount is not None and ma.amount > _SUSPICIOUS_AMOUNT_THRESHOLD:
            logger.warning(
                "Suspicious amount in %s: %s → %.2f (raw: '%s')",
                segment.segment_id,
                ma.description,
                ma.amount,
                ma.raw_text,
            )

    # Cap monetary amounts to prevent JSON truncation
    if len(segment.monetary_amounts) > 25:
        logger.warning(
            "Capping monetary_amounts from %d to 25 for %s",
            len(segment.monetary_amounts),
            segment.segment_id,
        )
        segment.monetary_amounts = segment.monetary_amounts[:25]

    return segment


# ---------------------------------------------------------------------------
# Phase 1 extraction (per-segment)
# ---------------------------------------------------------------------------

# OCR truncation thresholds for very large documents
_MAX_OCR_CHARS = 22000
_HEAD_CHARS = 15000
_TAIL_CHARS = 7000


def extract_segment_index(
    client: GeminiClient,
    ocr_text: str,
    segment_id: str,
) -> SegmentIndex:
    """Extract structured metadata from a single segment's OCR text.

    Calls the generic GeminiClient with Phase 1-specific prompts and
    schema, validates via Pydantic, injects segment_id, and applies
    post-processing.

    Args:
        client: Initialized GeminiClient instance.
        ocr_text: Cleaned OCR text for the segment.
        segment_id: Folder name to set on the result.

    Returns:
        A validated SegmentIndex with segment_id set.

    Raises:
        RuntimeError: If extraction fails after all retries.
    """
    # Truncate very large documents to prevent output token overflow
    if len(ocr_text) > _MAX_OCR_CHARS:
        logger.warning(
            "Truncating OCR for %s: %d chars → %d (head=%d, tail=%d)",
            segment_id,
            len(ocr_text),
            _MAX_OCR_CHARS,
            _HEAD_CHARS,
            _TAIL_CHARS,
        )
        ocr_text = (
            ocr_text[:_HEAD_CHARS]
            + "\n\n--- TRUNCATED MIDDLE ---\n\n"
            + ocr_text[-_TAIL_CHARS:]
        )

    user_prompt = USER_PROMPT_TEMPLATE.format(ocr_text=ocr_text)

    raw_json = client.generate_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=LLMSegmentResponse.model_json_schema(),
        model=PHASE1_MODEL,
    )

    llm_result = LLMSegmentResponse.model_validate(raw_json)

    segment = SegmentIndex(
        segment_id=segment_id,
        **llm_result.model_dump(),
    )

    return _post_process(segment)


# ---------------------------------------------------------------------------
# Thread pool helper
# ---------------------------------------------------------------------------


def _extract_one(
    client: GeminiClient,
    segments_dir: Path,
    segment_id: str,
) -> SegmentIndex | None:
    """Extract metadata for a single segment. Returns None on failure.

    This function is submitted to the thread pool. It catches all
    exceptions so one bad segment never crashes the pool.
    """
    start = time.perf_counter()
    try:
        ocr_text = load_ocr_text(segments_dir / segment_id)
        result = extract_segment_index(client, ocr_text, segment_id)
        elapsed = time.perf_counter() - start
        logger.info("Done: %s (%.1fs)", segment_id, elapsed)
        return result

    except FileNotFoundError:
        logger.error("OCR file missing for %s — skipping.", segment_id)
        return None

    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.error("Failed: %s (%.1fs): %s", segment_id, elapsed, exc)
        return None


def _load_existing(output_path: Path) -> list[SegmentIndex]:
    """Load previously extracted segments from index.json, if it exists."""
    if not output_path.exists():
        return []

    try:
        raw = json.loads(output_path.read_text(encoding="utf-8"))
        existing = [SegmentIndex.model_validate(s) for s in raw]
        logger.info("Loaded %d existing segments from %s", len(existing), output_path)
        return existing
    except Exception as exc:
        logger.warning("Could not load existing index (%s) — starting fresh.", exc)
        return []


def run_phase1(
    segments_dir: Path,
    output_path: Path,
    *,
    force: bool = False,
) -> list[SegmentIndex]:
    """Run Phase 1: extract structured metadata from all segments.

    Incrementally resumes by loading existing results and only
    processing segments not yet in the output file.

    Args:
        segments_dir: Path to the segments/ directory.
        output_path: Path to write outputs/index.json.
        force: If True, ignore existing output and reprocess everything.

    Returns:
        List of validated SegmentIndex objects (existing + new).
    """
    # ------------------------------------------------------------------
    # Load existing results (unless forced)
    # ------------------------------------------------------------------
    existing: list[SegmentIndex] = []
    if not force:
        existing = _load_existing(output_path)

    done_ids: set[str] = {s.segment_id for s in existing}

    # ------------------------------------------------------------------
    # Discover segments & compute pending
    # ------------------------------------------------------------------
    all_ids = discover_segments(segments_dir)
    if not all_ids:
        logger.error("No segments found in %s", segments_dir)
        return existing

    pending = [sid for sid in all_ids if sid not in done_ids]

    if not pending:
        logger.info("All %d segments already processed — nothing to do.", len(all_ids))
        return existing

    logger.info(
        "Phase 1: %d total, %d already done, %d pending (workers=%d)",
        len(all_ids),
        len(done_ids),
        len(pending),
        MAX_WORKERS,
    )

    # ------------------------------------------------------------------
    # Process pending segments in parallel
    # ------------------------------------------------------------------
    client = GeminiClient()
    new_results: list[SegmentIndex] = []
    failures: list[str] = []
    phase_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_id = {
            pool.submit(_extract_one, client, segments_dir, sid): sid
            for sid in pending
        }

        for i, future in enumerate(as_completed(future_to_id), 1):
            segment_id = future_to_id[future]
            result = future.result()  # _extract_one never raises

            if result is not None:
                new_results.append(result)
            else:
                failures.append(segment_id)

            logger.info(
                "Progress: %d/%d pending complete", i, len(pending)
            )

    # ------------------------------------------------------------------
    # Merge, sort, write
    # ------------------------------------------------------------------
    all_results = existing + new_results
    all_results.sort(key=lambda s: s.segment_id)

    phase_elapsed = time.perf_counter() - phase_start

    data = [s.model_dump() for s in all_results]
    write_json_output(data, output_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info(
        "Phase 1 complete: %d new succeeded, %d failed, %d total in index, %.1fs",
        len(new_results),
        len(failures),
        len(all_results),
        phase_elapsed,
    )
    if failures:
        logger.warning("Failed segments: %s", ", ".join(sorted(failures)))

    return all_results
