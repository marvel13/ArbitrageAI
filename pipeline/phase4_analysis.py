"""Phase 4 — Case-Level Reasoning & Report.

Assembles structured outputs from Phases 1-3, pre-computes analytical
data (timeline, coverage matrix, per-claim statistics) in pure Python,
then makes focused, section-specific LLM calls to produce a grounded
case analysis report.

The generic GeminiClient is imported from utils.llm.
All LLM calls use schema-enforced JSON mode via ``generate_json``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

from models.schemas import LLMMarkdownSection
from utils.llm import GeminiClient

logger = logging.getLogger(__name__)

# -- Config ----------------------------------------------------------------- #

MODEL = "gemini-2.5-pro"

# -- Data Loading ----------------------------------------------------------- #


def _load_json(path: Path) -> list[dict]:
    """Load a JSON array from *path*."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_all_inputs(
    outputs_dir: Path,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Load and return (index, type_classifications, claim_heads, claim_classifications).

    Raises:
        FileNotFoundError: If any required input file is missing.
    """
    index = _load_json(outputs_dir / "index.json")
    type_cls = _load_json(outputs_dir / "type_classifications.json")
    claim_heads = _load_json(outputs_dir / "claim_heads.json")
    claim_cls = _load_json(outputs_dir / "claim_classifications.json")

    logger.info(
        "Loaded inputs: %d segments, %d type classifications, "
        "%d claim heads, %d claim mappings",
        len(index),
        len(type_cls),
        len(claim_heads),
        len(claim_cls),
    )
    return index, type_cls, claim_heads, claim_cls


# -- Pre-Computation (pure Python, no LLM) --------------------------------- #


def _extract_case_facts(index: list[dict]) -> dict:
    """Derive high-level case facts from the segment index."""
    all_parties: dict[str, set[str]] = defaultdict(set)
    all_dates: list[str] = []
    total_monetary = 0.0

    for seg in index:
        for p in seg.get("parties", []):
            all_parties[p.get("role", "unknown")].add(p["name"])
        for d in seg.get("dates", []):
            if d.get("normalized"):
                all_dates.append(d["normalized"])
        for m in seg.get("monetary_amounts", []):
            if m.get("amount"):
                total_monetary += m["amount"]

    # Convert sets to sorted lists for JSON-friendliness
    parties_summary = {
        role: sorted(names) for role, names in sorted(all_parties.items())
    }

    sorted_dates = sorted(
        all_dates,
        key=lambda d: tuple(reversed(d.split("-"))),  # DD-MM-YYYY → (YYYY, MM, DD)
    )

    return {
        "total_segments": len(index),
        "parties": parties_summary,
        "date_range": {
            "earliest": sorted_dates[0] if sorted_dates else None,
            "latest": sorted_dates[-1] if sorted_dates else None,
        },
        "total_monetary_references": round(total_monetary, 2),
    }


def _build_document_landscape(
    index: list[dict],
    type_cls: list[dict],
) -> dict:
    """Compute per-cluster document distribution statistics."""
    seg_lookup = {s["segment_id"]: s for s in index}

    cluster_stats: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "segments": [], "doc_types": set()}
    )

    for cls in type_cls:
        sid = cls["segment_id"]
        primary = cls["primary_cluster"]
        seg_meta = seg_lookup.get(sid, {})

        cluster_stats[primary]["count"] += 1
        cluster_stats[primary]["segments"].append(sid)
        cluster_stats[primary]["doc_types"].add(seg_meta.get("document_type", "Unknown"))

        for secondary in cls.get("secondary_clusters", []):
            cluster_stats[secondary]["segments"].append(f"{sid} (secondary)")

    # Serialize sets → sorted lists
    result = {}
    for cluster_key, stats in sorted(cluster_stats.items()):
        result[cluster_key] = {
            "primary_count": stats["count"],
            "total_references": len(stats["segments"]),
            "segments": stats["segments"],
            "document_types": sorted(stats["doc_types"]),
        }

    return result


# -- Timeline: simple chronological collection -------------------------------- #

def _date_sort_key(date_str: str) -> tuple[str, ...]:
    """Convert DD-MM-YYYY to (YYYY, MM, DD) for chronological sorting."""
    parts = date_str.split("-")
    if len(parts) == 3:
        return (parts[2], parts[1], parts[0])
    return (date_str,)


def _build_timeline(index: list[dict]) -> list[dict]:
    """Build a simple chronological timeline of all documented dates.

    Collects all normalized dates from segments, deduplicates on
    (date, segment_id), and sorts chronologically.

    Returns a list of ``{date, event, segment_id}`` dicts sorted
    chronologically.
    """
    raw_events: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    for seg in index:
        sid = seg["segment_id"]

        for d in seg.get("dates", []):
            norm = d.get("normalized")
            if not norm:
                continue
            context = d.get("context", "")

            # Deduplicate on (date, segment_id)
            key = (norm, sid)
            if key in seen:
                continue
            seen.add(key)

            raw_events.append((norm, context, sid))

    # Sort chronologically
    raw_events.sort(key=lambda e: _date_sort_key(e[0]))

    return [
        {"date": e[0], "event": e[1], "segment_id": e[2]}
        for e in raw_events
    ]


def _build_claim_dossiers(
    index: list[dict],
    claim_heads: list[dict],
    claim_cls: list[dict],
) -> dict[str, dict]:
    """Pre-compute a rich analytical dossier for each claim head.

    Each dossier includes mapping statistics, supporting evidence details,
    monetary references, and rebuttal indicators.
    """
    seg_lookup = {s["segment_id"]: s for s in index}

    # Group mappings by claim_key
    mappings_by_claim: dict[str, list[dict]] = defaultdict(list)
    for m in claim_cls:
        mappings_by_claim[m["claim_key"]].append(m)

    dossiers: dict[str, dict] = {}

    for ch in claim_heads:
        ck = ch["claim_key"]
        maps = mappings_by_claim.get(ck, [])

        # Relevance type counts
        relevance_counts = defaultdict(int)
        party_counts = defaultdict(int)
        confidences: list[float] = []
        supporting_evidence: list[dict] = []

        for m in maps:
            relevance_counts[m["relevance_type"]] += 1
            party_counts[m["party_role"]] += 1
            confidences.append(m["confidence"])
            seg_meta = seg_lookup.get(m["segment_id"], {})
            supporting_evidence.append({
                "segment_id": m["segment_id"],
                "relevance_type": m["relevance_type"],
                "party_role": m["party_role"],
                "confidence": m["confidence"],
                "reasoning": m["reasoning"],
                "document_type": seg_meta.get("document_type", "Unknown"),
                "summary": seg_meta.get("summary", ""),
                "monetary_amounts": seg_meta.get("monetary_amounts", []),
            })

        # Collect all monetary refs from supporting segments
        monetary_refs: list[dict] = []
        for ev in supporting_evidence:
            for ma in ev.get("monetary_amounts", []):
                if ma.get("amount"):
                    monetary_refs.append({
                        "amount": ma["amount"],
                        "description": ma.get("description", ""),
                        "segment_id": ev["segment_id"],
                    })

        dossiers[ck] = {
            "claim_head": ch,
            "total_mappings": len(maps),
            "relevance_counts": dict(relevance_counts),
            "party_counts": dict(party_counts),
            "mean_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0,
            "min_confidence": round(min(confidences), 3) if confidences else 0,
            "has_rebuttal": relevance_counts.get("rebuttal", 0) > 0,
            "supporting_evidence": supporting_evidence,
            "monetary_references": monetary_refs,
        }

    return dossiers


def _build_coverage_matrix(
    index: list[dict],
    claim_heads: list[dict],
    claim_cls: list[dict],
) -> dict:
    """Build a segment × claim_head coverage matrix.

    Returns:
        dict with ``matrix`` (segment_id → claim_key → relevance_type),
        ``unmapped_segments`` (segments with 0 claim mappings),
        ``coverage_summary`` (per-claim coverage stats).
    """
    claim_keys = [ch["claim_key"] for ch in claim_heads]
    all_segment_ids = [s["segment_id"] for s in index]

    # Build the grid
    matrix: dict[str, dict[str, str | None]] = {
        sid: {ck: None for ck in claim_keys} for sid in all_segment_ids
    }

    segment_mapping_count: dict[str, int] = defaultdict(int)
    claim_mapping_count: dict[str, int] = defaultdict(int)
    claim_direct_count: dict[str, int] = defaultdict(int)

    for m in claim_cls:
        sid = m["segment_id"]
        ck = m["claim_key"]
        if sid in matrix and ck in matrix[sid]:
            matrix[sid][ck] = m["relevance_type"]
        segment_mapping_count[sid] += 1
        claim_mapping_count[ck] += 1
        if m["relevance_type"] == "direct":
            claim_direct_count[ck] += 1

    unmapped = [sid for sid in all_segment_ids if segment_mapping_count[sid] == 0]

    coverage_summary: dict[str, dict] = {}
    for ck in claim_keys:
        mapped = claim_mapping_count.get(ck, 0)
        direct = claim_direct_count.get(ck, 0)
        coverage_summary[ck] = {
            "total_mapped": mapped,
            "direct_mapped": direct,
            "coverage_pct": round(mapped / len(all_segment_ids) * 100, 1),
        }

    return {
        "matrix": matrix,
        "unmapped_segments": unmapped,
        "coverage_summary": coverage_summary,
        "claim_keys": claim_keys,
        "segment_ids": all_segment_ids,
    }


def _identify_gaps(
    dossiers: dict[str, dict],
    coverage: dict,
) -> list[dict]:
    """Identify documentary gaps and weak spots across the case.

    Returns a list of gap indicators with type, description, and affected entities.
    """
    gaps: list[dict] = []

    # Unmapped segments
    if coverage["unmapped_segments"]:
        gaps.append({
            "type": "unmapped_segments",
            "description": "Segments with no claim mapping — potential unused evidence",
            "affected": coverage["unmapped_segments"],
        })

    for ck, dossier in dossiers.items():
        title = dossier["claim_head"]["title"]

        # Claims with sparse direct support
        direct = dossier["relevance_counts"].get("direct", 0)
        if direct < 3:
            gaps.append({
                "type": "sparse_direct_support",
                "description": f"'{title}' has only {direct} direct supporting segments",
                "affected": [ck],
            })

        # Claims with low-confidence support
        if dossier["mean_confidence"] < 0.8:
            gaps.append({
                "type": "low_confidence_support",
                "description": (
                    f"'{title}' has mean mapping confidence of "
                    f"{dossier['mean_confidence']:.2f}"
                ),
                "affected": [ck],
            })

        # Claims with no monetary evidence in mapped segments
        if not dossier["monetary_references"]:
            gaps.append({
                "type": "no_monetary_evidence",
                "description": (
                    f"'{title}' has no monetary amounts in its mapped segments"
                ),
                "affected": [ck],
            })

        # Claims with no rebuttal (may indicate one-sided view)
        if not dossier["has_rebuttal"] and dossier["total_mappings"] > 3:
            gaps.append({
                "type": "no_rebuttal_evidence",
                "description": (
                    f"'{title}' has {dossier['total_mappings']} mappings but "
                    f"no rebuttal evidence — potentially one-sided documentary record"
                ),
                "affected": [ck],
            })

    return gaps


# -- LLM Section Generation ------------------------------------------------ #


def _generate_section(
    client: GeminiClient,
    *,
    section_name: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call LLM to generate a single report section as markdown.

    Args:
        client: Initialized GeminiClient instance.
        section_name: Human-readable name for logging.
        system_prompt: System instruction.
        user_prompt: Section-specific analytical prompt.

    Returns:
        Markdown string for this section.
    """
    start = time.perf_counter()
    raw = client.generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=LLMMarkdownSection.model_json_schema(),
        model=MODEL,
        temperature=0.3,
    )
    elapsed = time.perf_counter() - start

    result = LLMMarkdownSection.model_validate(raw)
    logger.info("Section '%s' generated in %.1fs", section_name, elapsed)
    return result.markdown


# System prompt shared by all Phase 4 LLM calls
_SYSTEM_PROMPT = """\
You are a senior construction arbitration analyst producing a case analysis \
report. You must:
- Write in professional, neutral legal analytical tone
- Ground every statement in the structured data provided — cite segment IDs \
  and claim keys where relevant
- NEVER introduce facts, amounts, or claims not present in the input data
- NEVER speculate beyond what the evidence supports
- Use markdown formatting (headers, bold, lists, tables) as appropriate
- Be concise but thorough — prioritise analytical insight over description\
"""


def _gen_case_overview(
    client: GeminiClient,
    case_facts: dict,
    claim_heads: list[dict],
) -> str:
    """Generate Section 1: Case Overview."""
    claims_summary = []
    for ch in claim_heads:
        claims_summary.append({
            "claim_key": ch["claim_key"],
            "title": ch["title"],
            "claimant": ch["claimant"],
            "approximate_amount": ch.get("approximate_claimed_amount"),
        })

    user_prompt = f"""\
Generate a **Case Overview** section (200-300 words) for a construction \
arbitration analysis report.

**Case Facts:**
{json.dumps(case_facts, indent=2)}

**Claim Heads Summary:**
{json.dumps(claims_summary, indent=2)}

Requirements:
- Identify the core dispute and its nature (construction, delay, payment, etc.)
- Name the principal parties and their roles
- Characterize the financial stakes by referencing the claim heads
- Note both contractor claims and employer counterclaims
- Do NOT list every claim head — synthesize the dispute's overall character
- End with a sentence on the dispute's current stage (arbitration/litigation)\
"""
    return _generate_section(
        client, section_name="Case Overview",
        system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt,
    )


def _gen_document_landscape(
    client: GeminiClient,
    landscape: dict,
    case_facts: dict,
) -> str:
    """Generate Section 2: Document Landscape."""
    user_prompt = f"""\
Generate a **Document Landscape** section (200-300 words) analyzing the \
documentary composition of this arbitration.

**Document Distribution by Taxonomy Cluster:**
{json.dumps(landscape, indent=2)}

**Case Facts:**
Total segments: {case_facts['total_segments']}

Requirements:
- Analyze what the document distribution reveals about each party's \
  documentary strategy
- Identify which project lifecycle stages are well-documented vs \
  under-documented
- Note any imbalance between claimant and respondent documentation
- Highlight clusters with unusually high or low representation
- Draw analytical conclusions about evidence preparation quality\
"""
    return _generate_section(
        client, section_name="Document Landscape",
        system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt,
    )


def _gen_claim_analysis(
    client: GeminiClient,
    claim_key: str,
    dossier: dict,
) -> str:
    """Generate a claim head analysis sub-section for one claim."""
    ch = dossier["claim_head"]

    # Condense supporting evidence for the prompt (avoid bloating)
    evidence_summary = []
    for ev in dossier["supporting_evidence"]:
        evidence_summary.append({
            "segment_id": ev["segment_id"],
            "document_type": ev["document_type"],
            "relevance_type": ev["relevance_type"],
            "party_role": ev["party_role"],
            "confidence": ev["confidence"],
            "reasoning": ev["reasoning"],
            "summary": ev["summary"][:200],  # Truncate to keep prompt lean
        })

    user_prompt = f"""\
Generate a detailed analysis (200-350 words) of the following claim head \
for a construction arbitration report.

**Claim Head:**
- Key: {ch['claim_key']}
- Title: {ch['title']}
- Description: {ch['description']}
- Claimant: {ch['claimant']}
- Approximate Amount: {ch.get('approximate_claimed_amount', 'Not quantified')}
- Sub-heads: {json.dumps(ch.get('sub_heads'), indent=2) if ch.get('sub_heads') else 'None'}

**Evidence Statistics:**
- Total mapped segments: {dossier['total_mappings']}
- Direct: {dossier['relevance_counts'].get('direct', 0)}, \
Contextual: {dossier['relevance_counts'].get('contextual', 0)}, \
Rebuttal: {dossier['relevance_counts'].get('rebuttal', 0)}
- Party support: {json.dumps(dossier['party_counts'])}
- Mean confidence: {dossier['mean_confidence']}, \
Min confidence: {dossier['min_confidence']}

**Supporting Evidence:**
{json.dumps(evidence_summary, indent=2)}

**Monetary References from Mapped Segments:**
{json.dumps(dossier['monetary_references'][:15], indent=2)}

Requirements:
- Assess the documentary strength of this claim
- Rate the evidence chain: strong / moderate / weak — with justification
- Identify the strongest pieces of evidence (cite segment IDs)
- Identify the weakest links or gaps in the evidence chain
- Note any contradictions between supporting and rebuttal documents
- Evaluate how well the claimed amount is quantified/documented
- If there are sub-heads, briefly assess coverage of each
- Do NOT restate the data — provide analytical interpretation\
"""
    return _generate_section(
        client, section_name=f"Claim: {claim_key}",
        system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt,
    )


def _gen_observations_and_risks(
    client: GeminiClient,
    gaps: list[dict],
    coverage_summary: dict,
    dossier_summaries: list[dict],
) -> str:
    """Generate Section 6: Observations, Gaps & Risk Areas."""
    user_prompt = f"""\
Generate an **Observations, Gaps & Risk Areas** section (300-450 words) \
for a construction arbitration analysis report.

**Identified Documentary Gaps:**
{json.dumps(gaps, indent=2)}

**Coverage Summary (per claim head):**
{json.dumps(coverage_summary, indent=2)}

**Claim Dossier Summaries:**
{json.dumps(dossier_summaries, indent=2)}

Requirements:
- Organize observations under clear sub-headings
- Identify specific documentary gaps that weaken particular claims \
  (cite claim keys and segment IDs)
- Assess which claims are most vulnerable to challenge and why
- Note areas where the opposing party has stronger documentary support
- Identify potential risk areas for each side (contractor and employer)
- Flag any procedural or evidentiary concerns (e.g., one-sided record, \
  missing rebuttals, unquantified claims)
- Suggest what additional documentary evidence would strengthen weak claims
- Be actionable and specific — avoid generic observations\
"""
    return _generate_section(
        client, section_name="Observations & Risks",
        system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt,
    )


# -- Timeline & Coverage Matrix (pure computation) ------------------------- #


def _render_timeline_markdown(timeline: list[dict]) -> str:
    """Render the timeline as a simple chronological markdown table."""
    if not timeline:
        return "*No dates could be extracted from the corpus.*"

    lines: list[str] = [
        "| Date | Event | Source |",
        "|------|-------|--------|"]

    for entry in timeline:
        event = entry["event"] or "—"
        lines.append(f"| {entry['date']} | {event} | {entry['segment_id']} |")

    return "\n".join(lines)


def _render_coverage_matrix_markdown(coverage: dict) -> str:
    """Render the segment × claim coverage matrix as a markdown table.

    Uses symbols: ● direct, ◐ contextual, ◯ rebuttal, · no mapping.
    """
    claim_keys = coverage["claim_keys"]
    matrix = coverage["matrix"]
    segment_ids = coverage["segment_ids"]

    # Abbreviate claim keys for header
    abbrevs = [ck.replace("_", " ").title()[:20] for ck in claim_keys]

    symbol_map = {
        "direct": "●",
        "contextual": "◐",
        "rebuttal": "◯",
        None: "·",
    }

    lines = [
        "Legend: ● direct, ◐ contextual, ◯ rebuttal, · no mapping",
        "",
        "| Segment | " + " | ".join(abbrevs) + " |",
        "|" + "---|" * (len(claim_keys) + 1),
    ]

    for sid in segment_ids:
        # Truncate long segment IDs
        sid_display = sid[:45] + "…" if len(sid) > 45 else sid
        row_cells = []
        for ck in claim_keys:
            cell_val = matrix.get(sid, {}).get(ck)
            row_cells.append(symbol_map.get(cell_val, "·"))
        lines.append(f"| {sid_display} | " + " | ".join(row_cells) + " |")

    # Summary row
    summary = coverage["coverage_summary"]
    lines.append("| **Total Mapped** | " + " | ".join(
        str(summary[ck]["total_mapped"]) for ck in claim_keys
    ) + " |")
    lines.append("| **Direct** | " + " | ".join(
        str(summary[ck]["direct_mapped"]) for ck in claim_keys
    ) + " |")
    lines.append("| **Coverage %** | " + " | ".join(
        f"{summary[ck]['coverage_pct']}%" for ck in claim_keys
    ) + " |")

    return "\n".join(lines)


# -- Report Assembly -------------------------------------------------------- #


def _assemble_report(sections: dict[str, str]) -> str:
    """Stitch all sections into the final case_analysis.md content."""
    parts = [
        "# Construction Arbitration — Case Analysis Report",
        "",
        "---",
        "",
        "## 1. Case Overview",
        "",
        sections["case_overview"],
        "",
        "---",
        "",
        "## 2. Document Landscape",
        "",
        sections["document_landscape"],
        "",
        "---",
        "",
        "## 3. Timeline of Key Events",
        "",
        sections["timeline"],
        "",
        "---",
        "",
        "## 4. Claim Head Analysis",
        "",
        sections["claim_analysis"],
        "",
        "---",
        "",
        "## 5. Evidence Coverage Matrix",
        "",
        sections["coverage_matrix"],
        "",
        "---",
        "",
        "## 6. Observations, Gaps & Risk Areas",
        "",
        sections["observations_risks"],
        "",
        "---",
        "",
        "*Report generated by the Construction Arbitration Multi-Agent Pipeline.*",
    ]
    return "\n".join(parts)


# -- Entry Point ------------------------------------------------------------ #


def run_phase4(outputs_dir: Path, *, force: bool = False) -> Path:
    """Execute Phase 4: generate the case-level analysis report.

    Args:
        outputs_dir: Directory containing all prior phase outputs.
        force: If True, regenerate even if output exists.

    Returns:
        Path to the generated case_analysis.md file.
    """
    output_path = outputs_dir / "case_analysis.md"
    phase_start = time.perf_counter()

    # -- Checkpoint --------------------------------------------------------- #
    if output_path.exists() and not force:
        logger.info("Phase 4 output already exists at %s — skipping", output_path)
        return output_path

    logger.info("=== Phase 4: Case-Level Reasoning & Report ===")

    # -- Load inputs -------------------------------------------------------- #
    index, type_cls, claim_heads, claim_cls = _load_all_inputs(outputs_dir)

    # -- Pre-computation (no LLM) ------------------------------------------ #
    logger.info("Pre-computing analytical data...")
    precomp_start = time.perf_counter()

    case_facts = _extract_case_facts(index)
    landscape = _build_document_landscape(index, type_cls)
    timeline = _build_timeline(index)
    dossiers = _build_claim_dossiers(index, claim_heads, claim_cls)
    coverage = _build_coverage_matrix(index, claim_heads, claim_cls)
    gaps = _identify_gaps(dossiers, coverage)

    logger.info(
        "Pre-computation complete in %.1fs: %d timeline events, %d gaps identified",
        time.perf_counter() - precomp_start,
        len(timeline),
        len(gaps),
    )

    # -- Render computed sections (no LLM) ---------------------------------- #
    timeline_md = _render_timeline_markdown(timeline)
    coverage_md = _render_coverage_matrix_markdown(coverage)

    # -- LLM section generation --------------------------------------------- #
    client = GeminiClient()
    sections: dict[str, str] = {}

    # Section 1: Case Overview
    logger.info("Generating Section 1: Case Overview...")
    sections["case_overview"] = _gen_case_overview(client, case_facts, claim_heads)

    # Section 2: Document Landscape
    logger.info("Generating Section 2: Document Landscape...")
    sections["document_landscape"] = _gen_document_landscape(
        client, landscape, case_facts,
    )

    # Section 3: Timeline (pre-computed)
    sections["timeline"] = timeline_md

    # Section 4: Claim Head Analysis (one call per claim, sequential)
    logger.info("Generating Section 4: Claim Head Analysis (%d claims)...", len(claim_heads))
    claim_sections: dict[str, str] = {}

    for ck in dossiers:
        claim_sections[ck] = _gen_claim_analysis(client, ck, dossiers[ck])

    # Assemble claim sections in original order
    claim_analysis_parts: list[str] = []
    for ch in claim_heads:
        ck = ch["claim_key"]
        claim_analysis_parts.append(f"### 4.{claim_heads.index(ch) + 1}. {ch['title']}")
        claim_analysis_parts.append("")
        claim_analysis_parts.append(claim_sections.get(ck, "*Not available.*"))
        claim_analysis_parts.append("")
    sections["claim_analysis"] = "\n".join(claim_analysis_parts)

    # Section 5: Coverage Matrix (pre-computed)
    sections["coverage_matrix"] = coverage_md

    # Section 6: Observations, Gaps & Risk Areas
    logger.info("Generating Section 6: Observations, Gaps & Risk Areas...")
    dossier_summaries = []
    for ck, d in dossiers.items():
        dossier_summaries.append({
            "claim_key": ck,
            "title": d["claim_head"]["title"],
            "claimant": d["claim_head"]["claimant"],
            "total_mappings": d["total_mappings"],
            "direct_support": d["relevance_counts"].get("direct", 0),
            "has_rebuttal": d["has_rebuttal"],
            "mean_confidence": d["mean_confidence"],
            "monetary_evidence_count": len(d["monetary_references"]),
        })

    sections["observations_risks"] = _gen_observations_and_risks(
        client, gaps, coverage["coverage_summary"], dossier_summaries,
    )

    # -- Assemble & write -------------------------------------------------- #
    report = _assemble_report(sections)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")
    tmp_path.write_text(report, encoding="utf-8")
    os.replace(tmp_path, output_path)

    elapsed = time.perf_counter() - phase_start
    logger.info(
        "Phase 4 complete — wrote %s (%.1f KB) in %.1fs",
        output_path,
        len(report) / 1024,
        elapsed,
    )

    return output_path
