"""Pydantic v2 schemas for the arbitration pipeline.

These models define the data contracts between pipeline phases.
Phase 1 produces a list of SegmentIndex objects written to outputs/index.json.
"""

from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Constrained types                                                           #
# --------------------------------------------------------------------------- #

# Allowed values for document_stage — must match prompt instructions exactly.
DocumentStage = Literal[
    "Contract Formation",
    "Execution",
    "Delay/EOT",
    "Financial/Billing",
    "Legal Escalation",
    "Security/BG",
]

# Preferred document_type labels. Used for post-extraction normalization.
# Maps common LLM variants → canonical label.
DOCUMENT_TYPE_ALIASES: dict[str, str] = {
    "Bill": "Invoice",
    "Escalation Bill": "Invoice",
    "Demand Notice": "Notice",
    "Legal Notice": "Notice",
    "Writ Petition": "Petition",
    "Office Memo": "Office Memorandum",
    "Memo": "Office Memorandum",
    "Agreement": "Contract Agreement",
}

# --------------------------------------------------------------------------- #
# Sub-models (used inside SegmentIndex)                                       #
# --------------------------------------------------------------------------- #


class Party(BaseModel):
    """An organization or individual mentioned in the document."""

    name: str = Field(description="Full name of the party")
    role: str = Field(description="Role in the dispute, e.g. 'contractor', 'employer', 'legal counsel'")


class DateEntry(BaseModel):
    """A date extracted from the document."""

    date_string: str = Field(description="Date as it appears in the OCR text")
    normalized: str | None = Field(
        default=None,
        description="Normalized date in DD-MM-YYYY format, or null if ambiguous",
    )
    context: str = Field(description="What this date refers to, e.g. 'work order date'")


class MonetaryAmount(BaseModel):
    """A monetary value extracted from the document."""

    amount: float | None = Field(
        default=None,
        description="Numeric value, or null if OCR formatting is corrupted/ambiguous",
    )
    currency: str = Field(default="INR", description="Currency code, defaults to INR")
    description: str = Field(description="What the amount represents")
    raw_text: str = Field(description="Original text string as found in the OCR")


class DisputeSignal(BaseModel):
    """A tag indicating a dispute theme present in the document."""

    signal_type: str = Field(
        description=(
            "One of: delay, escalation, payment_dispute, EOT, LD, BG, "
            "covid, force_majeure, extra_item, interest, arbitration, compensation"
        ),
    )
    description: str = Field(description="Brief explanation of why this signal was identified")


# --------------------------------------------------------------------------- #
# Per-segment output — the core Phase 1 schema                                #
# --------------------------------------------------------------------------- #


class SegmentIndex(BaseModel):
    """Structured metadata extracted from a single arbitration document segment.

    This is the per-segment output of Phase 1.
    `segment_id` is set programmatically — NOT by the LLM.
    """

    segment_id: str = Field(description="Folder name of the segment (set by code, not LLM)")
    document_type: str = Field(
        description="Type of document, e.g. 'Letter', 'Invoice', 'Contract Agreement'"
    )
    document_stage: DocumentStage = Field(
        description="Pipeline stage this document belongs to"
    )
    parties: list[Party] = Field(default_factory=list)
    dates: list[DateEntry] = Field(default_factory=list)
    monetary_amounts: list[MonetaryAmount] = Field(default_factory=list)
    summary: str = Field(description="2-4 sentence factual summary of the document")
    dispute_signals: list[DisputeSignal] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# LLM response schema — SegmentIndex minus segment_id                         #
# --------------------------------------------------------------------------- #


class LLMSegmentResponse(BaseModel):
    """Schema sent to Gemini's JSON mode. Identical to SegmentIndex but
    without segment_id, which is injected after the LLM call."""

    document_type: str = Field(
        description="Type of document, e.g. 'Letter', 'Invoice', 'Contract Agreement'"
    )
    document_stage: DocumentStage = Field(
        description="Pipeline stage this document belongs to"
    )
    parties: list[Party] = Field(default_factory=list)
    dates: list[DateEntry] = Field(default_factory=list)
    monetary_amounts: list[MonetaryAmount] = Field(default_factory=list)
    summary: str = Field(description="2-4 sentence factual summary of the document")
    dispute_signals: list[DisputeSignal] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Phase 2 — Document Type Taxonomy                                            #
# --------------------------------------------------------------------------- #


class TaxonomySubType(BaseModel):
    """A sub-type within a taxonomy cluster."""

    sub_type_key: str = Field(description="Unique key in format cluster_key/sub_type_slug")
    label: str = Field(description="Human-readable label for this sub-type")


class TaxonomyCluster(BaseModel):
    """A top-level cluster in the document taxonomy."""

    cluster_key: str = Field(description="Unique snake_case key for this cluster")
    label: str = Field(description="Human-readable label for this cluster")
    sub_types: list[TaxonomySubType] = Field(
        description="Sub-types within this cluster (1-5 items)"
    )


class LLMTaxonomyResponse(BaseModel):
    """Schema sent to Gemini for taxonomy generation."""

    clusters: list[TaxonomyCluster] = Field(
        description="List of 4-8 top-level document clusters"
    )


# --------------------------------------------------------------------------- #
# Phase 2 Step 2 — Classification                                             #
# --------------------------------------------------------------------------- #


class LLMClassificationResponse(BaseModel):
    """Schema sent to Gemini for segment classification."""

    primary_cluster: str = Field(description="Exactly one primary cluster key")
    secondary_clusters: list[str] = Field(
        default_factory=list,
        description="0-2 additional cluster keys (do not repeat primary)",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for primary_cluster between 0 and 1",
    )


class SegmentClassification(BaseModel):
    """Classification result for a single segment."""

    segment_id: str = Field(description="Folder name of the segment")
    primary_cluster: str = Field(description="Primary cluster key")
    secondary_clusters: list[str] = Field(
        default_factory=list,
        description="0-2 secondary cluster keys",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for primary_cluster",
    )


# --------------------------------------------------------------------------- #
# Phase 3 — Claim Head Discovery                                              #
# --------------------------------------------------------------------------- #


class ClaimSubHead(BaseModel):
    """Optional lightweight sub-head within a claim head.

    Used when a claim head has structural components (e.g., prolongation
    containing site overheads, machinery idle cost). No supporting IDs
    or amounts — purely descriptive.
    """

    sub_claim_key: str = Field(description="Unique key in format claim_key/sub_slug")
    title: str = Field(description="Human-readable title for this sub-head")
    description: str = Field(description="Brief description of this sub-head")


class ClaimHead(BaseModel):
    """A financially distinct claim head identified from the corpus.

    Represents a category of recovery being pursued by either the
    contractor (claim) or employer (counterclaim).
    """

    claim_key: str = Field(description="Unique snake_case key for this claim head")
    title: str = Field(description="Human-readable title for this claim head")
    description: str = Field(
        description="Description of the claim head and its basis in the dispute"
    )
    claimant: str = Field(
        description="Party pursuing this claim: 'contractor' or 'employer'"
    )
    approximate_claimed_amount: str | None = Field(
        default=None,
        description=(
            "Narrative estimate grounded in documents, e.g. "
            "'Approx. INR 6-7 crore based on claim statements'. "
            "Not a computed total."
        ),
    )
    supporting_segment_ids: list[str] = Field(
        description="Segment IDs that support this claim head (minimum 2 required)"
    )
    sub_heads: list[ClaimSubHead] | None = Field(
        default=None,
        description="Optional structural components of this claim head",
    )


class LLMClaimHeadsResponse(BaseModel):
    """Schema sent to Gemini for claim head discovery."""

    claim_heads: list[ClaimHead] = Field(
        description="List of 3-7 financially distinct claim heads discovered from corpus"
    )


# --------------------------------------------------------------------------- #
# Phase 3 Step 2 — Claim Mapping                                              #
# --------------------------------------------------------------------------- #

# Constrained types for claim mapping
RelevanceType = Literal["direct", "contextual", "rebuttal"]
PartyRole = Literal["supports_contractor", "supports_employer", "neutral"]


class ClaimMapping(BaseModel):
    """A single segment-to-claim mapping from LLM response."""

    claim_key: str = Field(description="Claim head key this segment relates to")
    relevance_type: RelevanceType = Field(
        description=(
            "How the segment relates: "
            "'direct' (evidences/quantifies), "
            "'contextual' (background/procedural), "
            "'rebuttal' (counters/disputes)"
        )
    )
    party_role: PartyRole = Field(
        description=(
            "Who the document supports: "
            "'supports_contractor', 'supports_employer', or 'neutral'"
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for this mapping (0.0-1.0)",
    )
    reasoning: str = Field(
        description="1-2 sentence explanation of why this mapping exists"
    )


class LLMClaimMappingsResponse(BaseModel):
    """Schema sent to Gemini for segment-to-claim mapping."""

    mappings: list[ClaimMapping] = Field(
        default_factory=list,
        description="0-N claim mappings for this segment (empty if not claim-relevant)",
    )


class SegmentClaimMapping(BaseModel):
    """Final output record with segment_id injected."""

    segment_id: str = Field(description="Folder name of the segment")
    claim_key: str = Field(description="Claim head key")
    relevance_type: RelevanceType = Field(description="How the segment relates to the claim")
    party_role: PartyRole = Field(description="Who the document supports")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(description="Explanation of the mapping")


# --------------------------------------------------------------------------- #
# Phase 4 — Case-Level Analysis                                               #
# --------------------------------------------------------------------------- #


class LLMMarkdownSection(BaseModel):
    """Schema for receiving a single markdown section from Gemini.

    Used by Phase 4 where each LLM call produces one report section.
    """

    markdown: str = Field(
        description="The full markdown content for this report section"
    )