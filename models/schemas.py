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
