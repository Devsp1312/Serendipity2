"""
Data models for all LLM JSON contracts.

Pydantic validates every LLM response. Invalid enum values (e.g. an unexpected
relationship_type) are coerced to safe defaults and logged — useful for spotting
prompt quality issues without crashing the pipeline.
"""

import json
from typing import Optional
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.logger import get_logger

logger = get_logger(__name__)


# ─── Valid enum sets ──────────────────────────────────────────────────────────

VALID_RELATIONSHIP_TYPES = {"colleague", "friend", "family", "mentor", "unknown"}
VALID_TONES              = {"collaborative", "tense", "neutral", "supportive", "romantic"}
VALID_OPERATIONS         = {"add", "update", "remove", "strengthen"}
# "relationship" is the LLM-facing label; the graph stores these as node_type="person"
VALID_NODE_TYPES         = {"core_value", "long_term_goal", "short_term_state", "relationship"}


def _coerce(v: str, valid: set, default: str, field: str) -> str:
    """If v is not in valid, warn, record a coercion counter tick, and return default."""
    if v in valid:
        return v
    logger.warning("Invalid %s %r coerced to %r (valid: %s)", field, v, default, sorted(valid))
    _record_coercion()
    return default


# ─── Phase 2 — Extraction models ─────────────────────────────────────────────

VALID_INTERACTION_TYPES = {"two_way_conversation", "one_way_media"}


class ExtractionItem(BaseModel):
    """A single extracted trait with a confidence score."""
    item: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ExtractionMetadata(BaseModel):
    """Identifies the Main Subject speaker from the diarized transcript."""
    main_subject_id: str = "SPEAKER_00"
    reasoning: str = ""


class RelationshipEntry(BaseModel):
    """One speaker relationship extracted from the diarized transcript."""
    speaker_id: str
    name: Optional[str] = None
    relationship_type: str = "unknown"
    interaction_type: str = "two_way_conversation"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("relationship_type", mode="before")
    @classmethod
    def coerce_relationship_type(cls, v: str) -> str:
        return _coerce(v, VALID_RELATIONSHIP_TYPES, "unknown", "relationship_type")

    @field_validator("interaction_type", mode="before")
    @classmethod
    def coerce_interaction_type(cls, v: str) -> str:
        return _coerce(v, VALID_INTERACTION_TYPES, "two_way_conversation", "interaction_type")


class ExtractionOutput(BaseModel):
    """Phase 2 output — what the LLM extracted from the diarized transcript.
    All fields default to empty so partial LLM responses still parse cleanly."""
    metadata:         ExtractionMetadata = Field(default_factory=ExtractionMetadata)
    core_values:      list[ExtractionItem] = Field(default_factory=list)
    long_term_goals:  list[ExtractionItem] = Field(default_factory=list)
    short_term_values: list[ExtractionItem] = Field(default_factory=list)
    interests:        list[ExtractionItem] = Field(default_factory=list)
    relationships:    list[RelationshipEntry] = Field(default_factory=list)


# ─── Phase 3 — Gatekeeper models ─────────────────────────────────────────────

class GatekeeperAction(BaseModel):
    """One graph edit from the gatekeeper LLM (add / strengthen / update / remove)."""
    operation:  str = "add"
    node_type:  str = "core_value"
    label:      str
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata:   dict = Field(default_factory=dict)

    @field_validator("operation", mode="before")
    @classmethod
    def coerce_operation(cls, v: str) -> str:
        return _coerce(v, VALID_OPERATIONS, "add", "operation")

    @field_validator("node_type", mode="before")
    @classmethod
    def coerce_node_type(cls, v: str) -> str:
        # Allow custom category types to pass through unmodified —
        # they are created by the pre-pass and handled by _apply_custom_action()
        if v in VALID_NODE_TYPES:
            return v
        # Log but don't coerce — treat as custom category
        logger.debug("Non-standard node_type %r — treating as custom category", v)
        return v


class GatekeeperOutput(BaseModel):
    """Phase 3 output — a list of graph edits to apply."""
    actions: list[GatekeeperAction] = Field(default_factory=list)


# ─── Error type ───────────────────────────────────────────────────────────────

class LLMSchemaError(Exception):
    """Raised when the LLM response fails Pydantic validation. Carries the raw response for logging."""
    def __init__(self, message: str, raw_response: str = ""):
        super().__init__(message)
        self.raw_response = raw_response


# ─── Pre-pass models ────────────────────────────────────────────────────────

class PassDefinition(BaseModel):
    """One extraction pass as decided by the pre-pass."""
    category: str                   # "short_term_state", "long_term_goal", "core_value", or custom
    focus_prompt: str = ""          # Brief instruction for what to look for
    priority: int = 1               # Execution order (1 = first)


class PrePassOutput(BaseModel):
    """Pre-pass output — extraction plan + main subject identification."""
    main_subject_id: str = "SPEAKER_00"
    reasoning: str = ""
    passes: list[PassDefinition] = Field(default_factory=list)


# ─── Focused extraction models ──────────────────────────────────────────────

class FocusedExtractionOutput(BaseModel):
    """Output from a single focused extraction pass."""
    category: str
    items: list[ExtractionItem] = Field(default_factory=list)
    relationships: list[RelationshipEntry] = Field(default_factory=list)


# ─── Dedup + cross-reasoning models ─────────────────────────────────────────

class DedupOutput(BaseModel):
    """Output from the dedup + cross-reasoning pass."""
    core_values:              list[ExtractionItem] = Field(default_factory=list)
    long_term_goals:          list[ExtractionItem] = Field(default_factory=list)
    short_term_values:        list[ExtractionItem] = Field(default_factory=list)
    interests:                list[ExtractionItem] = Field(default_factory=list)
    relationships:            list[RelationshipEntry] = Field(default_factory=list)
    cross_category_insights:  list[str] = Field(default_factory=list)
    custom_categories:        dict[str, list[ExtractionItem]] = Field(default_factory=dict)


# ─── Identity models ────────────────────────────────────────────────────────

class IdentityInfo(BaseModel):
    """Biographical identity extracted from transcript."""
    name: Optional[str] = None
    name_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    age: Optional[str] = None
    age_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    occupation: Optional[str] = None
    occupation_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    location: Optional[str] = None
    location_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    additional: dict[str, str] = Field(default_factory=dict)
    additional_confidence: dict[str, float] = Field(default_factory=dict)


class IdentityOutput(BaseModel):
    """Phase output for identity extraction."""
    identity: IdentityInfo = Field(default_factory=IdentityInfo)


# ─── Speaker diarization models ────────────────────────────────────────────

class SpeakerInfo(BaseModel):
    """One identified speaker from a diarized transcript."""
    speaker_id: str                    # "SPEAKER_00"
    name: Optional[str] = None         # "LeBron James" (after LLM identification)
    role: Optional[str] = None         # "NBA Player"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class DiarizedTranscript(BaseModel):
    """Result of the diarization + speaker identification pipeline."""
    raw_text: str                                    # diarized text with SPEAKER_XX labels
    labeled_text: str = ""                           # text with real names substituted
    speakers: list[SpeakerInfo] = Field(default_factory=list)
    recording_type: str = "conversation"             # monologue/interview/podcast/etc.
    topic: str = ""


# ─── Validation helpers ───────────────────────────────────────────────────────

def _validate(model_cls, raw: dict, label: str):
    """Shared validator — wraps Pydantic's ValidationError into LLMSchemaError."""
    try:
        return model_cls.model_validate(raw)
    except ValidationError as e:
        errors = "; ".join(f"{err['loc']}: {err['msg']}" for err in e.errors())
        raise LLMSchemaError(f"{label} validation failed: {errors}", raw_response=json.dumps(raw))


def validate_extraction(raw: dict) -> ExtractionOutput:
    return _validate(ExtractionOutput, raw, "Extraction")


def validate_gatekeeper(raw: dict) -> GatekeeperOutput:
    return _validate(GatekeeperOutput, raw, "Gatekeeper")


def validate_pre_pass(raw: dict) -> PrePassOutput:
    return _validate(PrePassOutput, raw, "PrePass")


def validate_focused_extraction(raw: dict) -> FocusedExtractionOutput:
    return _validate(FocusedExtractionOutput, raw, "FocusedExtraction")


def validate_dedup(raw: dict) -> DedupOutput:
    return _validate(DedupOutput, raw, "Dedup")


def validate_identity(raw: dict) -> IdentityOutput:
    return _validate(IdentityOutput, raw, "Identity")


# ─── Internal ─────────────────────────────────────────────────────────────────

def _record_coercion() -> None:
    """Bump the coercion counter on the active telemetry run (best-effort)."""
    try:
        from src.telemetry import get_collector
        get_collector().record_coercion()
    except Exception:
        pass  # Never let telemetry crash the validation path
