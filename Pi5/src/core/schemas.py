"""
Data models for all LLM JSON contracts in the Serendipity pipeline.

WHY Pydantic here:
  Every LLM call in the pipeline returns raw JSON text. LLMs are not perfectly
  reliable at following schema constraints — they may send a number where we
  expect a string, omit required keys, use an unlisted enum value, or wrap a
  string in a dict. Pydantic catches all of these at the boundary and either
  coerces the value to a safe default or raises LLMSchemaError (which the caller
  logs and handles gracefully, never crashing the pipeline).

Pipeline position of each model class:
  Pre-pass (Step 1 of extraction):
    PrePassOutput, PassDefinition

  Focused extraction (Step 2, one pass per category):
    FocusedExtractionOutput, ExtractionItem, RelationshipEntry

  Dedup + cross-reasoning (Step 3):
    DedupOutput, ExtractionMetadata, ExtractionOutput (legacy single-pass)

  Identity extraction (Step 4):
    IdentityOutput, IdentityInfo

  Gatekeeper (Phase 3, graph editing):
    GatekeeperOutput, GatekeeperAction

  Diarization:
    DiarizedTranscript, SpeakerInfo

Coercion philosophy:
  field_validators marked "mode=before" run on the RAW value from the LLM
  before Pydantic's own type conversion. They are used whenever the LLM
  reliably produces the wrong type — for example, returning a dict like
  {"item": "honesty"} where we expect a plain string. Coercing to a safe
  default keeps the pipeline alive while the warning log lets us track how
  often it happens and improve prompts accordingly.
"""

import json
from typing import Optional
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.core.logger import get_logger

logger = get_logger(__name__)


# ─── Valid enum sets ──────────────────────────────────────────────────────────
# These are the exhaustive lists the prompts instruct the LLM to use.
# Any value outside these sets is coerced to the stated default by _coerce().

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
# Used by the focused extraction passes (one pass per category such as
# "core_value", "long_term_goal", etc.).

VALID_INTERACTION_TYPES = {"two_way_conversation", "one_way_media"}


class ExtractionItem(BaseModel):
    """
    A single extracted trait with a confidence score.

    Represents one observation from the transcript — e.g. one core value
    ("honesty", 0.8) or one short-term state ("stressed about exams", 0.6).
    Multiple ExtractionItems are collected per category per extraction pass.
    """
    item: str
    confidence: float = Field(default=0.5)

    @field_validator("item", mode="before")
    @classmethod
    def coerce_item(cls, v) -> str:
        # LLMs occasionally return the item as a dict ({"item": "honesty"}) or
        # as None. We unwrap dicts and stringify everything else so downstream
        # code can always treat `item` as a plain string.
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            return str(v.get("item") or v.get("label") or next(
                (val for val in v.values() if isinstance(val, str) and val.strip()), ""
            ))
        if v is None:
            return ""
        return str(v)

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v) -> float:
        # LLMs sometimes send confidence as a string ("0.8") or out-of-range (>1).
        # We clamp to [0.0, 1.0] so graph edge weights are always valid.
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5


class ExtractionMetadata(BaseModel):
    """
    Identifies the Main Subject speaker from the diarized transcript.

    The main subject is the person whose profile we are building — usually
    the user themselves. The LLM picks the speaker ID (e.g. "SPEAKER_00")
    that talks most naturally about their own inner life.
    """
    main_subject_id: str = "SPEAKER_00"
    reasoning: str = ""


class RelationshipEntry(BaseModel):
    """
    One speaker relationship extracted from the diarized transcript.

    Captures who appears in the recording (by speaker_id or real name if
    identified) and how they relate to the main subject. These become
    "person" nodes in the knowledge graph connected via "knows" edges.
    """
    speaker_id: Optional[str] = None
    name: Optional[str] = None
    relationship_type: str = "unknown"
    interaction_type: str = "two_way_conversation"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_rel_confidence(cls, v) -> float:
        # Same null/string guard as ExtractionItem — LLMs often omit this field
        # for relationships, defaulting to 0.5 (moderate confidence).
        if v is None:
            return 0.5
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5

    @field_validator("relationship_type", mode="before")
    @classmethod
    def coerce_relationship_type(cls, v: str) -> str:
        # Reject any relationship_type not in the prompt's closed list.
        return _coerce(v, VALID_RELATIONSHIP_TYPES, "unknown", "relationship_type")

    @field_validator("interaction_type", mode="before")
    @classmethod
    def coerce_interaction_type(cls, v: str) -> str:
        # one_way_media vs two_way_conversation — relevant for how much
        # we should trust inferences about the main subject's reactions.
        return _coerce(v, VALID_INTERACTION_TYPES, "two_way_conversation", "interaction_type")


class ExtractionOutput(BaseModel):
    """
    Phase 2 output — what the LLM extracted from the diarized transcript.

    This is the legacy single-pass extraction schema (still used as a fallback).
    In the multi-pass pipeline, FocusedExtractionOutput is used per pass, then
    merged by the dedup step into DedupOutput.

    All fields default to empty so partial LLM responses still parse cleanly.
    """
    metadata:         ExtractionMetadata = Field(default_factory=ExtractionMetadata)
    core_values:      list[ExtractionItem] = Field(default_factory=list)
    long_term_goals:  list[ExtractionItem] = Field(default_factory=list)
    short_term_values: list[ExtractionItem] = Field(default_factory=list)
    interests:        list[ExtractionItem] = Field(default_factory=list)
    relationships:    list[RelationshipEntry] = Field(default_factory=list)


# ─── Phase 3 — Gatekeeper models ─────────────────────────────────────────────
# The gatekeeper LLM compares the current graph state with new extraction results
# and emits a list of GatekeeperActions to apply as graph edits.

class GatekeeperAction(BaseModel):
    """
    One graph edit from the gatekeeper LLM.

    Operations:
      add      — create a new node + edge if it doesn't exist yet
      strengthen — boost the confidence weight on an existing node's edge
      update   — overwrite metadata on an existing node (e.g. relationship tone changed)
      remove   — delete a node (e.g. new evidence contradicts the old belief)

    The label field is the human-readable trait name (e.g. "Academic Rigor").
    It is slugified by make_node_id() to form a stable, collision-free node ID.
    """
    operation:  str = "add"
    node_type:  str = "core_value"
    label:      Optional[str] = None
    confidence: Optional[float] = None
    metadata:   dict = Field(default_factory=dict)

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_gk_confidence(cls, v) -> Optional[float]:
        # None is valid here — it means "use DEFAULT_CONFIDENCE from config".
        if v is None:
            return None
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return None

    @field_validator("label", mode="before")
    @classmethod
    def coerce_label(cls, v) -> Optional[str]:
        # LLMs sometimes wrap the label in a dict ({"item": "..."} or {"label": "..."}).
        # We unwrap it so callers always get a plain string or None.
        if v is None:
            return None
        if isinstance(v, str):
            return v or None
        if isinstance(v, dict):
            # LLM sometimes returns {"item": "..."} or {"label": "..."}
            return str(v.get("item") or v.get("label") or next(iter(v.values()), ""))
        return str(v) if v else None

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
    """
    Phase 3 output — a list of graph edits to apply.

    The gatekeeper LLM emits this in a single response. apply_actions() in
    gatekeeper.py then iterates over the list and mutates the NetworkX graph
    in memory before it is saved back to SQLite.
    """
    actions: list[GatekeeperAction] = Field(default_factory=list)


# ─── Error type ───────────────────────────────────────────────────────────────

class LLMSchemaError(Exception):
    """
    Raised when the LLM response fails Pydantic validation.

    Carries the raw response string so the caller can log it for debugging
    without exposing it to the user. The pipeline catches this, logs the raw
    JSON, and skips the current step rather than crashing.
    """
    def __init__(self, message: str, raw_response: str = ""):
        super().__init__(message)
        self.raw_response = raw_response


# ─── Pre-pass models ────────────────────────────────────────────────────────
# The pre-pass is Step 1 of extraction: it reads the whole transcript once and
# decides WHAT to extract and in WHAT ORDER (higher-priority passes run first).

class PassDefinition(BaseModel):
    """
    One extraction pass as decided by the pre-pass.

    The pre-pass LLM emits a list of these. Each becomes one focused extraction
    LLM call in Step 2. The 3 core categories are always guaranteed to be present
    (run_pre_pass() adds them if the LLM forgot them).
    """
    category: str                   # "short_term_state", "long_term_goal", "core_value", or custom
    focus_prompt: str = ""          # Brief instruction for what to look for
    priority: int = 1               # Execution order (1 = first)

    @field_validator("category", mode="before")
    @classmethod
    def coerce_category(cls, v) -> str:
        # Guard against None or empty string — fall back to "custom" so the
        # pass still runs and doesn't crash the pipeline.
        if v is None or not isinstance(v, str) or not v.strip():
            return "custom"
        return v


class PrePassOutput(BaseModel):
    """
    Pre-pass output — extraction plan + main subject identification.

    The main_subject_id tells all subsequent extraction passes which speaker
    to focus on (e.g. "SPEAKER_00"). The passes list drives the extraction loop
    in run_multi_pass_pipeline().
    """
    main_subject_id: str = "SPEAKER_00"
    reasoning: str = ""
    passes: list[PassDefinition] = Field(default_factory=list)


# ─── Focused extraction models ──────────────────────────────────────────────
# Used in Step 2: one LLM call per category returns one FocusedExtractionOutput.

class FocusedExtractionOutput(BaseModel):
    """
    Output from a single focused extraction pass.

    `category` is echoed back from the pass definition so the dedup step knows
    which bucket each item belongs to. `items` are ExtractionItems for non-relationship
    categories; `relationships` carry the richer RelationshipEntry structure.
    """
    category: str
    items: list[ExtractionItem] = Field(default_factory=list)
    relationships: list[RelationshipEntry] = Field(default_factory=list)


# ─── Dedup + cross-reasoning models ─────────────────────────────────────────
# Step 3: all focused pass outputs are merged here. The dedup LLM removes
# duplicates, resolves conflicts, and can produce cross-category insights
# (e.g. "honesty as a value drives the long-term goal of transparent leadership").

def _coerce_extraction_list(v: object) -> list:
    """
    Coerce a list of ExtractionItem-like values — handles plain strings, dicts, None.

    LLMs sometimes return extraction lists as:
      - plain strings: ["honesty", "discipline"]  (no confidence key)
      - dicts with wrong keys: [{"label": "honesty", "score": 0.9}]
      - mixed: ["honesty", {"item": "discipline", "confidence": 0.7}]
    This normalizer converts all forms to the {"item": ..., "confidence": ...}
    dict format that ExtractionItem expects.
    """
    if not isinstance(v, list):
        return []
    result = []
    for item in v:
        if item is None:
            continue
        if isinstance(item, str):
            result.append({"item": item, "confidence": 0.5})
        elif isinstance(item, dict):
            # Ensure item field is a string (LLM sometimes sends null or nested dicts)
            raw_item = item.get("item") or item.get("label") or ""
            if isinstance(raw_item, dict):
                raw_item = next((val for val in raw_item.values() if isinstance(val, str)), "")
            item = {**item, "item": str(raw_item) if raw_item is not None else ""}
            result.append(item)
        else:
            result.append({"item": str(item), "confidence": 0.5})
    return [r for r in result if r.get("item")]  # drop empties


class DedupOutput(BaseModel):
    """
    Output from the dedup + cross-reasoning pass (Step 3).

    This is the final merged view of everything extracted from the transcript.
    It is passed directly to the gatekeeper (Phase 3) which decides which items
    actually change the graph.

    custom_categories holds any model-invented categories from the pre-pass
    (e.g. "humor_style", "communication_pattern") keyed by category name.
    """
    core_values:              list[ExtractionItem] = Field(default_factory=list)
    long_term_goals:          list[ExtractionItem] = Field(default_factory=list)
    short_term_values:        list[ExtractionItem] = Field(default_factory=list)
    interests:                list[ExtractionItem] = Field(default_factory=list)
    relationships:            list[RelationshipEntry] = Field(default_factory=list)
    cross_category_insights:  list[str] = Field(default_factory=list)
    custom_categories:        dict[str, list[ExtractionItem]] = Field(default_factory=dict)

    @field_validator("core_values", "long_term_goals", "short_term_values", "interests", mode="before")
    @classmethod
    def coerce_extraction_lists(cls, v: object) -> list:
        # All four standard extraction lists go through the same normalizer
        # because the dedup LLM may return any of the malformed shapes described
        # in _coerce_extraction_list above.
        return _coerce_extraction_list(v)

    @field_validator("cross_category_insights", mode="before")
    @classmethod
    def coerce_cross_insights(cls, v: object) -> list[str]:
        # The dedup LLM sometimes returns insights as dicts instead of plain strings.
        # We extract the string value from any dict and drop anything that isn't text.
        if not isinstance(v, list):
            return []
        result = []
        for item in v:
            if item is None:
                continue
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                # LLM sometimes returns {"item": "..."} or {"Free-text...": "..."}
                text = item.get("item") or next(
                    (val for val in item.values() if isinstance(val, str) and val.strip()),
                    None,
                )
                if text:
                    result.append(str(text))
            else:
                result.append(str(item))
        return result


# ─── Identity models ────────────────────────────────────────────────────────
# Step 4 of extraction: a dedicated LLM call extracts biographical facts about
# the main subject (name, age, job, city). These are stored as special attributes
# on the "user" anchor node rather than as separate graph nodes.

class IdentityInfo(BaseModel):
    """
    Biographical identity extracted from transcript.

    Each field has a paired *_confidence value so the smart-update logic in
    run_identity_extraction() can decide whether the new extraction is more
    certain than what's already in the graph. Only higher-confidence values
    overwrite existing ones — this prevents a noisy recording from erasing
    a previously well-established name.
    """
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

    @field_validator("name_confidence", "age_confidence", "occupation_confidence",
                     "location_confidence", mode="before")
    @classmethod
    def _clamp_core_confidence(cls, v) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0

    @field_validator("additional", mode="before")
    @classmethod
    def _coerce_additional_values(cls, v: object) -> dict:
        """Coerce non-string values to str — LLMs sometimes mix types in this dict."""
        if isinstance(v, dict):
            return {k: str(val) for k, val in v.items()}
        return v if v is not None else {}

    @field_validator("additional_confidence", mode="before")
    @classmethod
    def _coerce_additional_confidence(cls, v: object) -> dict:
        """Coerce non-float values to float — LLMs sometimes return strings or null."""
        if not isinstance(v, dict):
            return {}
        result = {}
        for k, val in v.items():
            try:
                result[k] = max(0.0, min(1.0, float(val)))
            except (TypeError, ValueError):
                result[k] = 0.5
        return result


class IdentityOutput(BaseModel):
    """Phase output for identity extraction — wraps IdentityInfo for JSON parsing."""
    identity: IdentityInfo = Field(default_factory=IdentityInfo)


# ─── Speaker diarization models ────────────────────────────────────────────
# These are produced by the diarization stage (before extraction) and represent
# the raw structural output of who-said-what in the audio.

class SpeakerInfo(BaseModel):
    """
    One identified speaker from a diarized transcript.

    speaker_id is the raw pyannote label ("SPEAKER_00"). name is populated
    later by spot_speaker_names() and _confirm_names_llm() once real names
    are identified. confidence reflects how certain we are of the name mapping.
    """
    speaker_id: str                    # "SPEAKER_00"
    name: Optional[str] = None         # "LeBron James" (after LLM identification)
    role: Optional[str] = None         # "NBA Player"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class DiarizedTranscript(BaseModel):
    """
    Result of the diarization + speaker identification pipeline.

    raw_text    — original "SPEAKER_XX: ..." format from pyannote + whisper merge
    labeled_text — same text but with real names substituted (e.g. "Monica: ...")
    speakers    — list of all identified speakers (used for downstream filtering)
    """
    raw_text: str                                    # diarized text with SPEAKER_XX labels
    labeled_text: str = ""                           # text with real names substituted
    speakers: list[SpeakerInfo] = Field(default_factory=list)
    recording_type: str = "conversation"             # monologue/interview/podcast/etc.
    topic: str = ""


# ─── Validation helpers ───────────────────────────────────────────────────────
# Each public validate_*() function is a thin wrapper around _validate() that
# gives callers a typed return value and a descriptive error label for logging.

def _validate(model_cls, raw: dict, label: str):
    """
    Shared validator — wraps Pydantic's ValidationError into LLMSchemaError.

    Args:
        model_cls: The Pydantic model class to validate against.
        raw:       The raw dict parsed from the LLM JSON response.
        label:     Human-readable name for the schema (used in error messages).

    Returns:
        A validated instance of model_cls.

    Raises:
        LLMSchemaError: If Pydantic validation fails after all field_validators run.
    """
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
        from src.core.telemetry import get_collector
        get_collector().record_coercion()
    except Exception:
        pass  # Never let telemetry crash the validation path
