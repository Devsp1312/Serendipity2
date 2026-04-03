"""
Multi-pass focused extraction pipeline.

Instead of one LLM call extracting everything, this module runs:
  1. Pre-pass:   Read transcript → decide extraction order + invent custom passes
  2. Focused extraction passes:  One per category (sequential, deeper analysis)
  3. Dedup + cross-reasoning:    Merge outputs, resolve overlaps, find connections
  4. Identity extraction:        Extract biographical info (smart update)

The old single-pass extraction (src/extraction.py) is preserved as a fallback.
"""

import json
from typing import Optional

import networkx as nx

from src.config import (
    MAX_EXTRA_PASSES,
    CORE_EXTRACTION_CATEGORIES,
    IDENTITY_MIN_CONFIDENCE,
)
from src.llm_client import (
    call_llm,
    get_pre_pass_prompt,
    get_focused_extraction_prompt,
    get_custom_extraction_prompt,
    get_dedup_prompt,
    get_identity_prompt,
)
from src.logger import get_logger
from src.schemas import (
    PrePassOutput,
    FocusedExtractionOutput,
    DedupOutput,
    IdentityOutput,
    IdentityInfo,
    validate_pre_pass,
    validate_focused_extraction,
    validate_dedup,
    validate_identity,
)

logger = get_logger(__name__)


# ─── Pre-pass ────────────────────────────────────────────────────────────────

def run_pre_pass(diarized_text: str, model: str) -> PrePassOutput:
    """Reads transcript, decides extraction order + custom passes."""
    logger.info("Pre-pass started  model=%s  chars=%d", model, len(diarized_text))

    user_prompt = f"Analyze this diarized transcript and create an extraction plan:\n\n{diarized_text}"

    raw_dict, raw_str = call_llm(
        system_prompt=get_pre_pass_prompt(),
        user_prompt=user_prompt,
        model=model,
    )

    result = validate_pre_pass(raw_dict)

    # Enforce: 3 core categories must be present
    existing_cats = {p.category for p in result.passes}
    priority = max((p.priority for p in result.passes), default=0)
    for core_cat in CORE_EXTRACTION_CATEGORIES:
        if core_cat not in existing_cats:
            priority += 1
            result.passes.append(
                __import__("src.schemas", fromlist=["PassDefinition"]).PassDefinition(
                    category=core_cat,
                    focus_prompt=f"Extract {core_cat.replace('_', ' ')} items from the transcript.",
                    priority=priority,
                )
            )
            logger.warning("Pre-pass missing core category %r — added with priority %d", core_cat, priority)

    # Enforce: max total passes (3 core + MAX_EXTRA_PASSES custom)
    core_passes = [p for p in result.passes if p.category in CORE_EXTRACTION_CATEGORIES]
    custom_passes = [p for p in result.passes if p.category not in CORE_EXTRACTION_CATEGORIES]
    if len(custom_passes) > MAX_EXTRA_PASSES:
        logger.warning("Pre-pass proposed %d custom passes, capping at %d", len(custom_passes), MAX_EXTRA_PASSES)
        custom_passes = sorted(custom_passes, key=lambda p: p.priority)[:MAX_EXTRA_PASSES]
    result.passes = sorted(core_passes + custom_passes, key=lambda p: p.priority)

    logger.info(
        "Pre-pass complete: subject=%s  passes=%d  order=%s",
        result.main_subject_id,
        len(result.passes),
        [p.category for p in result.passes],
    )
    return result


# ─── Focused extraction ──────────────────────────────────────────────────────

def run_focused_extraction(
    diarized_text: str,
    category: str,
    focus_prompt: str,
    main_subject_id: str,
    model: str,
) -> tuple[FocusedExtractionOutput, str]:
    """Runs one focused extraction pass for a single category."""
    logger.info("Focused extraction started  category=%s  model=%s", category, model)

    # Load category-specific or custom prompt
    if category in CORE_EXTRACTION_CATEGORIES:
        # Map category name to prompt file name
        prompt_name_map = {
            "short_term_state": "short_term",
            "long_term_goal": "long_term",
            "core_value": "core_value",
        }
        system_prompt = get_focused_extraction_prompt(prompt_name_map[category])
    else:
        system_prompt = get_custom_extraction_prompt()

    # Inject main_subject_id and focus_prompt into the system prompt template
    system_prompt = system_prompt.replace("{main_subject_id}", main_subject_id)
    system_prompt = system_prompt.replace("{category}", category)
    system_prompt = system_prompt.replace("{focus_prompt}", focus_prompt)

    user_prompt = f"Analyze this diarized transcript:\n\n{diarized_text}"

    raw_dict, raw_str = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
    )

    # Ensure category is set correctly in case the LLM changed it
    raw_dict["category"] = category
    result = validate_focused_extraction(raw_dict)

    logger.info(
        "Focused extraction complete  category=%s  items=%d  relationships=%d",
        category, len(result.items), len(result.relationships),
    )
    return result, raw_str


# ─── Dedup + cross-reasoning ─────────────────────────────────────────────────

def run_dedup(
    extractions: list[FocusedExtractionOutput],
    model: str,
) -> tuple[DedupOutput, str]:
    """Merges all extraction outputs, deduplicates, and performs cross-category reasoning."""
    logger.info("Dedup + cross-reasoning started  passes=%d  model=%s", len(extractions), model)

    extractions_data = [
        {
            "category": e.category,
            "items": [item.model_dump() for item in e.items],
            "relationships": [r.model_dump() for r in e.relationships],
        }
        for e in extractions
    ]

    user_prompt = (
        f"Here are the extraction outputs from {len(extractions)} focused passes:\n\n"
        f"{json.dumps({'extractions': extractions_data}, indent=2)}\n\n"
        "Deduplicate, cross-reason, and produce the unified output."
    )

    raw_dict, raw_str = call_llm(
        system_prompt=get_dedup_prompt(),
        user_prompt=user_prompt,
        model=model,
    )

    result = validate_dedup(raw_dict)

    total_items = (
        len(result.core_values) + len(result.long_term_goals) +
        len(result.short_term_values) + len(result.interests) +
        sum(len(items) for items in result.custom_categories.values())
    )
    logger.info(
        "Dedup complete  total_items=%d  relationships=%d  insights=%d  custom_cats=%d",
        total_items, len(result.relationships),
        len(result.cross_category_insights), len(result.custom_categories),
    )
    return result, raw_str


# ─── Identity extraction ─────────────────────────────────────────────────────

def _get_existing_identity(G: nx.DiGraph) -> dict:
    """Pulls current identity info from the user node for smart update comparison."""
    user_data = G.nodes.get("user", {})
    return {
        "name": user_data.get("identity_name"),
        "name_confidence": user_data.get("identity_name_confidence", 0.0),
        "age": user_data.get("identity_age"),
        "age_confidence": user_data.get("identity_age_confidence", 0.0),
        "occupation": user_data.get("identity_occupation"),
        "occupation_confidence": user_data.get("identity_occupation_confidence", 0.0),
        "location": user_data.get("identity_location"),
        "location_confidence": user_data.get("identity_location_confidence", 0.0),
        "additional": user_data.get("identity_additional", {}),
        "additional_confidence": user_data.get("identity_additional_confidence", {}),
    }


def run_identity_extraction(
    diarized_text: str,
    existing_graph: nx.DiGraph,
    main_subject_id: str,
    model: str,
) -> tuple[IdentityOutput, str]:
    """Extracts biographical info. Smart update: only overwrite if new confidence > existing."""
    logger.info("Identity extraction started  model=%s", model)

    existing_identity = _get_existing_identity(existing_graph)

    system_prompt = get_identity_prompt()
    system_prompt = system_prompt.replace("{main_subject_id}", main_subject_id)

    user_prompt = (
        f"Existing identity data:\n{json.dumps(existing_identity, indent=2)}\n\n"
        f"Analyze this diarized transcript for biographical information:\n\n{diarized_text}"
    )

    raw_dict, raw_str = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
    )

    result = validate_identity(raw_dict)

    # Smart update: only keep fields where new confidence > existing confidence
    identity = result.identity
    for field in ("name", "age", "occupation", "location"):
        new_conf = getattr(identity, f"{field}_confidence")
        old_conf = existing_identity.get(f"{field}_confidence", 0.0)
        if new_conf <= old_conf and existing_identity.get(field) is not None:
            # Keep existing value — new extraction isn't more confident
            setattr(identity, field, existing_identity[field])
            setattr(identity, f"{field}_confidence", old_conf)
            logger.debug("Identity %s: keeping existing (old=%.2f >= new=%.2f)", field, old_conf, new_conf)

    # Smart update for additional fields
    for key, new_conf in identity.additional_confidence.items():
        old_conf = existing_identity.get("additional_confidence", {}).get(key, 0.0)
        if new_conf <= old_conf and key in existing_identity.get("additional", {}):
            identity.additional[key] = existing_identity["additional"][key]
            identity.additional_confidence[key] = old_conf

    logger.info(
        "Identity extraction complete  name=%s  age=%s  occupation=%s  location=%s",
        identity.name, identity.age, identity.occupation, identity.location,
    )
    return result, raw_str


# ─── Full multi-pass orchestrator ────────────────────────────────────────────

def run_multi_pass_pipeline(
    diarized_text: str,
    model: str,
    existing_graph: nx.DiGraph,
) -> tuple[DedupOutput, IdentityOutput, list[str]]:
    """
    Full multi-pass extraction orchestrator:
      1. Pre-pass → extraction plan (order + custom passes)
      2. Run each focused extraction pass sequentially
      3. Dedup + cross-reasoning
      4. Identity extraction (smart update)

    Returns:
        (dedup_output, identity_output, raw_llm_logs)
    """
    raw_logs: list[str] = []

    # Phase 1: Pre-pass
    pre_pass = run_pre_pass(diarized_text, model)
    raw_logs.append(f"=== PRE-PASS ===\n{json.dumps(pre_pass.model_dump(), indent=2)}")

    # Phase 2: Focused extraction passes (sequential)
    extractions: list[FocusedExtractionOutput] = []
    for pass_def in pre_pass.passes:
        extraction, raw_str = run_focused_extraction(
            diarized_text=diarized_text,
            category=pass_def.category,
            focus_prompt=pass_def.focus_prompt,
            main_subject_id=pre_pass.main_subject_id,
            model=model,
        )
        extractions.append(extraction)
        raw_logs.append(f"=== EXTRACTION: {pass_def.category} ===\n{raw_str}")

    # Phase 3: Dedup + cross-reasoning
    dedup_output, raw_dedup = run_dedup(extractions, model)
    raw_logs.append(f"=== DEDUP + CROSS-REASONING ===\n{raw_dedup}")

    # Phase 4: Identity extraction
    identity_output, raw_identity = run_identity_extraction(
        diarized_text=diarized_text,
        existing_graph=existing_graph,
        main_subject_id=pre_pass.main_subject_id,
        model=model,
    )
    raw_logs.append(f"=== IDENTITY ===\n{raw_identity}")

    logger.info(
        "Multi-pass pipeline complete  passes=%d  total_llm_calls=%d",
        len(pre_pass.passes),
        1 + len(pre_pass.passes) + 1 + 1,  # pre-pass + extractions + dedup + identity
    )

    return dedup_output, identity_output, raw_logs
