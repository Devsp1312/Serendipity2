"""
Phase 2 — Extraction Engine.

Takes a diarized transcript string and sends it to the LLM with a structured
system prompt. The LLM identifies the Main Subject by speaker ID and extracts:
  - metadata:          main_subject_id and reasoning for identification
  - core_values:       recurring principles and habits
  - long_term_goals:   multi-month career or life goals
  - short_term_values: immediate desires, current stressors, or short-term priorities
  - interests:         hobbies, current fascinations, or topics discussed
  - relationships:     other speaker IDs and the nature of their interaction

The output is validated against ExtractionOutput (defined in schemas.py).
"""

from src.llm_client import call_llm, get_extraction_prompt
from src.logger import get_logger
from src.schemas import ExtractionOutput, validate_extraction

logger = get_logger(__name__)


def run_extraction(
    diarized_text: str,
    model: str,
) -> tuple[ExtractionOutput, str]:
    """
    Runs the extraction LLM on a diarized transcript string.

    Steps:
      1. Inject the diarized text into the user prompt
      2. Call the LLM with the v3 extraction system prompt
      3. Validate and parse the JSON response into ExtractionOutput

    Returns:
      (ExtractionOutput, raw_llm_json_string)
      The raw string is kept so the UI Log tab and CLI can display the model's reasoning.
    """
    logger.info("Extraction phase started  model=%s  chars=%d", model, len(diarized_text))

    user_prompt = f"Analyze this diarized transcript:\n\n{diarized_text}"

    raw_dict, raw_str = call_llm(
        system_prompt=get_extraction_prompt(),
        user_prompt=user_prompt,
        model=model,
    )

    extraction = validate_extraction(raw_dict)

    logger.info(
        "Extraction complete: subject=%s  values=%d  goals=%d  short_term=%d  interests=%d  relationships=%d",
        extraction.metadata.main_subject_id,
        len(extraction.core_values),
        len(extraction.long_term_goals),
        len(extraction.short_term_values),
        len(extraction.interests),
        len(extraction.relationships),
    )

    return extraction, raw_str
