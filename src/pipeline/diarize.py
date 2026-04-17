"""
Pipeline Step 2 — Speaker diarization ("who said what").

Input:  raw audio file (.mp3 / .wav / .m4a / etc.)
Output: DiarizedTranscript  — text with real speaker names, e.g. "Monica: ..."

Four-step flow:
  1. diarize_audio_chunked()  — pyannote 3.1 separates speakers in 10-min
                                subprocess chunks (memory-efficient)
  2. spot_speaker_names()     — finds real names from dialogue patterns:
                                  • self-id:  "I'm Monica"       (weight 2)
                                  • address:  "Hey Monica, ..."   (weight 1)
                                Names need ≥ 2 evidence points to be accepted.
  3. _confirm_names_llm()     — asks Ollama to identify any still-unknown speakers
                                based on context (filename, dialogue style)
  4. relabel_transcript()     — replaces "Speaker 3:" with "Monica:" throughout

Fallback: if no HuggingFace token is found, faster-whisper transcribes the audio
as a single "User" speaker (no separation). This is what batch_run.py uses by
default since pyannote requires a one-time model download.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave as _wave_module
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import ollama

from src.core.config import (
    HF_TOKEN_PATH,
    SPEAKER_ID_MODEL,
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE,
    WHISPER_BEAM_SIZE,
)
from src.core.logger import get_logger
from src.core.schemas import DiarizedTranscript, SpeakerInfo

logger = get_logger(__name__)


# ── HuggingFace token ─────────────────────────────────────────────────────────

def _get_hf_token() -> str | None:
    """Read HF token from HF_TOKEN env var or ~/.huggingface/token file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    if HF_TOKEN_PATH.exists():
        return HF_TOKEN_PATH.read_text().strip()
    return None


# ── Chunked diarization (subprocess-per-chunk for memory efficiency) ──────────

# Each chunk runs in a fresh subprocess so the ~500 MB pyannote model is fully
# released between chunks.  Results are saved to a JSON checkpoint file.
_CHUNK_WORKER = """\
import sys, gc, json, warnings
import numpy as np, scipy.io.wavfile as wf, torch
warnings.filterwarnings("ignore")
wav, t0, t1, out, tok = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
SR = 16000
_, raw = wf.read(wav)
chunk = raw[int(t0*SR):int(t1*SR)].astype(np.float32) / 32768.0
del raw; gc.collect()
wave = torch.from_numpy(chunk).unsqueeze(0); del chunk; gc.collect()
from pyannote.audio import Pipeline
p = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=tok)
out_data = p({"waveform": wave, "sample_rate": SR}); del wave; gc.collect()
ann, embs = out_data.speaker_diarization, out_data.speaker_embeddings
labels = sorted(ann.labels())
json.dump({
    "segs":    [[s.start+t0, s.end+t0, lbl] for s,_,lbl in ann.itertracks(yield_label=True)],
    "emb_map": {lbl: embs[i].tolist() for i, lbl in enumerate(labels) if i < len(embs)},
}, open(out, "w"))
print(f"OK:{len(labels)}")
"""


def _cosine_sim(a: list, b: np.ndarray) -> float:
    a = np.array(a, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0


def _match_or_create(lbl, emb_map, global_embs, global_counts, next_id,
                     threshold=0.65) -> str:
    if lbl not in emb_map:
        g = f"SPK_{next_id[0]:02d}"; next_id[0] += 1; return g
    emb = emb_map[lbl]
    best_gid, best_sim = None, -1.0
    for gid, gemb in global_embs.items():
        s = _cosine_sim(emb, gemb)
        if s > best_sim:
            best_sim, best_gid = s, gid
    if best_sim >= threshold and best_gid:
        n = global_counts[best_gid]
        global_embs[best_gid] = (global_embs[best_gid] * n + np.array(emb)) / (n + 1)
        global_counts[best_gid] = n + 1
        return best_gid
    g = f"SPK_{next_id[0]:02d}"; next_id[0] += 1
    global_embs[g] = np.array(emb, dtype=np.float32); global_counts[g] = 1
    return g


def _best_speaker(start: float, end: float, diar_segs: list) -> str:
    best, best_ov = "UNKNOWN", 0.0
    for ds, de, spk in diar_segs:
        ov = max(0.0, min(end, de) - max(start, ds))
        if ov > best_ov:
            best_ov, best = ov, spk
    return best


def diarize_audio_chunked(
    audio_path: str,
    hf_token: str | None = None,
    progress_callback: Callable | None = None,
) -> DiarizedTranscript:
    """
    Pyannote diarization in 10-min subprocess chunks + faster-whisper transcription.

    progress_callback(stage, current, total, message, **kwargs) is called at each step.
    Stages: "convert" | "chunk" | "chunk_done" | "transcribe" | "merge" | "done"
    """
    token = hf_token or _get_hf_token()
    if not token:
        logger.warning("No HuggingFace token — using basic transcription")
        return _transcribe_basic(audio_path)

    def report(stage, current, total, message, **kw):
        logger.info("[diarize] %s %d/%d %s", stage, current, total, message)
        if progress_callback:
            progress_callback(stage=stage, current=current, total=total,
                              message=message, **kw)

    CHUNK_SEC, OVERLAP_SEC = 10 * 60, 90
    ckpt_dir = tempfile.mkdtemp(prefix="srndy_")
    worker   = tempfile.mktemp(suffix="_worker.py")
    tmp_wav  = None

    try:
        Path(worker).write_text(_CHUNK_WORKER)

        # Convert to 16 kHz mono WAV
        report("convert", 0, 1, "Converting audio to 16 kHz WAV…")
        ffmpeg  = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        tmp_wav = tempfile.mktemp(suffix="_16k.wav")
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error",
             "-i", audio_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", tmp_wav],
            check=True,
        )

        with _wave_module.open(tmp_wav) as wf:
            total_sec = wf.getnframes() / wf.getframerate()

        # Build non-overlapping chunk list
        chunks: list[tuple[float, float]] = []
        start = 0.0
        while start < total_sec:
            end = min(start + CHUNK_SEC, total_sec)
            chunks.append((start, end))
            if end >= total_sec:
                break
            start = end - OVERLAP_SEC

        report("chunk", 0, len(chunks), f"{len(chunks)} chunk(s) to process…")

        all_segs: list[tuple[float, float, str]] = []
        global_embs:   dict[str, np.ndarray] = {}
        global_counts: dict[str, int]        = {}
        next_id = [0]

        for idx, (cs, ce) in enumerate(chunks):
            report("chunk", idx, len(chunks),
                   f"Chunk {idx+1}/{len(chunks)}  ({cs/60:.0f}–{ce/60:.0f} min)…")
            ckpt = os.path.join(ckpt_dir, f"c{idx:02d}.json")
            proc = subprocess.run(
                [sys.executable, worker, tmp_wav, str(cs), str(ce), ckpt, token],
                capture_output=True, text=True, timeout=600,
            )
            if proc.returncode != 0:
                logger.warning("Chunk %d failed — skipping\n%s", idx, proc.stderr[-300:])
                report("chunk_done", idx+1, len(chunks),
                       f"Chunk {idx+1} failed", speakers_found=0, failed=True)
                continue

            data    = json.loads(Path(ckpt).read_text())
            segs    = data["segs"]
            emb_map = data["emb_map"]

            local_map: dict[str, str] = {}
            if idx == 0:
                for lbl, emb in emb_map.items():
                    g = f"SPK_{next_id[0]:02d}"; next_id[0] += 1
                    global_embs[g] = np.array(emb, dtype=np.float32)
                    global_counts[g] = 1
                    local_map[lbl] = g
            else:
                for lbl in {s[2] for s in segs}:
                    local_map[lbl] = _match_or_create(
                        lbl, emb_map, global_embs, global_counts, next_id)

            skip_before = cs + OVERLAP_SEC if idx > 0 else 0.0
            chunk_spks: set[str] = set()
            for s, e, lbl in segs:
                if s >= skip_before:
                    gid = local_map.get(lbl, "UNK")
                    all_segs.append((s, e, gid))
                    chunk_spks.add(gid)

            report("chunk_done", idx+1, len(chunks),
                   f"Chunk {idx+1}/{len(chunks)} done — {len(chunk_spks)} speaker(s)",
                   speakers_found=len(chunk_spks))

        # Transcribe
        report("transcribe", 0, 1, "Transcribing audio…")
        from faster_whisper import WhisperModel
        fw = WhisperModel("base", device="cpu", compute_type="int8")
        w_segs = [
            {"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in fw.transcribe(audio_path, beam_size=1, language="en")[0]
        ]
        del fw

        # Assign speaker to each whisper segment
        report("merge", 0, 1, "Merging speaker labels with transcript…")
        for seg in w_segs:
            seg["speaker"] = _best_speaker(seg["start"], seg["end"], all_segs)

        # Build "Speaker N:" lines
        seen: list[str] = []
        for seg in w_segs:
            if seg["speaker"] not in seen:
                seen.append(seg["speaker"])
        spk_name = {sp: f"Speaker {i+1}" for i, sp in enumerate(seen)}

        lines: list[str] = []
        cur_spk, cur_parts = None, []
        for seg in w_segs:
            label = spk_name.get(seg["speaker"], seg["speaker"])
            if not seg["text"]:
                continue
            if label != cur_spk:
                if cur_spk:
                    lines.append(f"{cur_spk}: {' '.join(cur_parts)}")
                cur_spk, cur_parts = label, [seg["text"]]
            else:
                cur_parts.append(seg["text"])
        if cur_spk:
            lines.append(f"{cur_spk}: {' '.join(cur_parts)}")

        text = "\n".join(lines)
        speakers = sorted(set(spk_name.values()))
        report("done", len(chunks), len(chunks),
               f"Done — {len(speakers)} speaker(s), {len(lines)} turns")

        return DiarizedTranscript(
            raw_text=text,
            labeled_text=text,
            speakers=[SpeakerInfo(speaker_id=s) for s in speakers],
        )

    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        if tmp_wav:
            Path(tmp_wav).unlink(missing_ok=True)
        Path(worker).unlink(missing_ok=True)


# ── Fallback: basic transcription (no speaker separation) ─────────────────────

def _transcribe_basic(audio_path: str) -> DiarizedTranscript:
    """faster-whisper transcription — single speaker, no diarization."""
    from faster_whisper import WhisperModel
    logger.info("Basic transcription (no diarization)  path=%s", audio_path)
    fw = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    segs, _ = fw.transcribe(str(audio_path), beam_size=WHISPER_BEAM_SIZE)
    text = f"User: {' '.join(s.text.strip() for s in segs)}"
    return DiarizedTranscript(
        raw_text=text, labeled_text=text,
        speakers=[SpeakerInfo(speaker_id="User", name="User", confidence=1.0)],
        recording_type="monologue",
    )


# ── Speaker name identification ───────────────────────────────────────────────

# Words that look like names but are conversational filler
_NOT_NAMES = {
    # Greetings / conversational fillers
    "hey", "hi", "oh", "okay", "ok", "well", "look", "wait", "come", "yes",
    "no", "god", "man", "dude", "right", "sorry", "thanks", "thank", "please",
    "yeah", "yep", "nope", "wow", "gosh", "geez", "whoa", "ugh", "hmm",
    # Terms of endearment / group address (not real names)
    "honey", "sweetie", "babe", "baby", "buddy", "pal", "dear", "darling",
    "guys", "everyone", "everybody", "hello",
    # Common sentence starters
    "gonna", "gotta", "wanna", "could", "would", "should", "really", "just",
    "like", "know", "think", "good", "great", "fine", "sure", "kind", "nice",
    "true", "what", "that", "this", "there", "here", "when", "where", "how",
    "why", "which", "who", "will", "going", "said", "told", "asked", "listen",
    "now", "next", "then", "also", "still", "even", "only", "too", "very",
    "much", "more", "most", "some", "any", "all", "both", "each", "hold",
    "welcome", "congratulations", "together", "anyway", "actually", "nothing",
    "seriously", "exactly", "absolutely", "honestly", "totally", "probably",
    # Pronouns / articles that get capitalised
    "you", "she", "her", "him", "his", "they", "them", "their", "the",
    "not", "but", "and", "yet", "nor", "for", "can", "let", "new", "your",
}

# "Hey Monica" / "Hey, Monica" — prefix triggers followed by a capitalised name
# (?i:...) makes ONLY the prefix case-insensitive; the name group stays case-sensitive
_ADDRESS_RE = re.compile(
    r'(?i:\b(?:hey|hi|oh|look|listen|come\s+on|excuse\s+me|dear))[,\s]+([A-Z][a-z]{2,14})\b'
)
# "Monica, I ..." — capitalised name at position 0 of speech, followed by comma+space
_START_ADDRESS_RE = re.compile(r'^([A-Z][a-z]{3,14}),\s')

# "I'm Monica" / "This is Rachel" — current speaker identifies themselves
_SELF_ID_RE = re.compile(
    r"(?i:\bI'?m\s+)([A-Z][a-z]{2,14})\b|(?i:\bThis is\s+)([A-Z][a-z]{2,14})\b"
)


def spot_speaker_names(transcript: str) -> dict[str, str]:
    """
    Scans a 'Speaker N: text' transcript for name clues and returns a
    high-confidence {speaker_id: real_name} mapping.

    Two signals are used:
    - Self-identification:  "I'm Monica"  → that speaker is Monica  (weight 2)
    - Direct address:       "Hey Joey, …" → the next speaker is Joey (weight 1)

    Names must appear ≥ 2 times with ≥ 40% internal consistency to be accepted.
    Each name is assigned to at most one speaker.
    """
    pairs: list[tuple[str, str]] = []   # [(speaker_id, text), ...]
    for line in transcript.strip().split("\n"):
        if ":" in line:
            spk, _, text = line.partition(":")
            pairs.append((spk.strip(), text.strip()))

    votes: dict[str, Counter] = defaultdict(Counter)

    for i, (spk, text) in enumerate(pairs):
        # Self-identification (strong signal)
        for m in _SELF_ID_RE.finditer(text):
            name = (m.group(1) or m.group(2)).title()
            if name.lower() not in _NOT_NAMES:
                votes[spk][name] += 2

        # Direct address → vote for the next speaker
        if i + 1 < len(pairs):
            next_spk = pairs[i + 1][0]
            if next_spk != spk:
                for pattern in (_ADDRESS_RE, _START_ADDRESS_RE):
                    for m in pattern.finditer(text):
                        name = m.group(1).title()
                        if name.lower() not in _NOT_NAMES and len(name) > 2:
                            votes[next_spk][name] += 1

    # Greedy assignment: highest-evidence matches first, one name per speaker
    candidates = []
    for spk, counter in votes.items():
        if counter:
            top_name, top_count = counter.most_common(1)[0]
            total = sum(counter.values())
            candidates.append((spk, top_name, top_count, top_count / total))

    candidates.sort(key=lambda x: -x[2])   # most evidence first

    result: dict[str, str] = {}
    used: set[str] = set()
    for spk, name, count, confidence in candidates:
        if name not in used and count >= 2 and confidence >= 0.4:
            result[spk] = name
            used.add(name)

    logger.info("Name spotting found: %s", result)
    return result


def _confirm_names_llm(
    transcript: str,
    spotted: dict[str, str],
    filename: str,
    model: str | None = None,
) -> dict[str, str]:
    """
    Asks the LLM to identify any speakers that name-spotting didn't resolve.
    Merges the result with `spotted` and returns the combined mapping.
    """
    all_speakers = list(dict.fromkeys(
        line.partition(":")[0].strip()
        for line in transcript.split("\n")
        if ":" in line
    ))
    unresolved = [s for s in all_speakers if s not in spotted]

    if not unresolved:
        return dict(spotted)

    sample = "\n".join(
        line for line in transcript.split("\n") if ":" in line
    )[:4000]   # first ~4000 chars covers ~100 turns

    known_str = ", ".join(f"{s}={n}" for s, n in spotted.items()) or "none yet"
    prompt = (
        f'Audio file: "{filename}"\n'
        f"Already identified: {known_str}\n"
        f"Still unknown: {', '.join(unresolved)}\n\n"
        f"Transcript sample:\n{sample}\n\n"
        f"Based on dialogue style, topics, and filename, what are the real names "
        f"for the unknown speakers?\n"
        f'Return ONLY a JSON object, e.g. {{"Speaker 3": "Ross", "Speaker 5": "Phoebe"}}.\n'
        f"Only include speakers you are confident about."
    )

    try:
        resp = ollama.chat(
            model=model or SPEAKER_ID_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        llm_map = _parse_json(resp.message.content)
        result = dict(spotted)
        for spk, name in llm_map.items():
            if isinstance(name, str) and name.strip() and spk not in result:
                result[spk] = name.strip()
        logger.info("LLM name confirmation added: %s",
                    {k: v for k, v in result.items() if k not in spotted})
        return result
    except Exception as e:
        logger.warning("LLM name confirmation failed: %s", e)
        return dict(spotted)


def _parse_json(text: str) -> dict:
    """Extract a JSON object from LLM output (handles markdown fences, <think> tags)."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    for candidate in (
        cleaned,
        re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL),
        cleaned[cleaned.find("{"):cleaned.rfind("}") + 1] if "{" in cleaned else None,
    ):
        if candidate is None:
            continue
        blob = candidate.group(1) if hasattr(candidate, "group") else candidate
        try:
            return json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            pass
    logger.warning("Could not parse JSON from LLM response")
    return {}


# ── Transcript relabeling ─────────────────────────────────────────────────────

def relabel_transcript(transcript: str, name_map: dict[str, str]) -> str:
    """Replace 'Speaker N:' labels with real names throughout the transcript."""
    result = transcript
    for speaker_id, name in name_map.items():
        if name:
            result = result.replace(f"{speaker_id}:", f"{name}:")
    return result


# ── Main pipeline entry point ─────────────────────────────────────────────────

def run_diarization_pipeline(
    audio_path: str,
    filename: str,
    speaker_id_model: str | None = None,
    hf_token: str | None = None,
    progress_callback: Callable | None = None,
) -> DiarizedTranscript:
    """
    Full pipeline: audio file → named-speaker transcript.

      1. Diarize with pyannote (chunked) — or fall back to basic if no token
      2. Spot real names from "Hey Monica"-style patterns in the dialogue
      3. Ask LLM to fill in any names that weren't spotted
      4. Rewrite the transcript with real names

    Args:
        audio_path:         Path to audio file (.mp3 / .wav / .m4a)
        filename:           Original filename (gives context to the LLM)
        speaker_id_model:   Ollama model for name identification
        hf_token:           HuggingFace token (auto-detected if not provided)
        progress_callback:  callable(stage, current, total, message, **kwargs)
    """
    token = hf_token or _get_hf_token()

    # Step 1 — Diarize
    if token:
        try:
            transcript = diarize_audio_chunked(audio_path, token, progress_callback)
        except Exception as e:
            logger.warning("Chunked diarization failed (%s) — using basic", e)
            transcript = _transcribe_basic(audio_path)
    else:
        transcript = _transcribe_basic(audio_path)

    text = transcript.labeled_text or transcript.raw_text

    # Skip name identification for single-speaker (basic) mode
    if len(transcript.speakers) <= 1:
        return transcript

    # Step 2 — Spot names from dialogue
    spotted = spot_speaker_names(text)

    # Step 3 — LLM fills in the rest
    name_map = _confirm_names_llm(text, spotted, filename, model=speaker_id_model)

    # Step 4 — Relabel transcript
    if name_map:
        labeled = relabel_transcript(text, name_map)
        transcript.labeled_text = labeled
        transcript.speakers = [
            SpeakerInfo(speaker_id=name_map.get(s.speaker_id, s.speaker_id))
            for s in transcript.speakers
        ]

    return transcript
