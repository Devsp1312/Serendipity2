import base64
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import qrcode
from qrcode.constants import ERROR_CORRECT_L

from src.core.logger import get_logger

logger = get_logger(__name__)

# QR v40 at error correction L holds ~2953 bytes of binary data.
# After base64 encoding gzipped JSON, ~2900 bytes is our safe ceiling.
_MAX_QR_BYTES = 2900


def export_day_to_qr(
    extraction: object,
    out_dir: Path,
    day: Optional[str] = None,
) -> tuple[Path, Path]:
    """
    Serialize a day's DedupOutput (or any dict/Pydantic model) to a QR PNG + JSON pair.

    Returns (png_path, json_path). Merge proceeds regardless — this is fire-and-forget.
    If the payload is too large to fit in one QR code, the QR encodes a compact
    reference object pointing to the JSON file instead of the full data.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if day is None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    # Convert Pydantic model → dict if needed
    if hasattr(extraction, "model_dump"):
        payload_dict = extraction.model_dump()
    elif isinstance(extraction, dict):
        payload_dict = extraction
    else:
        payload_dict = {"data": str(extraction)}

    envelope = {
        "schema_version": 1,
        "day": day,
        "extraction": payload_dict,
    }

    json_path = out_dir / f"day_{day}.json"
    json_path.write_text(json.dumps(envelope, indent=2, default=str), encoding="utf-8")

    # Build compact QR payload: gzip → base64
    compact = json.dumps(envelope, separators=(",", ":"), default=str).encode()
    compressed = gzip.compress(compact, compresslevel=9)
    encoded = base64.b64encode(compressed).decode("ascii")

    if len(encoded) <= _MAX_QR_BYTES:
        qr_data = encoded
        logger.info("QR payload: %d bytes (compressed+b64)", len(qr_data))
    else:
        # Fallback: encode a reference so the server can request the JSON file
        sha = hashlib.sha256(compact).hexdigest()[:16]
        ref = {"v": 1, "ref": sha, "file": json_path.name}
        qr_data = json.dumps(ref, separators=(",", ":"))
        logger.warning(
            "Day payload too large for QR (%d bytes); encoding reference instead. "
            "Full data saved to %s",
            len(encoded),
            json_path,
        )

    png_path = out_dir / f"day_{day}.png"
    img = qrcode.make(
        qr_data,
        error_correction=ERROR_CORRECT_L,
        box_size=8,
        border=4,
    )
    img.save(str(png_path))

    logger.info("QR export complete: %s", png_path)
    return png_path, json_path


def decode_qr_json(png_path: Path) -> dict:
    """
    Decode a QR PNG produced by export_day_to_qr back to the original envelope dict.
    Requires pyzbar + Pillow. Used by scripts/decode_qr.py for verification.
    """
    try:
        from PIL import Image
        from pyzbar.pyzbar import decode as pyzbar_decode
    except ImportError as e:
        raise RuntimeError("pyzbar and Pillow are required for decode_qr_json: pip install pyzbar pillow") from e

    img = Image.open(str(png_path))
    results = pyzbar_decode(img)
    if not results:
        raise ValueError(f"No QR code found in {png_path}")

    raw = results[0].data.decode("ascii")

    # Detect whether it's a base64+gzip payload or a plain JSON reference
    try:
        obj = json.loads(raw)
        # If it parsed as JSON, it's a reference object (fallback path)
        return {"type": "reference", "payload": obj}
    except json.JSONDecodeError:
        pass

    # Standard path: base64 → gzip → JSON
    compressed = base64.b64decode(raw)
    decompressed = gzip.decompress(compressed)
    return json.loads(decompressed)
