# Serendipity — Raspberry Pi 5 Setup

Full Serendipity pipeline on a Pi 5 (16 GB RAM, ARM64, Raspberry Pi OS Bookworm 64-bit).

## Hardware requirements
- Raspberry Pi 5 **16 GB** RAM
- **Active cooler** (official Pi5 cooler) — sustained CPU inference throttles without it
- **NVMe HAT or USB3 SSD** strongly recommended — SD cards bottleneck model loading
- Put `~/.cache/huggingface` and Ollama model cache on the SSD

## Quick start

```bash
# Clone or copy the Pi5/ folder to your Pi, then:
cd Pi5
bash setup_pi5.sh
```

The script installs system packages, creates a venv, installs CPU-only PyTorch, all app deps, and Ollama.

## Manual setup (step by step)

```bash
# System deps
sudo apt update
sudo apt install -y ffmpeg libsndfile1 build-essential python3-dev python3-venv libzbar0

# Python venv
python3 -m venv .venv && source .venv/bin/activate

# CPU-only torch FIRST (prevents whisperx from pulling CUDA torch)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# All other deps
pip install -r requirements-pi5.txt

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b        # ~2 GB, fast on Pi5
# or for better quality (slower):
ollama pull qwen2.5:7b        # ~5 GB

# Env
cp .env.example .env
# Edit .env — add HF_TOKEN, set SERENDIPITY_DEFAULT_MODEL to match what you pulled
```

## HuggingFace token (required for speaker diarization)

1. Create account at https://huggingface.co
2. Get token at https://huggingface.co/settings/tokens
3. Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
4. Set `HF_TOKEN=hf_your_token` in your `.env`

## Running

```bash
source .venv/bin/activate

# Web UI (accessible from any device on your LAN)
streamlit run app.py --server.address 0.0.0.0 --server.port 8501

# Headless CLI
python main.py --audio path/to/recording.m4a

# Batch process a folder of audio files
python batch_run.py --input-dir /path/to/audio --profile default
```

Open `http://<pi-ip>:8501` from your laptop or phone.

## QR codes

After each day is processed (and before the graph merges), Serendipity saves:
- `data/qrcodes/day_<timestamp>.png` — QR code encoding the day's extraction
- `data/qrcodes/day_<timestamp>.json` — full JSON for the same day

Scan the QR with your phone to send the day's profile snapshot to the matchmaking server.

To verify a QR on the Pi:
```bash
python scripts/decode_qr.py data/qrcodes/day_2026-04-16T10-00-00Z.png
```

## Performance expectations on Pi5 (CPU only)

| Task | Approx time |
|------|-------------|
| Whisper `base` on 5 min audio | ~3–6 min |
| Speaker diarization (pyannote) | ~2–5× realtime |
| LLM extraction (qwen2.5:3b) | ~1–3 min |
| LLM extraction (qwen2.5:7b) | ~3–8 min |

Use `htop` during a run to monitor RAM. Expect 6–10 GB used at peak.

## Model recommendations

| Model | RAM | Quality | Speed |
|-------|-----|---------|-------|
| `qwen2.5:3b` | ~2 GB | Good | Fast |
| `llama3.2:3b` | ~2.5 GB | Good | Fast |
| `qwen2.5:7b` | ~5 GB | Better | Slow |

Set `SERENDIPITY_DEFAULT_MODEL` in `.env` to match your `ollama pull`.
