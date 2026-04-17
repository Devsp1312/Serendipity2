#!/usr/bin/env bash
# Serendipity Pi5 one-shot setup script
# Run from inside the Pi5/ directory: bash setup_pi5.sh

set -e

echo "=== Serendipity Pi5 Setup ==="

# 1. System packages
echo "[1/7] Installing system packages..."
sudo apt update -q
sudo apt install -y ffmpeg libsndfile1 build-essential python3-dev python3-venv libzbar0

# 2. Increase swap to 4 GB (torch import + model loading spikes RAM)
echo "[2/7] Configuring swap (4 GB)..."
if ! grep -q "CONF_SWAPSIZE=4096" /etc/dphys-swapfile 2>/dev/null; then
    sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=4096/' /etc/dphys-swapfile || true
    sudo dphys-swapfile setup && sudo dphys-swapfile swapon || true
fi

# 3. Python venv
echo "[3/7] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# 4. CPU-only torch (must install BEFORE whisperx to avoid pulling CUDA torch)
echo "[4/7] Installing CPU-only PyTorch..."
pip install --quiet torch  # ARM64 wheel comes from PyPI, not the pytorch whl index

# 5. App dependencies
echo "[5/7] Installing app dependencies..."
pip install --quiet -r requirements-pi5.txt || {
    echo "  whisperx pip install failed — trying GitHub install..."
    pip install --quiet "git+https://github.com/m-bain/whisperX.git"
}

# 6. Ollama
echo "[6/7] Installing Ollama..."
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "  Ollama already installed: $(ollama --version)"
fi

echo ""
echo "  Pulling recommended model (qwen2.5:3b)..."
ollama pull qwen2.5:3b

# 7. Env file
echo "[7/7] Setting up .env..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env — edit it to add your HF_TOKEN and preferred model."
else
    echo "  .env already exists, skipping."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env and fill in HF_TOKEN"
echo "  2. Accept pyannote terms: https://huggingface.co/pyannote/speaker-diarization-3.1"
echo "  3. source .venv/bin/activate"
echo "  4. streamlit run app.py --server.address 0.0.0.0 --server.port 8501"
echo "  5. Open http://<pi-ip>:8501 from another device on your LAN"
