#!/bin/bash
set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: ./run.sh <audio_file>"
    exit 1
fi

AUDIO_FILE=$(realpath "$1")
AUDIO_DIR=$(dirname "$AUDIO_FILE")
AUDIO_NAME=$(basename "$AUDIO_FILE")
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

mkdir -p "$MODELS_DIR"

# === PHASE 1: Setup (host machine, network allowed) ===
echo "Downloading model (host phase, network allowed)..."
python3 "$SCRIPT_DIR/setup_model.py"

# === PHASE 2: Build container (no network needed, model is mounted) ===
echo "Building container..."
docker build -t whisper-local "$SCRIPT_DIR"

# === PHASE 3: Transcribe (container, network fully disabled) ===
# --network none removes every network interface from the container (only
# loopback remains), so neither the model nor any third-party library can
# reach the internet. The model is mounted read-only; only the audio
# directory is writable, for the transcript output.
echo "Transcribing '$AUDIO_NAME' (container phase, network blocked)..."
docker run --rm \
    --network none \
    -v "$AUDIO_DIR":/app/audio \
    -v "$MODELS_DIR":/app/models:ro \
    whisper-local \
    "/app/audio/$AUDIO_NAME"
