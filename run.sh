#!/bin/bash
set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: ./run.sh <audio_file> [model_size]"
    echo "Model sizes: tiny, base, small, medium, large-v3, turbo (default)"
    exit 1
fi

AUDIO_FILE=$(realpath "$1")
MODEL_SIZE="${2:-turbo}"
AUDIO_DIR=$(dirname "$AUDIO_FILE")
AUDIO_NAME=$(basename "$AUDIO_FILE")
MODELS_DIR="$(cd "$(dirname "$0")" && pwd)/models"

mkdir -p "$MODELS_DIR"

# Build image
echo "Building image..."
docker build -t whisper-local "$(dirname "$0")"

# Download model once if not already present (network allowed only here)
if [ -z "$(ls -A "$MODELS_DIR")" ]; then
    echo "Downloading model '$MODEL_SIZE' (one-time setup, network allowed)..."
    docker run --rm \
        -v "$MODELS_DIR":/app/models \
        whisper-local \
        python -c "
from faster_whisper import WhisperModel
WhisperModel('$MODEL_SIZE', device='cpu', compute_type='int8', download_root='/app/models')
print('Model downloaded.')
"
fi

# Transcribe with no network access
echo "Transcribing '$AUDIO_NAME' (network disabled)..."
docker run --rm \
    --network none \
    -v "$AUDIO_DIR":/app/audio \
    -v "$MODELS_DIR":/app/models \
    whisper-local \
    "/app/audio/$AUDIO_NAME" "$MODEL_SIZE"
