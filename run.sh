#!/bin/bash
set -euo pipefail

# Secure, offline, GPU-accelerated transcription for confidential interviews.
#
# Engine: whisper.cpp (Metal/GPU on Apple Silicon).
# Isolation: the transcription phase runs under macOS `sandbox-exec` with all
#            network access denied, so the audio cannot leave this machine.

if [ -z "${1:-}" ]; then
    echo "Usage: ./run.sh <audio_file>"
    exit 1
fi

AUDIO_FILE=$(realpath "$1")
AUDIO_NAME=$(basename "$AUDIO_FILE")
AUDIO_STEM="${AUDIO_NAME%.*}"
AUDIO_DIR=$(dirname "$AUDIO_FILE")
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WHISPER_REF="v1.8.6"                                   # pinned for reproducibility
WHISPER_DIR="$SCRIPT_DIR/vendor/whisper.cpp"
WHISPER_BIN="$WHISPER_DIR/build/bin/whisper-cli"
MODEL="$SCRIPT_DIR/models/ggml-large-v3-turbo.bin"
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
SANDBOX="$SCRIPT_DIR/sandbox.sb"
THREADS=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.ncpu)

# ======================================================================
# PHASE 1: Setup -- network ALLOWED. Runs before any audio is touched.
# Builds the whisper.cpp engine and downloads the model once.
# ======================================================================
if [ ! -x "$WHISPER_BIN" ]; then
    echo "Building whisper.cpp $WHISPER_REF with Metal (one-time)..."
    if [ ! -d "$WHISPER_DIR/.git" ]; then
        git clone --depth 1 --branch "$WHISPER_REF" \
            https://github.com/ggerganov/whisper.cpp "$WHISPER_DIR"
    fi
    cmake -S "$WHISPER_DIR" -B "$WHISPER_DIR/build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$WHISPER_DIR/build" -j --config Release --target whisper-cli
fi

if [ ! -f "$MODEL" ]; then
    echo "Downloading ggml-large-v3-turbo model (~1.6 GB, one-time)..."
    mkdir -p "$SCRIPT_DIR/models"
    curl -L --fail --progress-bar -o "$MODEL.partial" "$MODEL_URL"
    mv "$MODEL.partial" "$MODEL"
fi

# ======================================================================
# PHASE 2: Transcribe -- network DENIED via macOS sandbox.
# ffmpeg (decode -> 16 kHz mono WAV) and whisper-cli (transcribe on GPU) both
# run under sandbox-exec with `(deny network*)`. Neither can send the audio
# anywhere. Metal/GPU still works -- that is IOKit, not network.
# ======================================================================
WORK=$(mktemp -d -t whisper)
trap 'rm -rf "$WORK"' EXIT

echo "Transcribing '$AUDIO_NAME' on GPU (network blocked, $THREADS threads)..."
sandbox-exec -f "$SANDBOX" /bin/bash -s -- \
    "$AUDIO_FILE" "$WORK/audio.wav" "$WHISPER_BIN" "$MODEL" "$AUDIO_DIR/$AUDIO_STEM" "$THREADS" <<'INNER'
set -euo pipefail
audio="$1"; wav="$2"; bin="$3"; model="$4"; outbase="$5"; threads="$6"
ffmpeg -nostdin -loglevel error -y -i "$audio" -ar 16000 -ac 1 -c:a pcm_s16le "$wav"
"$bin" -m "$model" -f "$wav" -l en -t "$threads" -otxt -of "$outbase"
INNER

echo "Transcript saved to: $AUDIO_DIR/$AUDIO_STEM.txt"
