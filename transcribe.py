import os
import sys

# Resolve paths relative to this script so it works in the container (/app)
# and on the host alike.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
MODEL_DIR = os.path.join(MODELS_DIR, "large-v3-turbo")

# Defense in depth. The container already runs with `--network none`, which
# physically removes all network access. These flags additionally guarantee
# that huggingface_hub / transformers never attempt to reach the network,
# even if the container hardening were ever weakened.
os.environ["HF_HOME"] = MODELS_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from faster_whisper import WhisperModel


def transcribe(audio_path: str):
    if not os.path.isdir(MODEL_DIR):
        sys.exit(
            f"Model not found at {MODEL_DIR}.\n"
            "Run ./run.sh, which downloads the model on the host before the "
            "offline container starts."
        )

    print("Loading model 'large-v3-turbo' (local files, offline)...")
    # Passing a local directory makes faster-whisper load the model files
    # directly and bypass the HuggingFace Hub entirely -- no cache lookup and
    # no network code path is ever taken. local_files_only=True is a redundant
    # safeguard in case the path handling ever changes upstream.
    model = WhisperModel(
        MODEL_DIR,
        device="cpu",
        compute_type="int8",
        local_files_only=True,
    )

    print(f"Transcribing: {audio_path}")
    segments, info = model.transcribe(
        audio_path,
        language="en",
        beam_size=5,
        vad_filter=True,
    )

    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})\n")

    full_text = []
    for segment in segments:
        line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text.strip()}"
        print(line)
        full_text.append(segment.text.strip())

    output_path = audio_path.rsplit(".", 1)[0] + ".txt"
    with open(output_path, "w") as f:
        f.write("\n".join(full_text))
    print(f"\nTranscript saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file>")
        sys.exit(1)

    transcribe(sys.argv[1])
