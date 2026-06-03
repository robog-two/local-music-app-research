import os
import subprocess
import sys

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package], stdout=subprocess.DEVNULL)


try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Installing faster-whisper...")
    install("faster-whisper")
    from faster_whisper import WhisperModel


def transcribe(audio_path: str, model_size: str = "turbo"):
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Loading model '{model_size}'...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8", download_root=MODELS_DIR)

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
        print("Usage: python transcribe.py <audio_file> [model_size]")
        print("Model sizes: tiny, base, small, medium, large-v3, turbo (default)")
        sys.exit(1)

    audio_file = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "turbo"
    transcribe(audio_file, model)
