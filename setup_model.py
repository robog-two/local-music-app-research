#!/usr/bin/env python3
"""Download the Whisper model on the host, where network access is allowed.

The model is stored as a plain directory of files (models/large-v3-turbo/) so
that the offline container can load it directly via faster-whisper, with no
HuggingFace Hub cache lookup and no network access.
"""
import shutil
import ssl
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODEL_DIR = MODELS_DIR / "large-v3-turbo"
MODEL_URL = "https://huggingface.co/h2oai/faster-whisper-large-v3-turbo/resolve/main/"
# vocabulary.json is required by CTranslate2 to load the model from a directory.
MODEL_FILES = [
    "model.bin",
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
]

try:
    import certifi
    ssl_context = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    print("Warning: certifi not found, using system certificates", file=sys.stderr)
    ssl_context = ssl.create_default_context()


def download_file(url, dest):
    """Stream a file to disk with a short status line."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name}...", end=" ", flush=True)
    try:
        with urllib.request.urlopen(url, context=ssl_context) as response:
            with open(dest, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"OK ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"FAILED\n  Error: {e}")
        if dest.exists():
            dest.unlink()
        raise


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    missing = [f for f in MODEL_FILES if not (MODEL_DIR / f).exists()]
    if not missing:
        print("Model 'large-v3-turbo' already downloaded")
        return

    print("Setting up model 'large-v3-turbo'...")
    try:
        for filename in missing:
            download_file(MODEL_URL + filename, MODEL_DIR / filename)
    except Exception:
        # download_file already removed the partial file it was writing; leave
        # any previously completed files in place so a re-run resumes.
        print("\nSetup failed. Re-run ./run.sh to retry.", file=sys.stderr)
        sys.exit(1)
    print(f"✓ Model 'large-v3-turbo' ready at {MODEL_DIR}")


if __name__ == "__main__":
    main()
