"""
Test 2: Non-streaming STT endpoint (REST, multipart file upload).

Checks:
  1. It accepts a wav file and returns a transcript
  2. What the response actually looks like (plain text? JSON? confidence? timestamps?)
  3. Latency vs audio duration
  4. Whether sample rate matters (8k vs 16k)

Run:  python tests/test_stt_rest.py path/to/audio.wav
      python tests/test_stt_rest.py            (generates a test tone instead)
"""

import json
import sys
import time
import wave

import requests

sys.path.insert(0, ".")
import config  # noqa: E402

cfg = config.stt_rest()

HEADERS = {
    "X-Client-ID": cfg.client_id,
    "X-Client-KEY": cfg.client_key,
    "Authorization": f"Basic {cfg.auth_basic}",
}


def wav_info(path):
    with wave.open(path, "rb") as w:
        return {
            "channels": w.getnchannels(),
            "sample_rate": w.getframerate(),
            "sample_width_bytes": w.getsampwidth(),
            "duration_sec": round(w.getnframes() / w.getframerate(), 2),
        }


def transcribe(path, language=None):
    """Send one wav file. Returns (response_obj, seconds)."""
    language = language or cfg.language
    start = time.perf_counter()
    with open(path, "rb") as f:
        r = requests.post(
            cfg.url,
            headers=HEADERS,
            files={"audioFile": (path.split("/")[-1], f, "audio/wav")},
            data={"language": language},
            timeout=120,
        )
    elapsed = time.perf_counter() - start
    return r, elapsed


def show(r, elapsed):
    print(f"status      : {r.status_code}")
    print(f"time        : {elapsed:.2f}s")
    print(f"content-type: {r.headers.get('content-type')}")
    body = r.text
    print(f"raw body    : {body[:800]}")
    try:
        parsed = r.json()
        print("\nparsed JSON:")
        print(json.dumps(parsed, indent=2, ensure_ascii=False)[:1200])
        print("\ntop-level keys:", list(parsed.keys()) if isinstance(parsed, dict) else "(list)")
        print("  -> Look for: transcript text, confidence, word timestamps.")
        print("  -> Word timestamps matter: hotspot analysis needs them.")
    except Exception:
        print("\n(not JSON -> plain text response)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/test_stt_rest.py path/to/audio.wav")
        print("Record a few seconds of yourself talking and pass it in.")
        sys.exit(1)

    path = sys.argv[1]

    print("=" * 60)
    print("INPUT FILE")
    print("=" * 60)
    try:
        print(json.dumps(wav_info(path), indent=2))
    except Exception as e:
        print(f"Could not read as wav: {e}")
        print("If this fails, convert first:")
        print("  ffmpeg -i input.m4a -ar 16000 -ac 1 -c:a pcm_s16le out.wav")

    print("\n" + "=" * 60)
    print("TRANSCRIBE")
    print("=" * 60)
    r, elapsed = transcribe(path)
    show(r, elapsed)

    print("\n" + "=" * 60)
    print("LANGUAGE FLAG TEST")
    print("=" * 60)
    print("Same audio, different language flags. Compare Hinglish handling.")
    for lang in ["ENG", "HIN"]:
        try:
            r2, e2 = transcribe(path, language=lang)
            snippet = r2.text[:200].replace("\n", " ")
            print(f"\n  language={lang:4s} ({e2:.2f}s): {snippet}")
        except Exception as ex:
            print(f"\n  language={lang:4s} FAILED: {ex}")

    print("\nDone. Key things to note:")
    print("  - Does the response include word-level timestamps?")
    print("  - Does it include a confidence score?")
    print("  - Which language flag handles code-switched speech better?")
