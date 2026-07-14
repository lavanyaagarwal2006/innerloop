"""
Test 3: TTS endpoint (REST, JSON in, audio out).

Checks:
  1. It returns audio
  2. What format the audio is in (raw bytes? base64? what sample rate?)
  3. Latency vs text length (matters: long reports = long waits)
  4. Whether it handles Hinglish text

Run:  python tests/test_tts.py
Writes output audio to out/ so you can listen to it.
"""

import base64
import json
import os
import sys
import time
import wave

import requests

sys.path.insert(0, ".")
from innerloop import config  # noqa: E402

cfg = config.tts()

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Basic {cfg.auth_basic}",
    "X-Client-ID": cfg.client_id,
    "X-Client-Key": cfg.client_key,
}

OUT_DIR = "out"


def synthesize(text, language=None):
    """Send text, get audio. Returns (response, seconds)."""
    language = language or cfg.language
    payload = {
        "text": text,
        "language": language,
        "outputFormat": "AUDIO",
    }
    start = time.perf_counter()
    r = requests.post(cfg.url, headers=HEADERS, json=payload, timeout=120)
    elapsed = time.perf_counter() - start
    return r, elapsed


def save_audio(r, name):
    """Handle either raw audio bytes or base64-in-JSON. Returns saved path."""
    os.makedirs(OUT_DIR, exist_ok=True)
    ctype = r.headers.get("content-type", "")

    if "json" in ctype:
        body = r.json()
        print(f"  JSON response, keys: {list(body.keys())}")
        # Find the field holding the audio.
        audio_b64 = None
        for key in ("audio", "audioContent", "data", "audioData", "output"):
            if key in body:
                audio_b64 = body[key]
                print(f"  audio found in field: '{key}'")
                break
        if audio_b64 is None:
            print(f"  Could not find audio field. Full response:")
            print(json.dumps(body, indent=2)[:600])
            return None
        raw = base64.b64decode(audio_b64)
    else:
        print(f"  raw audio bytes, content-type: {ctype}")
        raw = r.content

    path = os.path.join(OUT_DIR, name)
    with open(path, "wb") as f:
        f.write(raw)
    print(f"  saved: {path} ({len(raw):,} bytes)")

    # Try to read format details.
    try:
        with wave.open(path, "rb") as w:
            print(f"  format: {w.getframerate()} Hz, {w.getnchannels()} ch, "
                  f"{w.getsampwidth() * 8}-bit, "
                  f"{w.getnframes() / w.getframerate():.2f}s")
    except Exception:
        print("  (not a standard wav header, may be raw PCM or mp3)")

    return path


def section(name):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")


if __name__ == "__main__":
    section("1. Short English")
    r, t = synthesize("What is on your mind right now?", language="ENG")
    print(f"status: {r.status_code}, time: {t:.2f}s")
    if r.ok:
        save_audio(r, "01_short_en.wav")

    section("2. Hinglish")
    r, t = synthesize("Aaj aapka din kaisa raha? Tell me about it.", language="HIN")
    print(f"status: {r.status_code}, time: {t:.2f}s")
    if r.ok:
        save_audio(r, "02_hinglish.wav")

    section("3. Latency vs length")
    print("This is the key measurement: does a long report take forever to speak?")
    samples = {
        "short  (1 sentence)": "Tell me more about that.",
        "medium (3 sentences)": (
            "It sounds like work has been taking up a lot of space lately. "
            "You mentioned the deadline twice. "
            "What part of it feels heaviest right now?"
        ),
        "long   (report-sized)": (
            "Here is a summary of what we talked about today. "
            "You raised three things: the project deadline on Friday, "
            "a disagreement with a teammate, and trouble sleeping this week. "
            "Your pace picked up when you talked about the deadline. "
            "Based on that, here is one technique to try. "
            "Box breathing: breathe in for four counts, hold for four, "
            "out for four, hold for four. Repeat for two minutes."
        ),
    }
    for label, text in samples.items():
        r, t = synthesize(text, language="ENG")
        words = len(text.split())
        print(f"  {label:22s} {words:3d} words -> {t:5.2f}s "
              f"({t / words * 1000:.0f} ms/word)")

    print("\nDone. Key things to note:")
    print("  - Audio format + sample rate (needed to play it back in the browser)")
    print("  - ms/word: multiply by report length to predict end-of-session wait")
    print("  - If long text is slow: split into sentences, stream them one by one")
