"""
Test 1: LLM endpoint (gpt-oss-120b, OpenAI-compatible).

Checks:
  1. Basic call works
  2. Hinglish works
  3. JSON-only output works (needed for structured turns later)
  4. Latency: total time, and time-to-first-token when streaming

Run:  python tests/test_llm.py
"""

import json
import sys
import time

import requests

sys.path.insert(0, ".")
from innerloop import config  # noqa: E402

cfg = config.llm()

HEADERS = {
    "Authorization": f"Bearer {cfg.api_key}",
    "Content-Type": "application/json",
}


def call(messages, max_tokens=200, stream=False):
    """One chat-completions call. Returns (text, total_secs, ttft_secs)."""
    payload = {
        "model": cfg.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    start = time.perf_counter()

    if not stream:
        r = requests.post(cfg.base_url, headers=HEADERS, json=payload, timeout=120)
        r.raise_for_status()
        total = time.perf_counter() - start
        text = r.json()["choices"][0]["message"]["content"]
        return text, total, None

    # Streaming: measure time to first token.
    r = requests.post(cfg.base_url, headers=HEADERS, json=payload,
                      timeout=120, stream=True)
    r.raise_for_status()

    ttft = None
    chunks = []
    for line in r.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data.strip() == "[DONE]":
            break
        delta = json.loads(data)["choices"][0].get("delta", {})
        piece = delta.get("content")
        if piece:
            if ttft is None:
                ttft = time.perf_counter() - start
            chunks.append(piece)

    total = time.perf_counter() - start
    return "".join(chunks), total, ttft


def user(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def section(name):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")


if __name__ == "__main__":
    section("1. Basic English call")
    text, total, _ = call([user("Say hello in exactly 5 words.")], max_tokens=50)
    print(f"reply : {text}")
    print(f"time  : {total:.2f}s")

    section("2. Hinglish call")
    text, total, _ = call(
        [user("Reply in Hinglish, one sentence: aaj ka din kaisa hai?")],
        max_tokens=80,
    )
    print(f"reply : {text}")
    print(f"time  : {total:.2f}s")

    section("3. JSON-only output (structured turns need this)")
    text, total, _ = call(
        [
            {
                "role": "system",
                "content": [{"type": "text", "text":
                    "You output ONLY valid JSON. No markdown, no prose, no code fences."}],
            },
            user('Return JSON: {"emotion": "<one word>", "confidence": <0.0-1.0>} '
                 'for this line: "I am completely fine, everything is fine."'),
        ],
        max_tokens=60,
    )
    print(f"raw   : {text}")
    try:
        parsed = json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
        print(f"parsed: {parsed}  <- JSON mode works")
    except Exception as e:
        print(f"JSON PARSE FAILED: {e}")
        print("  -> Will need output cleaning/repair in the pipeline.")
    print(f"time  : {total:.2f}s")

    section("4. Latency: short output (a follow-up question)")
    text, total, ttft = call(
        [user("Ask one short follow-up question about someone's stressful workday.")],
        max_tokens=60,
        stream=True,
    )
    print(f"reply : {text}")
    print(f"TTFT  : {ttft:.2f}s   <- what the user actually waits for")
    print(f"total : {total:.2f}s")

    section("5. Latency: long output (an end-of-session report)")
    text, total, ttft = call(
        [user("Write a 300-word reflective summary of a stressful workday.")],
        max_tokens=500,
        stream=True,
    )
    print(f"chars : {len(text)}")
    print(f"TTFT  : {ttft:.2f}s")
    print(f"total : {total:.2f}s")

    print("\nDone. Note the gap between TTFT and total: that's why we stream.")
