"""
Test 5: FULL LOOP LATENCY.

This is the single most important test. It answers:
  "Does one conversational turn feel fast enough to be a conversation?"

Measures each stage of: audio -> STT -> LLM -> TTS -> audio

Run:  python tests/test_full_loop.py path/to/audio.wav
"""

import sys
import time

sys.path.insert(0, ".")
from tests.test_llm import call, user  # noqa: E402
from tests.test_stt_rest import transcribe  # noqa: E402
from tests.test_tts import synthesize, save_audio  # noqa: E402

SYSTEM = (
    "You are a structured check-in assistant. "
    "Ask ONE short follow-up question about what the person said. "
    "Two sentences maximum. Never diagnose or interpret."
)


def run(path):
    timings = {}

    print("=" * 60)
    print("FULL TURN: audio in -> spoken reply out")
    print("=" * 60)

    # --- Stage 1: STT ---
    print("\n[1/3] STT...")
    t0 = time.perf_counter()
    r, _ = transcribe(path)
    timings["stt"] = time.perf_counter() - t0
    try:
        transcript = r.json().get("transcript") or r.text
    except Exception:
        transcript = r.text
    print(f"      transcript: {transcript[:120]}")
    print(f"      took: {timings['stt']:.2f}s")

    # --- Stage 2: LLM (streaming, measure TTFT) ---
    print("\n[2/3] LLM...")
    t0 = time.perf_counter()
    reply, llm_total, ttft = call(
        [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            user(transcript),
        ],
        max_tokens=80,
        stream=True,
    )
    timings["llm_ttft"] = ttft or llm_total
    timings["llm_total"] = llm_total
    print(f"      reply: {reply.strip()[:120]}")
    print(f"      TTFT: {timings['llm_ttft']:.2f}s | total: {llm_total:.2f}s")

    # --- Stage 3: TTS ---
    print("\n[3/3] TTS...")
    t0 = time.perf_counter()
    tts_resp, _ = synthesize(reply.strip(), language="ENG")
    timings["tts"] = time.perf_counter() - t0
    if tts_resp.ok:
        save_audio(tts_resp, "loop_reply.wav")
    print(f"      took: {timings['tts']:.2f}s")

    # --- Results ---
    naive = timings["stt"] + timings["llm_total"] + timings["tts"]
    optimized = timings["stt"] + timings["llm_ttft"] + timings["tts"]

    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN")
    print("=" * 60)
    print(f"  STT                    {timings['stt']:6.2f}s")
    print(f"  LLM (full generation)  {timings['llm_total']:6.2f}s")
    print(f"  LLM (to first token)   {timings['llm_ttft']:6.2f}s")
    print(f"  TTS                    {timings['tts']:6.2f}s")
    print("  " + "-" * 40)
    print(f"  NAIVE   (wait for all) {naive:6.2f}s")
    print(f"  STREAMED (first sentence -> TTS early) ~{optimized:.2f}s")

    print("\nRead:")
    print("  under 2s  -> feels like conversation")
    print("  2s to 4s  -> usable, slight lag")
    print("  over 4s   -> needs optimization (stream + chunk TTS by sentence)")

    slowest = max(
        [("STT", timings["stt"]),
         ("LLM", timings["llm_total"]),
         ("TTS", timings["tts"])],
        key=lambda x: x[1],
    )
    print(f"\n  Bottleneck: {slowest[0]} ({slowest[1]:.2f}s)")
    print("  -> Optimize this first. Do not guess, this is the number that matters.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/test_full_loop.py path/to/audio.wav")
        sys.exit(1)
    run(sys.argv[1])
