"""
Test 4: Riva streaming STT (gRPC).

This is the one that matters most for the real-time feel.
Checks:
  1. Connection to the Riva server works
  2. Partial (interim) transcripts arrive while you are still speaking
  3. Final transcripts arrive when you stop
  4. Whether word-level timestamps are available (needed for hotspot analysis)

Install first:
  pip install nvidia-riva-client

Run (from a file):
  python tests/test_riva_streaming.py path/to/audio.wav

Run (live mic, if the VDI exposes one):
  python tests/test_riva_streaming.py --mic
"""

import sys
import time

sys.path.insert(0, ".")
from innerloop import config  # noqa: E402

try:
    import riva.client
except ImportError:
    print("Missing dependency. Run:  pip install nvidia-riva-client")
    sys.exit(1)

cfg = config.riva()


def make_auth():
    return riva.client.Auth(uri=cfg.host, use_ssl=cfg.use_ssl)


def make_config():
    """Streaming config. word_time_offsets=True gives us word timestamps."""
    recognition_config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        language_code=cfg.language_code,
        max_alternatives=1,
        enable_automatic_punctuation=True,
        # Word timestamps: required later for hotspot analysis
        # (matching acoustic features to specific phrases).
        enable_word_time_offsets=True,
    )
    return riva.client.StreamingRecognitionConfig(
        config=recognition_config,
        interim_results=True,  # partial transcripts while speaking
    )


def run_from_file(path):
    auth = make_auth()
    service = riva.client.ASRService(auth)
    stream_cfg = make_config()

    print(f"Streaming file: {path}")
    print(f"Riva host     : {cfg.host}")
    print(f"Language      : {cfg.language_code}")
    print("-" * 60)

    start = time.perf_counter()
    first_partial_at = None
    final_count = 0

    with riva.client.AudioChunkFileIterator(
        path,
        chunk_n_frames=1600,  # 100ms chunks at 16kHz
    ) as audio_chunk_iterator:
        responses = service.streaming_response_generator(
            audio_chunks=audio_chunk_iterator,
            streaming_config=stream_cfg,
        )

        for response in responses:
            for result in response.results:
                if not result.alternatives:
                    continue
                alt = result.alternatives[0]
                elapsed = time.perf_counter() - start

                if result.is_final:
                    final_count += 1
                    print(f"\n[FINAL  {elapsed:5.2f}s] {alt.transcript}")
                    if alt.words:
                        print("  word timestamps available:")
                        for w in alt.words[:5]:
                            print(f"    {w.word:<15s} "
                                  f"{w.start_time:>6}ms -> {w.end_time:>6}ms")
                        if len(alt.words) > 5:
                            print(f"    ... ({len(alt.words)} words total)")
                    else:
                        print("  NO word timestamps returned.")
                        print("  -> hotspot analysis will need sentence-level fallback")
                else:
                    if first_partial_at is None:
                        first_partial_at = elapsed
                    print(f"[partial {elapsed:5.2f}s] {alt.transcript}", end="\r")

    total = time.perf_counter() - start
    print("\n" + "-" * 60)
    print(f"first partial : {first_partial_at:.2f}s" if first_partial_at
          else "first partial : none received")
    print(f"final results : {final_count}")
    print(f"total         : {total:.2f}s")
    print("\nKey check: partials arriving early = real-time feel is achievable.")


def run_from_mic():
    auth = make_auth()
    service = riva.client.ASRService(auth)
    stream_cfg = make_config()

    print("Speak now. Ctrl+C to stop.")
    print("-" * 60)

    with riva.client.audio_io.MicrophoneStream(
        rate=16000,
        chunk=1600,
    ) as mic_stream:
        responses = service.streaming_response_generator(
            audio_chunks=mic_stream,
            streaming_config=stream_cfg,
        )
        for response in responses:
            for result in response.results:
                if not result.alternatives:
                    continue
                alt = result.alternatives[0]
                if result.is_final:
                    print(f"\n[FINAL] {alt.transcript}")
                else:
                    print(f"[...] {alt.transcript}", end="\r")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "--mic":
        run_from_mic()
    else:
        run_from_file(sys.argv[1])
