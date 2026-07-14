"""
Tone and emotion analysis: the four signal extractors from SDLC section 4.3.

  Signal 1: librosa     -> pitch, energy, MFCCs, pauses, speech rate
  Signal 2: openSMILE   -> eGeMAPS (jitter, shimmer, loudness, HNR)
  Signal 3: HuggingFace -> pretrained emotion label (inference only)
  Signal 4: heuristics  -> sighs, breaths, long inhalations

All local. No training anywhere.

The output is a compact TEXT summary, not a number dump. That is deliberate:
raw feature vectors are token-expensive and LLMs read them badly.
Bin first, then serialize.

Install:
  pip install librosa opensmile transformers torch soundfile numpy
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ToneSignals:
    """Everything one turn's audio tells us. Serialized into the LLM prompt."""

    # Signal 1 (librosa)
    pitch_mean_hz: float = 0.0
    pitch_variability: float = 0.0
    energy_mean: float = 0.0
    speech_rate_wps: float = 0.0
    pause_count: int = 0
    longest_pause_sec: float = 0.0

    # Signal 2 (openSMILE / eGeMAPS)
    jitter: float = 0.0
    shimmer: float = 0.0
    loudness: float = 0.0
    hnr: float = 0.0

    # Signal 3 (HuggingFace classifier)
    emotion_label: str = "unknown"
    emotion_confidence: float = 0.0

    # Signal 4 (nonverbal cues)
    nonverbal_cues: list = field(default_factory=list)

    # Derived
    arousal: str = "unknown"  # low / moderate / high

    def to_prompt_text(self) -> str:
        """
        Compact, qualitative summary for the LLM.

        Deliberately NOT raw numbers. Binned labels are fewer tokens and the
        LLM reasons about them far more reliably than about float values.
        """
        lines = [
            f"arousal: {self.arousal}",
            f"pitch: {_bin_pitch_var(self.pitch_variability)}",
            f"pace: {_bin_rate(self.speech_rate_wps)} "
            f"({self.speech_rate_wps:.1f} words/sec)",
            f"energy: {_bin_energy(self.energy_mean)}",
            f"voice quality: {_bin_voice_quality(self.jitter, self.shimmer)}",
            f"pauses: {self.pause_count} "
            f"(longest {self.longest_pause_sec:.1f}s)",
            f"audio emotion model: {self.emotion_label} "
            f"({self.emotion_confidence:.0%} confidence)",
        ]
        if self.nonverbal_cues:
            lines.append(f"nonverbal: {', '.join(self.nonverbal_cues)}")
        return "\n".join(lines)


# ---------- binning helpers (keeps the prompt short and readable) ----------

def _bin_pitch_var(v: float) -> str:
    if v < 15:
        return "flat, little variation"
    if v < 40:
        return "normal variation"
    return "highly variable"


def _bin_rate(v: float) -> str:
    if v == 0:
        return "unknown"
    if v < 2.0:
        return "slow"
    if v < 3.5:
        return "normal"
    return "fast, rushed"


def _bin_energy(v: float) -> str:
    if v < 0.02:
        return "quiet, low energy"
    if v < 0.08:
        return "normal"
    return "loud, high energy"


def _bin_voice_quality(jitter: float, shimmer: float) -> str:
    # Elevated jitter/shimmer = shaky, strained voice.
    if jitter > 0.02 or shimmer > 0.15:
        return "strained or shaky"
    return "steady"


# ---------- Signal 1: librosa ----------

def extract_librosa(audio: np.ndarray, sr: int,
                    word_count: Optional[int] = None) -> dict:
    import librosa

    out = {}

    # Pitch via pyin (probabilistic YIN). Voiced frames only.
    #
    # Range is SPEECH f0 (75-400 Hz), not librosa's usual C2-C7 music range
    # (65-2093 Hz). The music range is far too wide for voice: it searches
    # mostly empty space and locks onto harmonics instead of the fundamental.
    # Covers low male (~75 Hz) through high female / raised voice (~400 Hz).
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=75,
        fmax=400,
        sr=sr,
    )
    voiced_f0 = f0[~np.isnan(f0)]
    out["pitch_mean_hz"] = float(np.mean(voiced_f0)) if len(voiced_f0) else 0.0
    out["pitch_variability"] = float(np.std(voiced_f0)) if len(voiced_f0) else 0.0
    out["voiced_ratio"] = float(len(voiced_f0) / len(f0)) if len(f0) else 0.0

    # RMS energy.
    rms = librosa.feature.rms(y=audio)[0]
    out["energy_mean"] = float(np.mean(rms))

    # MFCCs: kept for downstream use, not sent to the LLM (too many numbers).
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    out["mfcc_means"] = mfcc.mean(axis=1).tolist()

    # Pauses: silent stretches in the energy envelope.
    #
    # Threshold is ABSOLUTE (dB below peak), not a percentile. A percentile
    # threshold always marks a fixed fraction of frames as silent regardless of
    # the audio, so a clip with real silences and one with none look identical.
    hop = 512
    frame_sec = hop / sr

    peak = float(rms.max())
    if peak <= 0:
        silence_thresh = 0.0
    else:
        # 35 dB below peak. Standard-ish for speech silence detection.
        silence_thresh = peak * (10 ** (-35 / 20))

    is_silent = rms < silence_thresh

    pauses = []
    run = 0
    for s in is_silent:
        if s:
            run += 1
        else:
            if run > 0:
                pauses.append(run * frame_sec)
            run = 0
    if run > 0:
        pauses.append(run * frame_sec)

    # Drop leading/trailing silence: that is not the speaker pausing,
    # it is just dead air at the edges of the recording.
    if len(is_silent) and is_silent[0] and pauses:
        pauses = pauses[1:]
    if len(is_silent) and is_silent[-1] and pauses:
        pauses = pauses[:-1]

    real_pauses = [p for p in pauses if p >= 0.3]  # under 300ms is not a pause
    out["pause_count"] = len(real_pauses)
    out["longest_pause_sec"] = float(max(real_pauses)) if real_pauses else 0.0
    out["pause_durations"] = real_pauses

    # Speech rate: needs the transcript's word count.
    duration = len(audio) / sr
    speaking_time = duration - sum(real_pauses)
    if word_count and speaking_time > 0:
        out["speech_rate_wps"] = word_count / speaking_time
    else:
        out["speech_rate_wps"] = 0.0

    out["duration_sec"] = duration
    return out


# ---------- Signal 2: openSMILE (eGeMAPS) ----------

_smile = None


def extract_opensmile(audio: np.ndarray, sr: int) -> dict:
    """
    eGeMAPS gives jitter and shimmer, which librosa does not provide.
    That is the whole reason openSMILE is in the stack.
    """
    global _smile
    import opensmile

    if _smile is None:
        _smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    df = _smile.process_signal(audio, sr)
    row = df.iloc[0]

    def get(name, default=0.0):
        return float(row[name]) if name in row.index else default

    return {
        "jitter": get("jitterLocal_sma3nz_amean"),
        "shimmer": get("shimmerLocaldB_sma3nz_amean"),
        "loudness": get("loudness_sma3_amean"),
        "hnr": get("HNRdBACF_sma3nz_amean"),
        "f0_semitone_mean": get("F0semitoneFrom27.5Hz_sma3nz_amean"),
        "f0_semitone_std": get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm"),
    }


# ---------- Signal 3: HuggingFace emotion classifier ----------

_emotion_pipe = None

# Audio-only. Never sees the transcript. Its label reflects HOW it sounded,
# not WHAT was said. The LLM handles the words.
EMOTION_MODEL = "superb/hubert-large-superb-er"


def extract_emotion(audio: np.ndarray, sr: int) -> dict:
    global _emotion_pipe
    from transformers import pipeline

    if _emotion_pipe is None:
        _emotion_pipe = pipeline(
            "audio-classification",
            model=EMOTION_MODEL,
        )

    # Most SER models expect 16kHz.
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    results = _emotion_pipe(audio, top_k=3)
    top = results[0]
    return {
        "emotion_label": top["label"],
        "emotion_confidence": float(top["score"]),
        "emotion_all": [(r["label"], round(float(r["score"]), 3))
                        for r in results],
    }


# ---------- Signal 4: nonverbal cues ----------

def detect_nonverbal(audio: np.ndarray, sr: int, librosa_feats: dict) -> list:
    """
    Rule-based. Sighs and audible breaths are documented arousal / cognitive
    load markers, and transcript-based analysis misses them entirely.

    A sigh: a long exhale = sustained low-frequency energy, low pitch content,
    following or preceding a pause.
    """
    import librosa as lb

    cues = []

    rms = lb.feature.rms(y=audio)[0]
    hop = 512
    frame_sec = hop / sr

    peak = float(rms.max())
    if peak <= 0:
        return cues

    # Breath is quiet but not silent: audible, well below speech level.
    # Absolute band relative to peak, not percentiles (see extract_librosa).
    floor = peak * (10 ** (-40 / 20))   # below this = silence
    ceiling = peak * (10 ** (-18 / 20))  # above this = actual speech
    mid_energy = (rms > floor) & (rms < ceiling)

    # Breath is noisy and unvoiced: high zero-crossing rate, no clear pitch.
    zcr = lb.feature.zero_crossing_rate(audio, hop_length=hop)[0]
    n = min(len(rms), len(zcr))
    rms, zcr, mid_energy = rms[:n], zcr[:n], mid_energy[:n]

    noisy = zcr > np.median(zcr)
    breath_like = mid_energy & noisy

    # A breath is a *sustained* run of breath-like frames (~200ms+).
    min_frames = int(0.2 / frame_sec)
    run = 0
    breath_events = 0
    for b in breath_like:
        if b:
            run += 1
        else:
            if run >= min_frames:
                breath_events += 1
            run = 0
    if run >= min_frames:
        breath_events += 1

    if breath_events >= 3:
        cues.append(f"audible breathing ({breath_events} events)")
    elif breath_events >= 1:
        cues.append("audible breath detected")

    # Sigh: a long pause paired with breath-like audio.
    if librosa_feats.get("longest_pause_sec", 0) > 1.0 and breath_events >= 1:
        cues.append("possible sigh (long pause with breath)")

    # Hesitation: many short pauses.
    if librosa_feats.get("pause_count", 0) >= 5:
        cues.append("frequent pausing (hesitation)")

    return cues


# ---------- Fusion ----------

def derive_arousal(feats: dict) -> str:
    """
    Arousal from audio ONLY. Never valence.

    Audio is reliable for arousal (calm vs agitated) but unreliable for valence
    (positive vs negative) -- an excited-happy voice and an angry voice look
    acoustically alike. This is the documented 'valence gap'.

    Valence is the LLM's job, from the transcript. Do not guess it here.
    """
    score = 0

    if feats.get("speech_rate_wps", 0) > 3.5:
        score += 1
    elif 0 < feats.get("speech_rate_wps", 0) < 2.0:
        score -= 1

    if feats.get("pitch_variability", 0) > 40:
        score += 1
    elif feats.get("pitch_variability", 0) < 15:
        score -= 1

    if feats.get("energy_mean", 0) > 0.08:
        score += 1
    elif feats.get("energy_mean", 0) < 0.02:
        score -= 1

    if feats.get("jitter", 0) > 0.02:
        score += 1

    if score >= 2:
        return "high"
    if score <= -2:
        return "low"
    return "moderate"


def analyze_turn(audio_path: str, word_count: Optional[int] = None) -> ToneSignals:
    """Run all four signals on one turn of audio. This is the module entrypoint."""
    import librosa as lb

    audio, sr = lb.load(audio_path, sr=None, mono=True)

    feats = extract_librosa(audio, sr, word_count=word_count)

    try:
        feats.update(extract_opensmile(audio, sr))
    except Exception as e:
        print(f"[tone] openSMILE failed: {e}")

    try:
        feats.update(extract_emotion(audio, sr))
    except Exception as e:
        print(f"[tone] emotion classifier failed: {e}")

    cues = detect_nonverbal(audio, sr, feats)
    arousal = derive_arousal(feats)

    return ToneSignals(
        pitch_mean_hz=feats.get("pitch_mean_hz", 0.0),
        pitch_variability=feats.get("pitch_variability", 0.0),
        energy_mean=feats.get("energy_mean", 0.0),
        speech_rate_wps=feats.get("speech_rate_wps", 0.0),
        pause_count=feats.get("pause_count", 0),
        longest_pause_sec=feats.get("longest_pause_sec", 0.0),
        jitter=feats.get("jitter", 0.0),
        shimmer=feats.get("shimmer", 0.0),
        loudness=feats.get("loudness", 0.0),
        hnr=feats.get("hnr", 0.0),
        emotion_label=feats.get("emotion_label", "unknown"),
        emotion_confidence=feats.get("emotion_confidence", 0.0),
        nonverbal_cues=cues,
        arousal=arousal,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m innerloop.tone path/to/audio.wav [word_count]")
        sys.exit(1)

    wc = int(sys.argv[2]) if len(sys.argv) > 2 else None
    signals = analyze_turn(sys.argv[1], word_count=wc)

    print("=" * 60)
    print("TONE SIGNALS (what the LLM will receive)")
    print("=" * 60)
    print(signals.to_prompt_text())
    print()
    print("=" * 60)
    print("RAW (for debugging, not sent to the LLM)")
    print("=" * 60)
    print(f"  pitch mean      : {signals.pitch_mean_hz:.1f} Hz")
    print(f"  pitch std       : {signals.pitch_variability:.1f}")
    print(f"  jitter          : {signals.jitter:.4f}")
    print(f"  shimmer         : {signals.shimmer:.4f}")
    print(f"  HNR             : {signals.hnr:.2f} dB")
    print(f"  energy          : {signals.energy_mean:.4f}")
