# InnerLoop

An "emotionally intelligent" voice-based check-in companion that leads a structured conversation, reads
both what you say and how you say it, and ends every session with a concrete
next step.

Built on Airtel's speech-to-text, text-to-speech, and LLM endpoints. No model
training or fine-tuning is involved.

---

## What it does

Most reflective tools wait for you to talk and then respond. InnerLoop leads.
It opens with a framing question, steers the conversation toward clarity, and
closes with a plan.

A session runs as a loop:

1. **It asks.** The opening question gives you something to push against, not a
   blank page. Never "how are you today".
2. **You speak.** Audio is captured in the browser and transcribed.
3. **It listens on two channels.** The transcript tells it *what* you said. The
   raw audio tells it *how* you said it: pace, pitch, pauses, voice strain,
   sighs.
4. **It probes.** One question per turn, asking you to say more about something
   specific you raised. It does not tell you what you "really" mean.
5. **It converges.** As the session progresses, it shifts from exploring to
   narrowing to closing.
6. **It reports.** Two outputs, every time.

### The two reports

**Report 1 — what was said.** Purely observational. The issues you raised in
your own framing, any to-dos you mentioned yourself, and tone patterns tied to
the moments they occurred ("your pace picked up when the deadline came up").
It never claims to know what is underneath.

**Report 2 — the plan.** One grounding technique matched to your measured
arousal, and one journaling style matched to how you process things. Not a
menu. One clear next step you can take today.

---

## How the emotion analysis works

The core design principle: **audio tells you arousal, text tells you valence.**

Acoustic features are reliable for arousal (calm versus agitated) but
unreliable for valence (positive versus negative), because an excited voice and
an angry voice look acoustically similar. Text is the reverse. This is the
documented "valence gap" in speech emotion recognition, and InnerLoop's
architecture is built around it: each signal is used only for what it measures
well, then fused.

Four signals run on every turn of audio:

| Signal | Tool | What it measures |
|---|---|---|
| Acoustic prosody | librosa | Pitch contour, energy, MFCCs, pauses, speech rate |
| Voice quality | openSMILE (eGeMAPS) | Jitter, shimmer, loudness, harmonics-to-noise ratio |
| Emotion label | HuggingFace (inference only) | An emotion category with a confidence score |
| Nonverbal cues | Custom heuristics | Sighs, audible breaths, hesitation patterns |

All four are serialized into a compact text summary and handed to the LLM
alongside the transcript. The LLM makes the combined read.

**Disagreement between the channels is signal, not noise.** Angry words in a
flat voice is a meaningfully different state from angry words shouted. A system
with only audio would miss the anger; a system with only text would miss the
flatness. InnerLoop sees both, and can ask about the gap.

Tone is measured per sentence, not per word — word-level prosody measurement is
unstable, sentence-level is reliable.

---

## Architecture

```
        ┌─────────────────────────────────────────┐
        │              Browser                    │
        │        record / playback / UI           │
        └────────────────┬────────────────────────┘
                         │ audio
                         ▼
        ┌─────────────────────────────────────────┐
        │           FastAPI backend               │
        └──┬──────────────┬──────────────┬────────┘
           │              │              │
           ▼              ▼              ▼
    ┌──────────┐   ┌────────────┐  ┌──────────┐
    │ Riva STT │   │    Tone    │  │   TTS    │
    │streaming │   │  analysis  │  │ endpoint │
    └────┬─────┘   │ (4 signals)│  └─────▲────┘
         │         └──────┬─────┘        │
         │ transcript     │ signals      │ reply
         └────────┬───────┘              │
                  ▼                      │
         ┌─────────────────┐             │
         │  gpt-oss-120b   │─────────────┘
         │  next question  │
         │  + reports      │
         └─────────────────┘
```

---

## Setup

Requires access to the Airtel endpoints, which are reachable from the VDI.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with the endpoint configuration:

```
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=gpt-oss-120b

RIVA_HOST=
RIVA_LANGUAGE_CODE=hi-IN
RIVA_USE_SSL=false

STT_REST_URL=
STT_CLIENT_ID=
STT_CLIENT_KEY=
STT_AUTH_BASIC=

TTS_URL=
TTS_AUTH_BASIC=
TTS_CLIENT_ID=
TTS_CLIENT_KEY=
```

`.env` is gitignored and must never be committed.

---

## Running the tone analysis on its own

The tone module requires no endpoints and runs entirely locally.

```bash
python -m src.tone sample.wav 30
```

The second argument is the word count of the utterance, used to compute speech
rate. It prints the compact summary the LLM receives, plus raw feature values.

---

## Latency design

A voice loop makes latency immediately obvious, so it is handled by design
rather than optimized afterwards:

- **Streaming output.** Sentences are sent to TTS as they complete, so speech
  begins before the LLM has finished generating.
- **Capped turn outputs.** Follow-up questions are short by construction.
  Long generations only happen once, at session close.
- **Compact signal serialization.** Acoustic features are binned into
  qualitative labels before reaching the prompt. Raw feature vectors are
  token-expensive and LLMs reason about them poorly.
- **Pre-filtered practice bank.** Only techniques matching the measured arousal
  are sent to the model, not the entire bank.
- **Sliding-window history.** Recent turns are kept verbatim; older turns are
  compressed into a summary.

---

## Design boundaries

InnerLoop is a reflective check-in and stabilization tool. It is not therapy and
does not diagnose.

- It reports what it **observed**, never what it concluded about someone's
  psychology.
- It probes by asking for elaboration ("tell me more about that"), never by
  reinterpreting ("you said X but you mean Y").
- The grounding techniques it suggests are stabilization practices, drawn from
  established frameworks (DBT distress tolerance, somatic and polyvagal-informed
  practice, CBT). Stabilization reduces acute distress; it does not treat
  underlying causes.
- If someone describes being in danger or at risk of harm, the normal flow stops
  and it directs them to professional support.
