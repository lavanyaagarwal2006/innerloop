# InnerLoop

Structured voice check-in companion. Leads a conversation, reads tone and
content together, ends every session with a concrete next step.

## Setup (inside the VDI)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Fill in .env with the real endpoint values. Never commit this file.
```

Add to `.gitignore` immediately:
```
.env
out/
sessions/
venv/
```

## Test the endpoints (do this first)

Run in this order. Each one answers a specific question.

```bash
# 1. Does the LLM work? Does it do Hinglish? Does it return clean JSON?
python tests/test_llm.py

# 2. What does the STT endpoint actually return?
#    Record ~10s of yourself talking, save as test.wav
python tests/test_stt_rest.py test.wav

# 3. What audio format does TTS return? How slow is a long report?
python tests/test_tts.py

# 4. Do partial transcripts arrive while you speak? Word timestamps?
python tests/test_riva_streaming.py test.wav

# 5. THE IMPORTANT ONE. How long does one full turn actually take?
python tests/test_full_loop.py test.wav
```

### What to write down

From test 2 (STT):
- [ ] Response format: JSON or plain text?
- [ ] Word-level timestamps included? (hotspot analysis needs these)
- [ ] Confidence scores included?
- [ ] Which language flag handles Hinglish better, ENG or HIN?

From test 3 (TTS):
- [ ] Audio format and sample rate (needed for browser playback)
- [ ] ms per word (multiply by report length = end-of-session wait)

From test 5 (full loop):
- [ ] Which stage is the bottleneck? Optimize that one, not the others.
- [ ] Under 2s = feels conversational. Over 4s = needs the streaming fixes.

## Test the tone module

Works offline, no endpoints needed.

```bash
python -m innerloop.tone test.wav 25    # 25 = word count from the transcript
```

Prints the compact summary the LLM will receive, plus raw values for debugging.

## Structure

```
innerloop/
  config.py          # loads .env, no secrets in code
  tone.py            # the 4 signals (SDLC 4.3)
  prompts.py         # conversation logic + safety boundaries
  session.py         # state across turns, reports, latency handling
  practice_bank.json # techniques + journaling styles
tests/               # endpoint tests, run these first
```

## Design rules (do not drift from these)

- **Audio gives arousal. Text gives valence.** Never read positive/negative
  from acoustics; an excited voice and an angry voice look the same. That is
  the valence gap, and the architecture is built around it.
- **Observe, never interpret.** "Your pace picked up when work came up" is
  fine. "You are anxious about work" is not.
- **InnerLoop leads.** Never opens with "how are you". Always a frame.
- **Every session ends with one concrete next step.** Not a menu.

## Security

Endpoints, tokens, and hostnames stay in the VDI. `.env` is never committed,
never copied out, never pasted into chat tools.
