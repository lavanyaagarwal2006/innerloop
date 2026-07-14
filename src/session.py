"""
Session orchestration: holds state across turns, calls the LLM, builds reports.

Latency decisions baked in (these are the SDLC's rules, in code):
  - Per-turn calls: small prompt, capped output. Short = fast.
  - Practice bank is NOT sent every turn. Only at close, and pre-filtered
    in Python first so we send 4 candidates, not all 12.
  - Conversation history uses a sliding window. Old turns get compressed.
  - Streaming so TTS can start on the first sentence, not the last token.
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import requests

from . import config, prompts
from .tone import ToneSignals

MAX_TURNS = 8
RECENT_TURNS_VERBATIM = 4  # older turns get summarized, not sent in full

_BANK = None


def bank() -> dict:
    global _BANK
    if _BANK is None:
        path = os.path.join(os.path.dirname(__file__), "practice_bank.json")
        with open(path, encoding="utf-8") as f:
            _BANK = json.load(f)
    return _BANK


@dataclass
class Turn:
    user_text: str
    assistant_text: str = ""
    tone: Optional[ToneSignals] = None
    hotspots: list = field(default_factory=list)


@dataclass
class Session:
    turns: list = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def arousal_trajectory(self) -> list:
        """Per-turn arousal. This is what the frontend plots."""
        return [t.tone.arousal if t.tone else "unknown" for t in self.turns]

    def dominant_arousal(self) -> str:
        """Most common arousal across the session. Drives technique selection."""
        vals = [a for a in self.arousal_trajectory() if a != "unknown"]
        if not vals:
            return "moderate"
        return max(set(vals), key=vals.count)

    def conversation_text(self, window: Optional[int] = None) -> str:
        """
        Sliding window. Recent turns verbatim, older ones compressed.
        Without this, turn 8's prompt carries everything ever said.
        """
        turns = self.turns
        if window and len(turns) > window:
            old, recent = turns[:-window], turns[-window:]
            topics = ", ".join(t.user_text[:40] for t in old)
            head = f"[Earlier in the session, they talked about: {topics}]\n\n"
        else:
            head, recent = "", turns

        body = "\n".join(
            f"InnerLoop: {t.assistant_text}\nThem: {t.user_text}"
            if t.assistant_text else f"Them: {t.user_text}"
            for t in recent
        )
        return head + body

    def tone_history_text(self) -> str:
        lines = []
        for i, t in enumerate(self.turns, 1):
            if t.tone:
                lines.append(f"Turn {i}: {t.tone.arousal} arousal, "
                             f"{t.tone.emotion_label} "
                             f"({t.tone.emotion_confidence:.0%}), "
                             f"{t.tone.pause_count} pauses"
                             + (f", {', '.join(t.tone.nonverbal_cues)}"
                                if t.tone.nonverbal_cues else ""))
        return "\n".join(lines) or "(no tone data)"


class LLM:
    def __init__(self):
        self.cfg = config.llm()
        self.headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

    def complete(self, system: str, user: str,
                 max_tokens: int = 120, stream: bool = False):
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [{"type": "text", "text": user}]},
            ],
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if not stream:
            r = requests.post(self.cfg.base_url, headers=self.headers,
                              json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

        return self._stream(payload)

    def _stream(self, payload):
        """Yields sentences, not tokens. TTS speaks whole sentences."""
        r = requests.post(self.cfg.base_url, headers=self.headers,
                          json=payload, timeout=120, stream=True)
        r.raise_for_status()

        buf = ""
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
            if not piece:
                continue
            buf += piece
            # Emit as soon as a sentence completes: TTS can start speaking
            # while the rest is still generating.
            while True:
                m = re.search(r"[.!?]\s", buf)
                if not m:
                    break
                sentence, buf = buf[:m.end()].strip(), buf[m.end():]
                if sentence:
                    yield sentence
        if buf.strip():
            yield buf.strip()

    def complete_json(self, system: str, user: str, max_tokens: int = 600) -> dict:
        """LLMs wrap JSON in fences even when told not to. Strip, then parse."""
        raw = self.complete(system, user, max_tokens=max_tokens)
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Last resort: grab the outermost braces.
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                return json.loads(m.group())
            raise ValueError(f"LLM did not return JSON:\n{raw[:400]}")


class InnerLoop:
    def __init__(self):
        self.llm = LLM()
        self.session = Session()

    def opener(self) -> str:
        """Never 'how are you'. Always a frame that gives them something to push against."""
        import random
        return random.choice(prompts.OPENERS)

    def next_question(self, user_text: str, tone: ToneSignals,
                      stream: bool = False):
        turn = Turn(user_text=user_text, tone=tone)
        self.session.turns.append(turn)

        n = len(self.session.turns)
        system = prompts.NEXT_TURN.format(
            boundaries=prompts.BOUNDARIES,
            turn_number=n,
            max_turns=MAX_TURNS,
            phase_instruction=prompts.phase_for(n, MAX_TURNS),
            tone_signals=tone.to_prompt_text() if tone else "(none)",
        )
        convo = self.session.conversation_text(window=RECENT_TURNS_VERBATIM)

        if stream:
            def gen():
                parts = []
                for sentence in self.llm.complete(system, convo,
                                                  max_tokens=100, stream=True):
                    parts.append(sentence)
                    yield sentence
                turn.assistant_text = " ".join(parts)
            return gen()

        q = self.llm.complete(system, convo, max_tokens=100).strip()
        turn.assistant_text = q
        return q

    def find_hotspots(self, transcript: str) -> list:
        try:
            out = self.llm.complete_json(
                "You output only valid JSON.",
                prompts.HOTSPOT.format(transcript=transcript),
                max_tokens=150,
            )
            return out.get("hotspots", [])
        except Exception as e:
            print(f"[hotspot] failed: {e}")
            return []

    def should_close(self) -> bool:
        return len(self.session.turns) >= MAX_TURNS

    def issue_report(self) -> dict:
        """Report 1: what was said. Observational only."""
        system = prompts.ISSUE_REPORT.format(
            boundaries=prompts.BOUNDARIES,
            conversation=self.session.conversation_text(),  # full, no window
            tone_history=self.session.tone_history_text(),
        )
        return self.llm.complete_json(system, "Write Report 1.", max_tokens=700)

    def fix_report(self) -> dict:
        """
        Report 2: the plan.

        Pre-filter the bank in Python BEFORE the prompt. Sending all 12
        techniques wastes tokens and gives the LLM room to pick something
        that contradicts the measured arousal. Send 4 valid candidates.
        """
        arousal = self.session.dominant_arousal()
        b = bank()

        candidates = [t for t in b["techniques"]
                      if t["arousal_target"] == arousal]
        if len(candidates) < 3:
            candidates = b["techniques"]

        tech_text = "\n".join(
            f"- {t['id']}: {t['name']}. {t['instructions']} "
            f"(why: {t['why']})"
            for t in candidates
        )
        style_text = "\n".join(
            f"- {s['id']}: {s['name']}. {s['format']} "
            f"(fits: {', '.join(s['signals'])})"
            for s in b["journaling_styles"]
        )

        system = prompts.FIX_REPORT.format(
            boundaries=prompts.BOUNDARIES,
            arousal_summary=f"{arousal} "
                            f"(trajectory: {' -> '.join(self.session.arousal_trajectory())})",
            techniques=tech_text,
            journaling_styles=style_text,
            conversation=self.session.conversation_text(),
        )
        out = self.llm.complete_json(system, "Write Report 2.", max_tokens=500)

        # Attach the full technique/style objects so the frontend has the
        # instructions without a second lookup.
        tid = out.get("technique_id")
        sid = out.get("journaling_style_id")
        out["technique"] = next((t for t in b["techniques"] if t["id"] == tid), None)
        out["journaling_style"] = next(
            (s for s in b["journaling_styles"] if s["id"] == sid), None)
        return out

    def close(self) -> dict:
        return {
            "issue_report": self.issue_report(),
            "fix_report": self.fix_report(),
            "arousal_trajectory": self.session.arousal_trajectory(),
            "turn_count": len(self.session.turns),
            "started_at": self.session.started_at,
        }

    def save(self, path: str = "sessions"):
        os.makedirs(path, exist_ok=True)
        stamp = self.session.started_at.replace(":", "-")[:19]
        fn = os.path.join(path, f"session_{stamp}.json")
        with open(fn, "w", encoding="utf-8") as f:
            json.dump({
                "started_at": self.session.started_at,
                "turns": [
                    {
                        "user": t.user_text,
                        "assistant": t.assistant_text,
                        "arousal": t.tone.arousal if t.tone else None,
                        "emotion": t.tone.emotion_label if t.tone else None,
                        "hotspots": t.hotspots,
                    }
                    for t in self.session.turns
                ],
            }, f, indent=2, ensure_ascii=False)
        return fn
