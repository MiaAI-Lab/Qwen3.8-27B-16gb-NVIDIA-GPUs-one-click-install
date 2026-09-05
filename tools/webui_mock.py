#!/usr/bin/env python3
"""A scripted stand-in for the model, for working on the UI without a GPU.

`MockClient` has the same `stream()` shape as webui_agent.ModelClient and
decides what to "say" from keywords in the last user message, so every path
the UI has to render - thinking, streaming text, a tool call, an approval, a
failed tool, a two-step loop - can be exercised in a second.

    python tools/chatui.py --mock --open
"""
from __future__ import annotations

import json
import os
import time

SCRIPTS = [
    (("search", "news", "look up", "latest"), "web_search",
     lambda text: {"query": text.strip()[:80] or "local llm news"}),
    (("fetch", "read the page", "http"), "web_fetch",
     lambda text: {"url": next((w for w in text.split() if w.startswith("http")),
                               "https://example.com")}),
    (("list", "files", "what is in"), "list_dir", lambda text: {"path": "."}),
    (("write", "create a file", "save"), "write_file",
     lambda text: {"path": "notes.md",
                   "content": "# Notes\n\nWritten by the mock model.\n"}),
    (("run", "python", "calculate"), "run_python",
     lambda text: {"code": "print(sum(range(101)))"}),
    (("ask", "which", "prefer"), "ask_user",
     lambda text: {"question": "Which format should the report use?",
                   "header": "Choose format",
                   "options": [{"label": "Markdown (Recommended)",
                                "description": "Easy to edit later"},
                               {"label": "HTML"}]}),
    (("plan", "steps", "multi"), "update_plan",
     lambda text: {"todos": [{"content": "read the config", "status": "in_progress"},
                             {"content": "write the summary", "status": "pending"}]}),
    (("background", "long", "watch"), "run_command",
     lambda text: {"command": "sleep 30", "run_in_background": True,
                   "description": "Start a long job"}),
]

THINKING = ("The user asked: {text}\nI should decide whether a tool helps here. "
            "{plan}\n")


class MockClient:
    """Same interface as ModelClient; no network, no weights."""

    def __init__(self, delay=None):
        # CHATUI_MOCK_DELAY slows the fake stream down, which is how the live
        # tokens/second readout gets something realistic to show.
        self.delay = float(os.environ.get("CHATUI_MOCK_DELAY") or delay or 0.01)
        self.model = "mock-model"

    @staticmethod
    def _last_user(messages):
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    return content
                return " ".join(p.get("text", "") for p in content or []
                                if isinstance(p, dict))
        return ""

    def _emit(self, text, kind="content"):
        for word in text.split(" "):
            time.sleep(self.delay)
            yield kind, word + " "

    def stream(self, messages, tools=None, sampling=None):
        text = self._last_user(messages)
        tool_names = {t["function"]["name"] for t in (tools or [])}
        already = [m for m in messages if m.get("role") == "tool"]

        if already:                       # second pass: summarise the results
            yield from self._emit(THINKING.format(
                text=text[:60], plan="The tool came back, so I can answer now."),
                "reasoning")
            summary = "\n\n".join(f"`{m.get('name')}` returned:\n\n```\n"
                                  f"{(m.get('content') or '')[:600]}\n```"
                                  for m in already[-2:])
            yield from self._emit("Here is what I found.")
            yield "content", "\n\n" + summary + "\n"
            yield "finish", "stop"
            yield "usage", {"prompt_tokens": 900, "completion_tokens": 120}
            return

        lowered = text.lower()
        for keywords, name, make_args in SCRIPTS:
            if name in tool_names and any(k in lowered for k in keywords):
                yield from self._emit(THINKING.format(
                    text=text[:60], plan=f"I will call {name}."), "reasoning")
                yield from self._emit(f"Let me use `{name}` for that.")
                yield "tool_calls", [{
                    "id": f"call_{int(time.time() * 1000) % 100000}",
                    "type": "function",
                    "function": {"name": name,
                                 "arguments": json.dumps(make_args(text))}}]
                yield "finish", "tool_calls"
                return

        yield from self._emit(THINKING.format(
            text=text[:60], plan="No tool is needed; I can answer directly."),
            "reasoning")
        yield from self._emit(
            "This is the mock model. It streams words like the real one, folds "
            "its thinking above, and renders Markdown:")
        yield "content", ("\n\n- lists\n- `inline code`\n\n```python\n"
                          "def hello():\n    return 'world'\n```\n\n"
                          "Ask me to *search*, *list files*, *write* a file or "
                          "*run* python to see tool calls and approvals.\n")
        yield "finish", "stop"
        yield "usage", {"prompt_tokens": 420, "completion_tokens": 90}
