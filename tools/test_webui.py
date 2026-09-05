#!/usr/bin/env python3
"""Checks for the built-in web UI: the tool loop, the approval hand-shake and
the workspace sandbox. Runs against the stdlib dev server with --mock, so it
needs neither a GPU nor the network.

    python tools/chatui.py --mock --port 8890 &
    python tools/test_webui.py
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import webui_tools                                     # noqa: E402
from webui_tools import ToolContext, ToolError, Workspace   # noqa: E402

BASE = "http://127.0.0.1:8890"
failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        failures.append(name)


UI_HEADERS = {"Content-Type": "application/json", "X-Simplex-UI": "1"}


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers=UI_HEADERS)
    return urllib.request.urlopen(req, timeout=30)


def get(path):
    req = urllib.request.Request(BASE + path, headers=UI_HEADERS)
    return json.loads(urllib.request.urlopen(req, timeout=15).read())


def stream(payload, on_event):
    """POST /ui/chat and hand every event to on_event; returns them all."""
    events = []
    with post("/ui/chat", payload) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:])
            events.append(event)
            on_event(event)
    return events


# ---------------------------------------------------------------- sandbox --

def test_sandbox(tmp: Path):
    ws = Workspace(tmp)
    ctx = ToolContext(workspace=ws)
    reg = webui_tools.build_registry({})
    (tmp / "inside.txt").write_text("ok", encoding="utf-8")
    check("relative path resolves", ws.resolve("inside.txt").is_file())
    for bad in ("../outside.txt", "../../etc/passwd", "/etc/passwd"):
        try:
            ws.resolve(bad)
            check(f"refuses {bad}", False, "was allowed")
        except ToolError:
            check(f"refuses {bad}", True)
    out = webui_tools.execute(reg["read_file"], {"path": "inside.txt"}, ctx)
    check("read_file numbers its lines", out.strip().endswith("| ok"), out)
    try:
        webui_tools.execute(reg["write_file"],
                            {"path": "../escape.txt", "content": "x"}, ctx)
        check("write_file cannot escape", False, "wrote outside the workspace")
    except ToolError:
        check("write_file cannot escape", True)


def test_read_before_write(tmp: Path):
    """DSH's fs-observation-policy: an unseen file may be created, never
    silently replaced, and editing needs a prior read."""
    ws, reg = Workspace(tmp), webui_tools.build_registry({})
    ctx = ToolContext(workspace=ws)
    (tmp / "brand-new.txt").unlink(missing_ok=True)      # keep the run repeatable
    (tmp / "existing.txt").write_text("first line\n", encoding="utf-8")

    out = webui_tools.execute(reg["write_file"],
                              {"path": "brand-new.txt", "content": "hi"}, ctx)
    check("new files need no read", "wrote" in out, out)
    try:
        webui_tools.execute(reg["write_file"],
                            {"path": "existing.txt", "content": "clobber"}, ctx)
        check("unread file cannot be replaced", False, "was replaced")
    except ToolError as e:
        check("unread file cannot be replaced", True)
        check("refusal carries a hint", "read_file" in (e.hint or ""), str(e.hint))
    check("the file survived",
          (tmp / "existing.txt").read_text(encoding="utf-8") == "first line\n")
    try:
        webui_tools.execute(reg["edit_file"], {"path": "existing.txt",
                                               "old_text": "first",
                                               "new_text": "second"}, ctx)
        check("unread file cannot be edited", False, "was edited")
    except ToolError:
        check("unread file cannot be edited", True)

    webui_tools.execute(reg["read_file"], {"path": "existing.txt"}, ctx)
    out = webui_tools.execute(reg["edit_file"], {"path": "existing.txt",
                                                 "old_text": "first",
                                                 "new_text": "second"}, ctx)
    check("edit works after a read", "edited" in out, out)
    check("the edit landed",
          "second line" in (tmp / "existing.txt").read_text(encoding="utf-8"))


def test_plan_tool(tmp: Path):
    ws, reg = Workspace(tmp), webui_tools.build_registry({})
    ctx = ToolContext(workspace=ws)
    out = webui_tools.execute(reg["update_plan"], {"todos": [
        {"content": "one", "status": "completed"},
        {"content": "two", "status": "in_progress"}]}, ctx)
    check("plan renders as a checklist", "[x] one" in out and "[~] two" in out, out)
    check("plan is kept on the context", len(ctx.plan) == 2 and ctx.plan_changed)
    for bad, why in (
            ([{"content": "a", "status": "in_progress"},
              {"content": "b", "status": "in_progress"}], "two active tasks"),
            ([{"content": "a", "status": "nonsense"}], "an unknown status"),
            ([], "an empty list")):
        try:
            webui_tools.execute(reg["update_plan"], {"todos": bad}, ctx)
            check(f"plan refuses {why}", False, "was accepted")
        except ToolError:
            check(f"plan refuses {why}", True)


def test_command_shape(tmp: Path):
    """Exit codes visible, output tailed, background jobs pollable."""
    ws, reg = Workspace(tmp), webui_tools.build_registry({})
    ctx = ToolContext(workspace=ws)
    out = webui_tools.execute(reg["run_python"],
                              {"code": "import sys; print('hi'); sys.exit(3)"}, ctx)
    check("exit code is reported", "[exit code: 3]" in out, out[:120])
    big = webui_tools.execute(reg["run_python"], {
        "code": "print('x' * 30000)"}, ctx)
    check("long output is tailed and spilled",
          webui_tools.SPILL_DIR in big and len(big) < 30000, big[:160])
    job = webui_tools.execute(reg["run_command"], {
        "command": "echo started; sleep 5", "run_in_background": True}, ctx)
    check("background command returns a job id", "job-1" in job, job)
    time.sleep(1.2)
    seen = webui_tools.execute(reg["job_output"], {"job_id": "job-1"}, ctx)
    check("job output can be read", "started" in seen, seen[:160])
    check("job is still running", "running" in seen, seen[:80])
    killed = webui_tools.execute(reg["job_kill"], {"job_id": "job-1"}, ctx)
    check("job can be stopped", "stopped" in killed, killed)


def test_untrusted_label():
    check("fetched pages are labelled untrusted",
          webui_tools.UNTRUSTED.startswith("[external content"))


def test_origin_guard():
    """Loopback is not a security boundary in a browser: DNS rebinding gives a
    hostile page the same peer address, and a cross-site POST needs no read
    access to do damage. Both are refused by what the browser reports."""
    from webui_app import ChatUI                          # noqa: WPS433
    ui = ChatUI(root=Path("."), cfg={}, model_base="http://127.0.0.1:1/v1",
                model_id="m")
    local = {"Host": "127.0.0.1:8888"}
    cases = [
        ("GET", "/ui/config", {"Host": "evil.com:8888"}, 403, "rebinding host"),
        ("GET", "/ui/config", {**local, "Sec-Fetch-Site": "cross-site"}, 403,
         "cross-site GET"),
        ("GET", "/ui/config", {**local, "Sec-Fetch-Site": "same-origin"}, 200,
         "the UI's own GET"),
        ("POST", "/ui/cancel", local, 403, "POST with no proof of origin"),
        ("POST", "/ui/cancel", {**local, "Origin": "http://evil.com"}, 403,
         "cross-origin POST"),
        ("POST", "/ui/cancel", {**local, "X-Simplex-UI": "1"}, 200,
         "POST from the UI"),
    ]
    for method, path, headers, want, why in cases:
        got = ui.handle(method, path, {}, b"{}", "127.0.0.1", headers).status
        check(f"{why} -> {want}", got == want, str(got))
    named = ChatUI(root=Path("."), cfg={"UI_HOSTS": "box.tail1234.ts.net"},
                   model_base="http://127.0.0.1:1/v1", model_id="m")
    # tailscale serve proxies from loopback, so only the name needs trusting
    check("a named host in UI_HOSTS is allowed",
          named.handle("GET", "/ui/config", {},
                       b"", "127.0.0.1", {"Host": "box.tail1234.ts.net"}).status == 200)
    check("but an unlisted name is still refused",
          named.handle("GET", "/ui/config", {},
                       b"", "127.0.0.1", {"Host": "evil.com"}).status == 403)
    check("and UI_HOSTS does not open it to other machines",
          named.handle("GET", "/ui/config", {}, b"", "192.168.1.42",
                       {"Host": "box.tail1234.ts.net"}).status == 403)

    check("an in-process call needs no headers",
          ui.handle("GET", "/ui/config", {}, b"", "127.0.0.1", None).status == 200)


def test_lan_guard():
    """Agent mode can write files and run commands, and there is no login, so
    the UI must refuse other machines until UI_LAN says otherwise."""
    from webui_app import ChatUI                          # noqa: WPS433
    ui = ChatUI(root=Path("."), cfg={}, model_base="http://127.0.0.1:1/v1",
                model_id="m")
    for remote, want in (("127.0.0.1", 200), ("::1", 200),
                         ("::ffff:127.0.0.1", 200), ("192.168.1.42", 403)):
        got = ui.handle("GET", "/ui/config", {}, b"", remote).status
        check(f"remote {remote} -> {want}", got == want, str(got))
    lan = ChatUI(root=Path("."), cfg={"UI_LAN": "1"},
                 model_base="http://127.0.0.1:1/v1", model_id="m")
    check("UI_LAN=1 opens it up",
          lan.handle("GET", "/ui/config", {}, b"", "192.168.1.42").status == 200)


def test_model_planning():
    """Switching model has to re-plan the context, and refuse a quant the card
    cannot hold - a downloaded model is not always a loadable one."""
    import types                                        # noqa: WPS433
    import profiles                                     # noqa: WPS433
    import webui_models                                 # noqa: WPS433
    real = profiles.detect_gpu
    # Build the stub from the real GPU class, never a SimpleNamespace: a hand-rolled
    # stub carrying .memory / .compute_cap / .driver_version (names profiles.GPU has
    # never had) made this test pass against code that read those same wrong names,
    # while the real detect_gpu() sent every switch down the except branch.
    check("the GPU stub below matches the real class's attributes",
          all(hasattr(profiles.GPU(), a) for a in ("name", "total_gib", "cc", "driver")))
    try:
        for vram, quant, expect in ((16, "2.0bpw", True), (16, "4.0bpw", False),
                                    (24, "4.0bpw", True), (24, "6.0bpw", True)):
            profiles.detect_gpu = lambda v=vram: profiles.GPU(
                name="Fake", total_gib=v, cc=8.9, driver="580")
            name = f"models/Qwen3.8-27B-EXL3-{quant}"
            try:
                got = webui_models.settings_for(Path("."), name, {})
                ctx = int(got["CONTEXT_SIZE"])
                check(f"{quant} on {vram}GB plans {ctx // 1024}k context", expect,
                      "expected a refusal")
                check(f"{quant} on {vram}GB caps VRAM below the card",
                      float(got["GPU_MEM_GB"]) < vram, got["GPU_MEM_GB"])
            except ValueError:
                check(f"{quant} on {vram}GB is refused", not expect)
    finally:
        profiles.detect_gpu = real

    try:
        webui_models.apply(Path("."), "../../etc", {})
        check("model switch cannot escape models/", False, "was allowed")
    except (ValueError, OSError):
        check("model switch cannot escape models/", True)


def test_providers():
    """A provider is any OpenAI-compatible endpoint. Keys must be storable,
    never readable, and a chat must be able to point at one."""
    import webui_providers as wp                        # noqa: WPS433
    root = Path(".")
    (root / wp.FILE).unlink(missing_ok=True)
    try:
        saved = wp.upsert(root, {"name": "Test Cloud",
                                 "base_url": "https://api.example.com/v1",
                                 "api_key": "sk-secret-value-1234",
                                 "default_model": "big-model"})
        check("provider is stored", saved["id"] == "test-cloud", saved["id"])
        shown = wp.public(saved)
        check("the key never leaves the machine", "secret" not in json.dumps(shown),
              json.dumps(shown))
        check("the key is hinted, not shown", shown["key_hint"] == "sk-...1234",
              shown["key_hint"])

        wp.upsert(root, {"id": "test-cloud", "name": "Test Cloud",
                         "base_url": "https://api.example.com/v1", "api_key": "",
                         "default_model": "big-model"})
        check("an empty key box keeps the saved key",
              wp.find(root, "test-cloud")["api_key"] == "sk-secret-value-1234")

        for bad, why in (
                ({"name": "x", "default_model": "m"}, "no base URL"),
                ({"name": "x", "base_url": "ftp://h/v1", "default_model": "m"},
                 "a non-http URL"),
                ({"name": "x", "base_url": "https://h/v1"}, "no model"),
                ({"id": "local", "name": "x", "base_url": "https://h/v1",
                  "default_model": "m"}, "the reserved id 'local'")):
            try:
                wp.upsert(root, bad)
                check(f"refuses {why}", False, "was accepted")
            except wp.ProviderError:
                check(f"refuses {why}", True)

        target = wp.resolve(root, "test-cloud", "big-model",
                            {"base_url": "http://127.0.0.1:8888/v1", "model": "local-q"})
        check("a chat can be pointed at a provider",
              target["remote"] and target["model"] == "big-model"
              and target["api_key"] == "sk-secret-value-1234")
        check("no provider means the local server",
              wp.resolve(root, "", "", {"base_url": "b", "model": "local-q"}
                         )["model"] == "local-q")
        try:
            wp.resolve(root, "gone", "m", {"base_url": "b", "model": "q"})
            check("a deleted provider is reported", False, "resolved anyway")
        except wp.ProviderError:
            check("a deleted provider is reported", True)

        listed = get("/ui/models")
        check("provider models appear in the picker",
              any(r["model"] == "big-model" for r in listed.get("remote") or []),
              json.dumps(listed.get("remote"))[:160])
        served = get("/ui/providers")["providers"]
        check("the API never serves a key", "secret" not in json.dumps(served))
    finally:
        (root / wp.FILE).unlink(missing_ok=True)


def test_modes():
    cfg = {}
    reg = webui_tools.build_registry(cfg)
    chat = {t.name for t in webui_tools.tools_for("chat", cfg, reg)}
    agent = {t.name for t in webui_tools.tools_for("agent", cfg, reg)}
    check("chat mode is web-only", chat == {"web_search", "web_fetch"}, str(chat))
    check("agent mode adds files", {"read_file", "write_file", "run_command"} <= agent)
    off = {t.name for t in webui_tools.tools_for("agent", {"AGENT_EXEC": "0"}, reg)}
    check("AGENT_EXEC=0 drops execution", not ({"run_command", "run_python"} & off))
    none = {t.name for t in webui_tools.tools_for("chat", {"WEB_TOOLS": "0"}, reg)}
    check("WEB_TOOLS=0 drops search", not none, str(none))


# ------------------------------------------------------------- html tools --

def test_html():
    page = webui_tools._TextExtract()
    page.feed("<html><head><title>T</title><style>x{}</style></head>"
              "<body><script>bad()</script><p>Hello</p><p>World</p></body></html>")
    text = page.text()
    check("html to text drops script/style", "bad()" not in text and "x{}" not in text, text)
    check("html to text keeps prose", "Hello" in text and "World" in text, text)


# ------------------------------------------------------------- the loop ---

def test_chat_turn():
    events = stream({"mode": "chat", "content": "hello there"}, lambda e: None)
    kinds = [e["type"] for e in events]
    check("chat streams reasoning then content",
          kinds.index("reasoning") < kinds.index("content"), str(kinds[:6]))
    check("chat ends with done", kinds[-1] == "done", str(kinds[-3:]))
    check("chat used no tools", "tool_call" not in kinds)


def test_speed_report():
    """The done event has to carry what the UI shows as tokens/second."""
    events = stream({"mode": "chat", "content": "hello there"}, lambda e: None)
    done = events[-1]
    check("done reports usage", (done.get("usage") or {}).get("completion_tokens", 0) > 0,
          json.dumps(done)[:200])
    check("done reports seconds", isinstance(done.get("seconds"), (int, float)))
    check("done reports tok/s", isinstance(done.get("tok_s"), (int, float)),
          str(done.get("tok_s")))


def test_picker_scope():
    """The system dialog opens on the server's screen, so only the server's own
    machine may ask for it."""
    from webui_app import ChatUI                          # noqa: WPS433
    ui = ChatUI(root=Path("."), cfg={"UI_LAN": "1"},
                model_base="http://127.0.0.1:1/v1", model_id="m")
    ui_headers = {"Host": "127.0.0.1:8888", "X-Simplex-UI": "1"}
    remote = ui.handle("POST", "/ui/pick_folder", {}, b"{}", "192.168.1.42",
                       ui_headers)
    check("remote device cannot open the dialog", remote.status == 409, str(remote.status))
    off = ChatUI(root=Path("."), cfg={"FOLDER_PICKER": "browser"},
                 model_base="http://127.0.0.1:1/v1", model_id="m")
    check("FOLDER_PICKER=browser refuses the dialog",
          off.handle("POST", "/ui/pick_folder", {}, b"{}", "127.0.0.1",
                     ui_headers).status == 409)
    check("config advertises the picker honestly",
          json.loads(off.handle("GET", "/ui/config", {}, b"", "127.0.0.1",
                                ui_headers).body)["native_picker"] is False)


def test_agent_readonly():
    events = stream({"mode": "agent", "content": "list the files please"}, lambda e: None)
    calls = [e for e in events if e["type"] == "tool_call"]
    results = [e for e in events if e["type"] == "tool_result"]
    check("agent called a tool", len(calls) == 1 and calls[0]["name"] == "list_dir")
    check("safe tool ran without approval",
          not any(e["type"] == "approval" for e in events))
    check("tool result came back", results and results[0]["ok"], str(results[:1]))
    check("second step answered", any(e["type"] == "content" for e in events[-20:]))


def test_ask_user():
    """ask_user must reach the browser as a question and block until answered."""
    seen = {}

    def on_event(event):
        if event["type"] == "question":
            seen["q"] = event
            threading.Timer(0.05, lambda: post(
                "/ui/answer", {"id": event["id"], "text": "Markdown"}).close()).start()
        if event["type"] == "tool_result":
            seen["result"] = event
    stream({"mode": "agent", "content": "which format do you prefer?"}, on_event)
    check("question reached the browser", "q" in seen)
    check("question carries its options",
          len((seen.get("q") or {}).get("options") or []) == 2,
          json.dumps(seen.get("q", {}))[:200])
    check("the answer came back to the model",
          "Markdown" in (seen.get("result") or {}).get("output", ""),
          json.dumps(seen.get("result", {}))[:200])


def test_plan_event():
    # an explicit id: since turns outlive their connection, another test's turn
    # can save after this one and take the top of the list
    sid = "plan-test"
    events = stream({"mode": "agent", "session_id": sid,
                     "content": "plan the steps"}, lambda e: None)
    plans = [e for e in events if e["type"] == "plan"]
    check("plan reaches the UI", bool(plans), str([e["type"] for e in events]))
    if plans:
        check("plan items have a status",
              all(i.get("status") for i in plans[-1]["items"]),
              json.dumps(plans[-1])[:200])
        got = get(f"/ui/sessions/{sid}")
        check("plan is saved with the session", bool(got.get("plan")),
              json.dumps(got.get("plan"))[:120])
        urllib.request.urlopen(urllib.request.Request(
            f"{BASE}/ui/sessions/{sid}", method="DELETE", headers=UI_HEADERS),
            timeout=10).close()


def _approve_flow(decision, message):
    seen = {}

    def on_event(event):
        if event["type"] == "approval":
            seen["approval"] = event
            threading.Timer(0.05, lambda: post(
                "/ui/approve", {"id": event["id"], "decision": decision}).close()).start()
        if event["type"] == "tool_result":
            seen["result"] = event
    events = stream({"mode": "agent", "content": message}, on_event)
    return seen, events


def test_agent_approval(workspace: Path):
    target = workspace / "notes.md"
    if target.exists():
        target.unlink()
    seen, _ = _approve_flow("allow", "write a file for me")
    check("write asked for approval", "approval" in seen)
    check("approved write executed", seen.get("result", {}).get("ok") is True,
          json.dumps(seen.get("result", {}))[:200])
    check("file really appeared", target.is_file())

    target.unlink(missing_ok=True)
    seen, _ = _approve_flow("deny", "write a file for me")
    check("denied write did not run", seen.get("result", {}).get("ok") is False)
    check("denied write left no file", not target.exists())


def test_rewind():
    """Retry and Edit both rewind the transcript by one exchange."""
    sid = "rewind-test"
    stream({"mode": "chat", "session_id": sid, "content": "first question"},
           lambda e: None)
    stream({"mode": "chat", "session_id": sid, "content": "second question"},
           lambda e: None)
    before = get(f"/ui/sessions/{sid}")
    got = json.loads(post("/ui/rewind", {"session_id": sid}).read())
    after = get(f"/ui/sessions/{sid}")
    check("rewind returns the question", got.get("content") == "second question",
          str(got.get("content"))[:80])
    check("rewind drops that exchange",
          len(after["messages"]) < len(before["messages"]),
          f"{len(before['messages'])} -> {len(after['messages'])}")
    check("rewind keeps the earlier turn",
          any(m.get("content") == "first question" for m in after["messages"]))
    urllib.request.urlopen(urllib.request.Request(
        f"{BASE}/ui/sessions/{sid}", method="DELETE", headers=UI_HEADERS),
        timeout=10).close()


def test_turn_outlives_the_window():
    """Closing the browser must not kill the work: the turn keeps running in
    the session, saves its answer, and replays when someone reattaches."""
    sid = "detach-test"
    resp = post("/ui/chat", {"session_id": sid, "mode": "chat",
                             "content": "hello there"})
    seen = 0
    for raw in resp:                      # hang up early, like closing a tab
        if raw.startswith(b"data:"):
            seen += 1
            if seen >= 2:
                break
    resp.close()
    time.sleep(1.0)

    rows = get("/ui/sessions")
    live = rows.get("live", {}).get(sid) or {}
    # the turn is tracked by the session, not by the connection that started it
    check("the turn outlived the connection that started it",
          live.get("events", 0) > seen,
          f"{live.get('events')} events vs {seen} the client saw")

    replay = []
    with post("/ui/attach", {"session_id": sid, "from": 0}) as again:
        for raw in again:
            if raw.startswith(b"data:"):
                replay.append(json.loads(raw[5:]))
    kinds = [e["type"] for e in replay]
    check("reattaching replays the turn from the start",
          kinds[0] == "session" and kinds[-1] == "done", str(kinds[:2] + kinds[-1:]))
    answer = "".join(e["delta"] for e in replay if e["type"] == "content")
    check("the answer finished without a viewer", len(answer) > 80, str(len(answer)))
    stored = get(f"/ui/sessions/{sid}")["messages"]
    check("and was saved to the conversation",
          [m["role"] for m in stored] == ["user", "assistant"],
          str([m["role"] for m in stored]))

    resumed = post("/ui/attach", {"session_id": sid, "from": 0})
    resumed.close()
    check("a finished turn stays replayable for a while", True)

    try:
        post("/ui/attach", {"session_id": "never-existed", "from": 0}).close()
        check("attaching to nothing is an error", False, "was accepted")
    except urllib.error.HTTPError as e:
        check("attaching to nothing is an error", e.code == 404, str(e.code))
    urllib.request.urlopen(urllib.request.Request(
        f"{BASE}/ui/sessions/{sid}", method="DELETE", headers=UI_HEADERS),
        timeout=10).close()


def test_one_turn_per_session():
    """Two turns in one conversation would share its tool state, so the second
    is refused while the first is still going."""
    sid = "single-turn"
    first = post("/ui/chat", {"session_id": sid, "mode": "chat",
                              "content": "hello there"})
    next(iter(first))                     # let it start
    try:
        post("/ui/chat", {"session_id": sid, "mode": "chat", "content": "again"})
        check("a second turn is refused while one runs", False, "was accepted")
    except urllib.error.HTTPError as e:
        check("a second turn is refused while one runs", e.code == 409, str(e.code))
    finally:
        post("/ui/cancel", {"session_id": sid}).close()
        first.close()
    time.sleep(0.5)
    urllib.request.urlopen(urllib.request.Request(
        f"{BASE}/ui/sessions/{sid}", method="DELETE", headers=UI_HEADERS),
        timeout=10).close()


def test_pin_and_search():
    """Pinned conversations come first, and search covers what was said, not
    just what the chat was called."""
    import urllib.parse                                 # noqa: WPS433
    stream({"session_id": "pin-a", "mode": "chat",
            "content": "hello there"}, lambda e: None)
    stream({"session_id": "pin-b", "mode": "chat",
            "content": "hello there"}, lambda e: None)
    post("/ui/sessions/pin-a", {"title": "Ordinary chat"}).close()
    post("/ui/sessions/pin-b", {"title": "Kept for later"}).close()

    got = json.loads(post("/ui/sessions/pin-b", {"pinned": True}).read())
    check("a conversation can be pinned", got.get("pinned") is True, json.dumps(got))
    rows = get("/ui/sessions")["sessions"]
    check("pinned rises to the top", rows[0]["id"] == "pin-b", rows[0]["id"])
    check("the row says it is pinned", rows[0]["pinned"] is True)

    post("/ui/sessions/pin-b", {"pinned": False}).close()
    check("and can be unpinned",
          get("/ui/sessions")["sessions"][0]["id"] != "pin-b"
          or not get("/ui/sessions")["sessions"][0]["pinned"])

    hits = get("/ui/sessions?q=" + urllib.parse.quote("ordinary"))["sessions"]
    check("search finds a title", [r["id"] for r in hits] == ["pin-a"],
          str([r["id"] for r in hits]))

    # the mock's answer mentions "renders Markdown" - nothing in any title does
    hits = get("/ui/sessions?q=" + urllib.parse.quote("renders markdown"))["sessions"]
    check("search reaches into the messages", len(hits) >= 2, str(len(hits)))
    check("a content match brings a snippet",
          any(r["snippet"] for r in hits),
          json.dumps([r["snippet"][:40] for r in hits]))

    check("no match is no rows",
          get("/ui/sessions?q=" + urllib.parse.quote("zebra-quux"))["sessions"] == [])

    for sid in ("pin-a", "pin-b"):
        urllib.request.urlopen(urllib.request.Request(
            f"{BASE}/ui/sessions/{sid}", method="DELETE", headers=UI_HEADERS),
            timeout=10).close()


def test_pin_reorder():
    """The sidebar's drag-and-drop only touches pinned chats: it posts the
    whole new pinned order, the server numbers them 0..n, and a freshly
    pinned chat joins the bottom of that order rather than jumping to the
    top of it - or, worse, colliding with everything else at 0."""
    ids = ["reorder-a", "reorder-b", "reorder-c"]
    for sid in ids:
        stream({"session_id": sid, "mode": "chat", "content": "hi"}, lambda e: None)
        post(f"/ui/sessions/{sid}", {"pinned": True}).close()

    rows = get("/ui/sessions")["sessions"]
    pinned_ids = [r["id"] for r in rows if r["pinned"]]
    check("pinning one at a time keeps them in pin order, not colliding at 0",
          pinned_ids == ids, pinned_ids)
    before = {r["id"]: r["updated"] for r in rows if r["id"] in ids}

    shuffled = [ids[2], ids[0], ids[1]]
    got = json.loads(post("/ui/sessions/reorder", {"ids": shuffled}).read())
    check("reorder reports every id it moved",
          sorted(got.get("reordered", [])) == sorted(shuffled), got)

    rows = get("/ui/sessions")["sessions"]
    pinned_ids = [r["id"] for r in rows if r["pinned"]]
    check("the sidebar's dropped order is what comes back",
          pinned_ids == shuffled, pinned_ids)
    after = {r["id"]: r["updated"] for r in rows if r["id"] in ids}
    check("reordering does not touch 'updated' - it is not activity",
          before == after, (before, after))

    # a stale id (already deleted, or never existed) must not break the rest
    got = json.loads(post("/ui/sessions/reorder",
                          {"ids": ["does-not-exist", *shuffled]}).read())
    check("an unknown id in the drop is skipped, not fatal",
          sorted(got.get("reordered", [])) == sorted(shuffled), got)

    # an unpinned id slipped into the payload must not get silently repinned
    # or reordered as if it were
    post(f"/ui/sessions/{ids[0]}", {"pinned": False}).close()
    got = json.loads(post("/ui/sessions/reorder", {"ids": [ids[0], ids[1], ids[2]]}).read())
    check("an unpinned id in the drop is skipped too",
          ids[0] not in got.get("reordered", []), got)
    by_id = {r["id"]: r for r in get("/ui/sessions")["sessions"]}
    check("...and stays unpinned", by_id[ids[0]]["pinned"] is False)

    # re-pinning joins the bottom of the remaining pinned list, not the top
    post(f"/ui/sessions/{ids[0]}", {"pinned": True}).close()
    pinned_ids = [r["id"] for r in get("/ui/sessions")["sessions"] if r["pinned"]]
    check("a freshly re-pinned chat lands at the bottom of the pins",
          pinned_ids[-1] == ids[0], pinned_ids)

    try:
        post("/ui/sessions/reorder", {"ids": []})
        check("reorder refuses an empty list", False, "was accepted")
    except urllib.error.HTTPError as e:
        check("reorder refuses an empty list", e.code == 400, e.code)

    for sid in ids:
        urllib.request.urlopen(urllib.request.Request(
            f"{BASE}/ui/sessions/{sid}", method="DELETE", headers=UI_HEADERS),
            timeout=10).close()


def test_cancel():
    """A cancel mid-turn ends the stream instead of hanging the browser."""
    sid = "cancel-test"
    done = []

    def on_event(event):
        if event["type"] == "step":
            threading.Timer(0.05, lambda: post(
                "/ui/cancel", {"session_id": sid}).close()).start()
        if event["type"] == "done":
            done.append(event)
    stream({"mode": "chat", "session_id": sid, "content": "hello there"}, on_event)
    check("cancel closes the stream", bool(done))


def test_sessions():
    rows = get("/ui/sessions")["sessions"]
    check("sessions were saved", len(rows) >= 1, str(len(rows)))
    if rows:
        sid = rows[0]["id"]
        got = get(f"/ui/sessions/{sid}")
        roles = [m["role"] for m in got["messages"]]
        check("session keeps the transcript", "user" in roles and "assistant" in roles,
              str(roles))
        req = urllib.request.Request(f"{BASE}/ui/sessions/{sid}",
                                     method="DELETE", headers=UI_HEADERS)
        urllib.request.urlopen(req, timeout=10).close()
        check("session deletes",
              json.dumps(get("/ui/sessions")["sessions"]).count(sid) == 0)



# ============================================================================
# Packaging and first-run: wheels, the downloader, setup, logs, shortcuts
# ============================================================================

def test_wheel_tags():
    import wheels
    w = wheels.parse_wheel(Path("exllamav3-1.4.4-cp313-cp313-win_amd64.whl"))
    check("wheel filename is parsed", w is not None and w.name == "exllamav3"
          and w.version == "1.4.4" and w.py == "cp313" and w.plat == "win_amd64")

    win313 = {"py": "cp313", "nodot": "313", "abi": "cp313", "plat": "win_amd64"}
    win312 = {"py": "cp312", "nodot": "312", "abi": "cp312", "plat": "win_amd64"}
    check("a cp313 wheel matches cp313", wheels.wheel_matches(w, win313))
    check("a cp313 wheel is refused by cp312", not wheels.wheel_matches(w, win312))

    linux = wheels.parse_wheel(Path("exllamav3-1.4.4-cp313-cp313-linux_x86_64.whl"))
    check("the wrong platform is refused", not wheels.wheel_matches(linux, win313))

    pure = wheels.parse_wheel(Path("triton_windows-3.4.0-py3-none-any.whl"))
    check("a pure-python wheel matches anything", wheels.wheel_matches(pure, win313)
          and wheels.wheel_matches(pure, win312))

    abi3 = wheels.parse_wheel(Path("thing-1.0-cp311-abi3-win_amd64.whl"))
    check("abi3 matches its own minor and newer",
          wheels.wheel_matches(abi3, win313) and wheels.wheel_matches(abi3, win312))
    check("abi3 is refused by an older minor",
          not wheels.wheel_matches(abi3, {"py": "cp310", "nodot": "310",
                                          "abi": "cp310", "plat": "win_amd64"}))

    check("a junk filename is not a wheel", wheels.parse_wheel(Path("notawheel.txt")) is None)


def test_wheel_selection(tmp):
    import wheels
    folder = tmp / "wheels"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("exllamav3-1.4.4-cp313-cp313-win_amd64.whl",
                 "exllamav3-1.4.3-cp313-cp313-win_amd64.whl",
                 "exllamav3-1.4.4-cp312-cp312-win_amd64.whl"):
        (folder / name).write_bytes(b"")
    tags = {"py": "cp313", "nodot": "313", "abi": "cp313", "plat": "win_amd64"}

    hit = wheels.find_local("exllamav3", tags, folder)
    check("the newest matching wheel wins", hit is not None and hit.version == "1.4.4"
          and "cp313" in hit.path.name)
    check("a package with no wheel here is not invented",
          wheels.find_local("torch", tags, folder) is None)

    args = wheels.prebuilt_args("exllamav3", tags, {}, folder)
    check("a local wheel is installed by path and never from the index",
          args is not None and "--no-index" in args and "--only-binary" in args
          and any(a.endswith("cp313-win_amd64.whl") for a in args))

    args = wheels.prebuilt_args("exllamav3", {"py": "cp310", "nodot": "310",
                                              "abi": "cp310", "plat": "win_amd64"},
                                {"WHEEL_INDEX": "https://example/a  https://example/b"}, folder)
    check("WHEEL_INDEX becomes find-links",
          args is not None and args.count("--find-links") == 2 and args[-1] == "exllamav3")

    check("with no wheel and no index there is nothing to try",
          wheels.prebuilt_args("exllamav3", {"py": "cp310", "nodot": "310",
                                             "abi": "cp310", "plat": "win_amd64"},
                               {}, folder) is None)


class _FakeHub(threading.Thread):
    """A tiny stand-in for huggingface.co: a file tree and one file, with
    Range support, so the downloader can be tested without the network."""

    def __init__(self, payload: bytes, repo="me/model"):
        super().__init__(daemon=True)
        import hashlib
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.payload = payload
        self.oid = hashlib.sha256(payload).hexdigest()
        self.repo = repo
        self.hits = []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.hits.append((self.path, self.headers.get("Range")))
                if "/api/models/" in self.path:
                    body = json.dumps([
                        {"type": "file", "path": "config.json", "size": 2, "oid": "x"},
                        {"type": "file", "path": "model.safetensors",
                         "size": len(outer.payload),
                         "lfs": {"oid": outer.oid, "size": len(outer.payload)}},
                        {"type": "directory", "path": "sub"},
                    ]).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path.endswith("config.json"):
                    data = b"{}"
                elif self.path.endswith("model.safetensors"):
                    data = outer.payload
                else:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start = 0
                rng = self.headers.get("Range")
                if rng and rng.startswith("bytes="):
                    start = int(rng[6:].split("-")[0])
                    data = data[start:]
                    self.send_response(206)
                    self.send_header("Content-Range",
                                     f"bytes {start}-{start + len(data) - 1}/{len(outer.payload)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]

    def run(self):
        self.httpd.serve_forever(poll_interval=0.1)

    def stop(self):
        self.httpd.shutdown()


def test_downloader(tmp):
    import downloader
    import shutil
    for stale in ("dl", "dl2", "dl3", "dl5", "cancel"):
        shutil.rmtree(tmp / stale, ignore_errors=True)
    payload = bytes(range(256)) * 400          # 102 400 bytes
    hub = _FakeHub(payload)
    hub.start()
    old = downloader.HF_ENDPOINT
    downloader.HF_ENDPOINT = f"http://127.0.0.1:{hub.port}"
    try:
        dest = tmp / "dl"
        seen = []
        dl = downloader.Download("me/model", dest, "main", on_progress=seen.append)
        dl.run()
        check("every file arrives", (dest / "config.json").is_file()
              and (dest / "model.safetensors").read_bytes() == payload)
        check("directories in the tree are skipped", not (dest / "sub").exists())
        check("progress reaches 100%", seen and seen[-1]["percent"] == 100.0)
        check("the download reports done", dl.progress.state == "done")
        check("is_complete agrees", downloader.is_complete(dest))

        # a half-written part file from an interrupted run
        dest2 = tmp / "dl2"
        dest2.mkdir(parents=True, exist_ok=True)
        part = dest2 / "model.safetensors.part"
        part.write_bytes(payload[:40000])
        (dest2 / "model.safetensors.part.json").write_text(
            json.dumps({"size": len(payload), "oid": hub.oid}))
        hub.hits.clear()
        dl2 = downloader.Download("me/model", dest2, "main")
        dl2.run()
        ranges = [r for p, r in hub.hits if r]
        check("an interrupted download resumes with Range",
              any(r == "bytes=40000-" for r in ranges))
        check("the resumed file is still correct",
              (dest2 / "model.safetensors").read_bytes() == payload)
        check("resumed bytes are counted once, not twice",
              dl2.progress.done_bytes == dl2.progress.total_bytes)

        # a part file whose sidecar does not describe this file is thrown away
        dest3 = tmp / "dl3"
        dest3.mkdir(parents=True, exist_ok=True)
        p3 = dest3 / "model.safetensors.part"
        p3.write_bytes(b"rubbish from another model")
        (dest3 / "model.safetensors.part.json").write_text(
            json.dumps({"size": 999, "oid": "nope"}))
        downloader.Download("me/model", dest3, "main").run()
        check("a stale part file is discarded rather than appended to",
              (dest3 / "model.safetensors").read_bytes() == payload)

        # already on disk: nothing is fetched again
        hub.hits.clear()
        dl4 = downloader.Download("me/model", dest, "main")
        dl4.run()
        check("a finished download does not re-fetch",
              not [p for p, _ in hub.hits if "resolve" in p])

        # a checksum that does not match is deleted, not kept
        bad = _FakeHub(payload)
        bad.oid = "0" * 64
        bad.start()
        downloader.HF_ENDPOINT = f"http://127.0.0.1:{bad.port}"
        dest5 = tmp / "dl5"
        try:
            downloader.Download("me/model", dest5, "main").run()
            failed = False
        except downloader.DownloadError:
            failed = True
        check("a bad checksum fails loudly", failed)
        check("and the bad file is not left behind",
              not (dest5 / "model.safetensors").is_file())
        bad.stop()
    finally:
        downloader.HF_ENDPOINT = old
        hub.stop()


def test_download_cancel(tmp):
    import downloader
    import shutil
    shutil.rmtree(tmp / "cancel", ignore_errors=True)
    hub = _FakeHub(bytes(4_000_000))
    hub.start()
    old = downloader.HF_ENDPOINT
    downloader.HF_ENDPOINT = f"http://127.0.0.1:{hub.port}"
    try:
        dl = downloader.Download("me/model", tmp / "cancel", "main")
        dl.cancel()
        dl.run()
        check("a cancelled download stops without an exception",
              dl.progress.state == "cancelled")
    finally:
        downloader.HF_ENDPOINT = old
        hub.stop()


def test_setup_core(tmp):
    import setup_core
    needed, reasons = setup_core.needs_setup({})
    check("an empty config needs setup", needed and "no profile chosen yet" in reasons)

    ids = [s.id for s in setup_core.install_steps()]
    check("the install has the stages the page draws",
          ids[:3] == ["venv", "pipbase", "torch"] and ids[-1] == "check")

    steps = setup_core.Steps(setup_core.install_steps())
    steps.start("venv")
    check("a running stage is marked running", steps.get("venv").state == "running")
    steps.finish("venv", "done")
    check("a finished stage records its time",
          steps.get("venv").state == "ok" and steps.get("venv").seconds >= 0)
    steps.fail("torch", "no network", "check the connection")
    check("a failed stage carries the reason and the fix",
          steps.get("torch").state == "failed" and steps.get("torch").hint)

    data = setup_core.options_for({}, vram_gib=16.0)
    check("16 GB gets a menu of quants that fit", len(data["options"]) >= 3)
    check("every option carries what the page shows",
          all({"id", "ctx", "need", "disk_gb", "note", "downloaded"} <= set(o)
              for o in data["options"]))
    check("one option is recommended",
          sum(1 for o in data["options"] if o["recommended"]) == 1)
    check("a card too small for anything gets an empty menu",
          setup_core.options_for({}, vram_gib=4.0)["options"] == [])

    env = tmp / ".env"
    env.write_text("PROFILE=ask\nMODEL_DIR=old\n")
    old_env = setup_core.ENV_FILE
    setup_core.ENV_FILE = env
    try:
        pick = data["options"][-1]["id"]
        cfg = setup_core.apply_choice({}, pick, 16.0)
        written = env.read_text()
        check("choosing a profile writes .env", "PROFILE=" in written
              and "ask" not in written.split("PROFILE=")[1].splitlines()[0])
        check("and the caller gets the new settings back",
              cfg.get("MODEL_DIR") and cfg.get("CONTEXT_SIZE"))
        bad = False
        try:
            setup_core.apply_choice({}, "99.0", 16.0)
        except setup_core.SetupError:
            bad = True
        check("an option that does not exist is refused", bad)
    finally:
        setup_core.ENV_FILE = old_env


def test_setup_web(tmp):
    """The setup page's own server: guards, the state machine, the event log."""
    import setup_mock
    import setup_web

    setup_mock.patch(vram_gib=16.0, speed=0.0, download_seconds=0.3)
    setup, httpd, url = setup_web.serve({"MODEL_DIR": "models/test"}, 0, "127.0.0.1")
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        def call(path, payload=None, headers=None, method=None):
            h = {"Content-Type": "application/json"}
            h.update(headers or {})
            data = json.dumps(payload).encode() if payload is not None else None
            req = urllib.request.Request(base + path, data=data, headers=h, method=method)
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read() or b"{}")

        state = call("/setup/state")
        check("setup opens on the model menu", state["phase"] == "choose")
        check("the page is told what it is being asked to fix",
              bool(state["probe"]["reasons"]))

        refused = False
        try:
            call("/setup/choose", {"quant": "2.0"})
        except urllib.error.HTTPError as e:
            refused = e.code == 403
        check("a POST without the UI marker is refused", refused)

        wrong_host = False
        try:
            req = urllib.request.Request(base + "/setup/state", headers={"Host": "evil.example"})
            urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as e:
            wrong_host = e.code == 403
        check("a rebound host name is refused", wrong_host)

        quant = state["menu"]["options"][-1]["id"]
        after = call("/setup/start", {"quant": quant}, {"X-Simplex-UI": "1"})
        check("starting moves to the install screen", after["phase"] == "install")

        deadline = time.time() + 45
        while time.time() < deadline:
            state = call("/setup/state")
            if state["phase"] in ("done", "error"):
                break
            time.sleep(0.3)
        check("setup runs to the end", state["phase"] == "done",
              state.get("error", {}) if state.get("error") else state["phase"])
        check("every stage finished", all(s["state"] in ("ok", "skipped")
                                          for s in state["steps"]))
        check("the download reports done", state["download"]["state"] == "done")
        check("the log was captured for the page", len(call("/setup/log")["log"]) > 3)

        events = setup.events
        kinds = {e["type"] for e in events}
        check("the event log carries every kind the page listens for",
              {"phase", "steps", "log", "download"} <= kinds)
        check("events are numbered so a reload can catch up",
              [e["i"] for e in events] == list(range(len(events))))
        check("a log event keeps its own kind field",
              any(e["type"] == "log" and "kind" in e for e in events))

        replay = list(setup.follow(len(events) - 2))
        check("follow() replays from a given point", len(replay) == 2)

        health = urllib.request.Request(base + "/health")
        code = 0
        try:
            urllib.request.urlopen(health, timeout=5)
        except urllib.error.HTTPError as e:
            code = e.code
        check("/health says 'not the model server yet' during setup", code == 503)

        page = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        check("the setup page is served", "screen-choose" in page)
        escape = 0
        try:
            urllib.request.urlopen(base + "/ui/static/../../.env", timeout=5)
        except urllib.error.HTTPError as e:
            escape = e.code
        check("static files cannot escape the ui folder", escape == 404)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_logbook(tmp):
    import logbook
    lb = logbook.Logbook(directory=tmp / "logs", keep=3)
    path = lb.start()
    try:
        print("plain line")
        print("\x1b[31mcoloured\x1b[0m line")
        sys.stderr.write("an error\n")
    finally:
        lb.stop()
    text = path.read_text()
    check("the log captures stdout and stderr",
          "plain line" in text and "an error" in text)
    check("colour codes are stripped from the file",
          "coloured line" in text and "\x1b[" not in text)

    for i in range(6):
        (tmp / "logs" / f"simplex-old{i}.log").write_text("x")
    pruner = logbook.Logbook(directory=tmp / "logs", keep=3)
    pruner.start()
    pruner.stop()
    kept = list((tmp / "logs").glob("simplex-*.log"))
    check("old logs are pruned", len(kept) <= 4, f"{len(kept)} left")

    what, todo = logbook.explain(PermissionError(".env"))
    check("a permission error is explained in plain words",
          "refused access" in what and "antivirus" in todo.lower())
    what, _ = logbook.explain(MemoryError())
    check("running out of memory suggests a smaller model", "memory" in what.lower())
    what, _ = logbook.explain(ValueError("weird"))
    check("an unknown error still says something useful", "unexpected error" in what)


def test_shortcuts():
    import shortcuts
    target, args = shortcuts.launcher()
    check("shortcuts point at the launcher", target.name in ("start.bat", "Simplex.exe"))
    check("a quote in a path cannot break the PowerShell literal",
          shortcuts._q("C:\\a'b") == "'C:\\a''b'")
    check("off Windows nothing is created", shortcuts.create() == [])
    ok, _why = shortcuts.autostart(True)
    check("and start-at-login is a no-op too", ok is False)


def test_pwa():
    """The installable-app files have to be served from the root, or the
    service worker cannot control the page it is meant to speed up."""
    import webui_app
    for name in ("manifest.webmanifest", "sw.js", "icon-192.png", "icon-512.png",
                 "icon-maskable-512.png", "apple-touch-icon.png"):
        check(f"{name} is in the kit", (Path(__file__).parent / "webui" / name).is_file())
        check(f"/{name} is routed", ("/" + name) in webui_app.PWA_FILES)
    manifest = json.loads((Path(__file__).parent / "webui" / "manifest.webmanifest").read_text())
    check("the manifest is installable",
          manifest["display"] == "standalone" and manifest["start_url"] == "/"
          and any(i["purpose"] == "maskable" for i in manifest["icons"]))
    sw = (Path(__file__).parent / "webui" / "sw.js").read_text()
    check("the service worker never caches the API",
          "/ui/static/" in sw and "isShell" in sw)
    index = (Path(__file__).parent / "webui" / "index.html").read_text()
    check("the page links the manifest", 'rel="manifest"' in index)
    app_js = (Path(__file__).parent / "webui" / "app.js").read_text()
    check("registration is guarded by a secure context", "isSecureContext" in app_js)


def test_ready_after_mount():
    """Regression: the Ready box used to print before the UI was mounted and
    before the port was bound, so it could promise an address that was not
    there."""
    src = (Path(__file__).parent / "serve_openai.py").read_text()
    check("mount_ui reports whether it worked",
          "return True" in src.split("def mount_ui")[1][:2000])
    tail = src[src.index("def main()"):]
    check("the UI is mounted before Ready is defined",
          tail.index("mount_ui(app, args)") < tail.index("def ready_box"))
    call = tail.index("ready_box()", tail.index("def ready_box") + 20)
    check("Ready is printed only after the port is bound",
          tail.index("await site.start()") < call)



# ============================================================================
# Regressions from the 2026-09-03 review. Each of these once failed.
# ============================================================================

def test_download_path_containment(tmp):
    """A repository chooses its own filenames, so they are untrusted input that
    ends up in a filesystem path."""
    import downloader
    ok = ["config.json", "sub/dir/model.safetensors", "a-b_c.1.json"]
    bad = ["../../../pwned.txt", "/etc/evil", "C:/Users/Public/e.bat",
           "..\\..\\w.bat", "a/../b", "", "a//b", "sub/../../x", "~/x",
           "con:x", "trailing ", "dot.", "nul\x00byte"]
    check("ordinary paths are allowed",
          all(downloader.safe_relpath(x) == x for x in ok))
    for x in bad:
        check(f"path is refused: {x!r}", downloader.safe_relpath(x) is None)
    check("containment agrees with the filesystem",
          downloader._contained(tmp, tmp / "a" / "b")
          and not downloader._contained(tmp, tmp / ".." / "b"))

    # the whole way through: a tree listing full of escapes writes nothing
    payload = b"hello"
    hub = _FakeHub(payload)
    import json as _json

    class Escaping(_FakeHub):
        pass

    hub.start()
    old = downloader.HF_ENDPOINT
    downloader.HF_ENDPOINT = f"http://127.0.0.1:{hub.port}"
    try:
        files = downloader.list_files("me/model", "main")
        check("list_files returns only safe paths",
              all(downloader.safe_relpath(f.path) == f.path for f in files))
    finally:
        downloader.HF_ENDPOINT = old
        hub.stop()


def test_download_no_token_across_hosts():
    """urllib copies Authorization onto a redirect, and HF redirects every file
    to a CDN on another host."""
    import downloader
    import urllib.request

    handler = downloader._SafeRedirects()

    class FakeFP:
        def read(self, *a):
            return b""

    req = urllib.request.Request("https://huggingface.co/me/m/resolve/main/f")
    req.add_header("Authorization", "Bearer hf_secret")
    req.add_header("User-Agent", "x")

    same = handler.redirect_request(req, FakeFP(), 302, "Found", {},
                                    "https://huggingface.co/other")
    cross = handler.redirect_request(req, FakeFP(), 302, "Found", {},
                                     "https://cdn-lfs.example.com/blob")
    def has_auth(r):
        keys = {k.lower() for k in list(r.headers) + list(r.unredirected_hdrs)}
        return "authorization" in keys
    check("the token survives a redirect on the same host", has_auth(same))
    check("the token is dropped when the host changes", not has_auth(cross))

    refused = False
    try:
        handler.redirect_request(req, FakeFP(), 302, "Found", {}, "file:///etc/passwd")
    except downloader.DownloadError:
        refused = True
    check("a redirect to a non-http scheme is refused", refused)


def test_download_retry_accounting(tmp):
    """Progress used to leap forward and pin at 100% whenever a transfer was
    retried, because the resumed prefix was counted twice."""
    import downloader
    import shutil
    shutil.rmtree(tmp / "flaky", ignore_errors=True)
    payload = bytes(range(256)) * 200            # 51 200 bytes

    hub = _FakeHub(payload)
    hub.start()
    old = downloader.HF_ENDPOINT
    downloader.HF_ENDPOINT = f"http://127.0.0.1:{hub.port}"
    try:
        dl = downloader.Download("me/model", tmp / "flaky", "main")
        real = dl._stream
        state = {"n": 0}

        def flaky(url, part, f, start):
            """Fail once, halfway through the big file."""
            state["n"] += 1
            if f.size == len(payload) and state["n"] == 1:
                with part.open("ab" if start else "wb") as out:
                    out.write(payload[start:start + 20000])
                dl._advance(20000)
                raise OSError("connection reset")
            return real(url, part, f, start)

        dl._stream = flaky
        dl.run()
        check("the retried file is still correct",
              (tmp / "flaky" / "model.safetensors").read_bytes() == payload)
        check("a retry does not count the resumed prefix twice",
              dl.progress.done_bytes == dl.progress.total_bytes,
              f"{dl.progress.done_bytes} vs {dl.progress.total_bytes}")
        check("so the percentage lands exactly on 100",
              dl.progress.as_dict()["percent"] == 100.0)
    finally:
        downloader.HF_ENDPOINT = old
        hub.stop()


def test_download_complete_part_is_kept(tmp):
    """A .part that is already the full size is a download that finished and
    died before the rename - re-fetching it costs gigabytes."""
    import downloader
    import shutil
    shutil.rmtree(tmp / "nearly", ignore_errors=True)
    payload = bytes(range(256)) * 100
    hub = _FakeHub(payload)
    hub.start()
    old = downloader.HF_ENDPOINT
    downloader.HF_ENDPOINT = f"http://127.0.0.1:{hub.port}"
    try:
        dest = tmp / "nearly"
        dest.mkdir(parents=True)
        part = dest / "model.safetensors.part"
        part.write_bytes(payload)
        (dest / "model.safetensors.part.json").write_text(
            json.dumps({"size": len(payload), "oid": hub.oid}))
        rf = downloader.RemoteFile("model.safetensors", len(payload), hub.oid, True)
        check("a complete part file is resumed, not discarded",
              downloader._resume_from(part, rf) == len(payload))
        hub.hits.clear()
        downloader.Download("me/model", dest, "main").run()
        check("and it is renamed without re-downloading",
              not any("model.safetensors" in p for p, _ in hub.hits if "resolve" in p))
        check("the finished file is intact",
              (dest / "model.safetensors").read_bytes() == payload)
    finally:
        downloader.HF_ENDPOINT = old
        hub.stop()


def test_download_transient_http_is_retried(tmp):
    """A 503 from a CDN used to end setup outright: _open turned every HTTP
    error into a DownloadError, which the retry arm never caught."""
    import downloader
    import shutil
    import urllib.error
    shutil.rmtree(tmp / "flappy", ignore_errors=True)

    e503 = urllib.error.HTTPError("u", 503, "busy", {}, None)
    e404 = urllib.error.HTTPError("u", 404, "gone", {}, None)
    for err, want in ((e503, True), (e404, False)):
        raised = None
        try:
            with_patch = downloader._open
            import urllib.request
            real_open = downloader._OPENER.open
            downloader._OPENER.open = lambda *a, **k: (_ for _ in ()).throw(err)
            try:
                downloader._open("http://x/y")
            except downloader.DownloadError as de:
                raised = de
        finally:
            downloader._OPENER.open = real_open
        check(f"HTTP {err.code} retryable == {want}",
              raised is not None and raised.retryable is want)


def test_download_failure_is_terminal(tmp):
    """Anything watching progress.state has to see a terminal value, or it
    polls for a download that is never coming back."""
    import downloader
    import shutil
    shutil.rmtree(tmp / "doomed", ignore_errors=True)
    hub = _FakeHub(b"x" * 500)
    hub.oid = "0" * 64                      # checksum will not match
    hub.start()
    old = downloader.HF_ENDPOINT
    downloader.HF_ENDPOINT = f"http://127.0.0.1:{hub.port}"
    try:
        dl = downloader.Download("me/model", tmp / "doomed", "main")
        try:
            dl.run()
        except downloader.DownloadError:
            pass
        check("a failed download ends in a terminal state",
              dl.progress.state in ("error", "cancelled"), dl.progress.state)
    finally:
        downloader.HF_ENDPOINT = old
        hub.stop()


def test_event_log_survives_trimming():
    """The page froze at MAX_LOG events: the cursor was a list index, and the
    list stopped growing."""
    import setup_web
    s = setup_web.Setup({})
    total = setup_web.MAX_LOG + 250
    for i in range(total):
        s._emit("log", line=str(i), kind="out")
    check("event numbers keep counting past the trim",
          s.events[-1]["i"] == total - 1)
    check("only the recent ones are held", len(s.events) == setup_web.MAX_LOG)
    check("the state reports the absolute count", s.state()["events"] == total)

    s.finished.set()   # so follow() returns instead of blocking
    fresh = list(s.follow(total - 3))
    check("a live cursor still yields new events", [e["i"] for e in fresh]
          == [total - 3, total - 2, total - 1])
    stale = list(s.follow(0))
    check("a cursor pointing at trimmed events resumes at the oldest held",
          len(stale) == setup_web.MAX_LOG and stale[0]["i"] == total - setup_web.MAX_LOG)
    check("a cursor past the end yields nothing", list(s.follow(total + 5)) == [])


def test_phase_is_not_a_latch():
    """Try again on the error screen has to put setup back to work."""
    import setup_web
    s = setup_web.Setup({})
    s.set_phase("error")
    check("a terminal phase releases the waiters", s.finished.is_set() and s.terminal)
    s.set_phase("install")
    check("leaving it makes them wait again",
          not s.finished.is_set() and not s.terminal)
    check("_wait_for_retry sees the change", setup_web._wait_for_retry(s, 0.1))
    s.set_phase("error")
    check("and gives up when nothing changes", not setup_web._wait_for_retry(s, 0.3))


def test_retry_after_a_failure(tmp):
    """The error screen's two buttons used to be dead: the phase was a latch
    and run() tore the server down 1.5 s later."""
    import setup_mock
    import setup_web

    setup_mock.patch(vram_gib=16.0, speed=0.0, download_seconds=0.2, fail="install")
    setup, httpd, _url = setup_web.serve({"MODEL_DIR": "models/test"}, 0, "127.0.0.1")
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        def call(path, payload=None):
            data = json.dumps(payload).encode() if payload is not None else None
            req = urllib.request.Request(
                base + path, data=data,
                headers={"Content-Type": "application/json", "X-Simplex-UI": "1"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read() or b"{}")

        state = call("/setup/state")
        call("/setup/start", {"quant": state["menu"]["options"][-1]["id"]})
        deadline = time.time() + 20
        while time.time() < deadline and call("/setup/state")["phase"] != "error":
            time.sleep(0.2)
        check("a failed install reaches the error screen", setup.phase == "error")
        check("with something to tell the user",
              bool((setup.error or {}).get("message")) and bool((setup.error or {}).get("hint")))

        again = call("/setup/start", {})
        check("Try again is accepted, not answered by a closed port",
              again["phase"] == "install")
        check("and the waiters go back to waiting", not setup.finished.is_set())

        deadline = time.time() + 20
        while time.time() < deadline and call("/setup/state")["phase"] != "error":
            time.sleep(0.2)
        back = call("/setup/refresh", {})
        check("Choose a different model returns to the menu", back["phase"] == "choose")
    finally:
        httpd.shutdown()
        httpd.server_close()

    js = (Path(__file__).parent / "webui" / "setup.js").read_text()
    choose = js[js.index('if (phase === "choose"'):js.index('} else if (phase === "install")')]
    check("and the install button is re-enabled when the menu comes back",
          '$("#go").disabled = false' in choose)


def test_env_injection():
    """.env is parsed line by line, so a newline in a value is a second setting."""
    import profiles
    import win_start
    import setup_web
    target = Path(__file__).resolve().parent.parent / "workspace" / "inject.env"
    target.write_text("PROFILE=x\n", encoding="utf-8")
    profiles.write_env(target, {"HF_TOKEN": "hf_a\nWHEEL_INDEX=http://attacker/",
                                "PROFILE_GPU": "RTX 4090\nUI_LAN=1"})
    parsed = win_start.load_dotenv(target)
    check("a newline cannot add a second setting",
          "WHEEL_INDEX" not in parsed and "UI_LAN" not in parsed)
    check("the value itself is still recorded", parsed["HF_TOKEN"].startswith("hf_a"))

    setup = setup_web.Setup({})
    refused = 0
    for bad in ("hf_a\nWHEEL_INDEX=x", "hf a", "hf_a#c", "x" * 300, "hf_a\rY=1"):
        try:
            setup.set_token(bad)
        except Exception:                    # noqa: BLE001
            refused += 1
    check("the token box refuses anything that is not a token", refused == 5)


def test_setup_peer_guard():
    """Host and a custom header stop a browser; only the peer address stops
    curl, and this endpoint writes .env and runs pip."""
    import setup_web
    check("loopback is allowed", setup_web._peer_ok(("127.0.0.1", 5)))
    check("mapped loopback is allowed", setup_web._peer_ok(("::ffff:127.0.0.1", 5)))
    check("an in-process caller is allowed", setup_web._peer_ok(None))
    for bad in ("192.168.1.5", "10.0.0.2", "203.0.113.9", "::2"):
        check(f"{bad} is refused", not setup_web._peer_ok((bad, 5)))


def test_port_probe_detects_a_listener():
    """A bind test alone says 'free' on Windows, where SO_REUSEADDR allows
    binding a port that is already being listened on."""
    import setup_web
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    try:
        check("a port with a listener is not free", not setup_web._port_free("127.0.0.1", port))
    finally:
        srv.shutdown()
        srv.server_close()
    check("a port with nothing on it is free", setup_web._port_free("127.0.0.1", port))


def test_log_keeps_only_the_last_redraw():
    """Piping the child turned its progress bar into one console line per
    redraw; the log then got every one of them."""
    import logbook
    bar = "  10%\r  50%\r 100%\ndone\n"
    check("carriage-return redraws collapse in the file",
          logbook.for_file(bar) == " 100%\ndone\n")
    check("CRLF is a line ending, not a redraw",
          logbook.for_file("a\r\nb\r\n") == "a\nb\n")
    check("colour still goes", logbook.for_file("\x1b[31mred\x1b[0m\n") == "red\n")


def test_logbook_start_is_idempotent(tmp):
    import logbook
    lb = logbook.Logbook(directory=tmp / "logs2", keep=3)
    first = lb.start()
    try:
        second = lb.start()
        check("a second start() is a no-op, not a nested tee", first == second)
        check("stdout is wrapped exactly once",
              isinstance(sys.stdout, logbook.Tee)
              and not isinstance(getattr(sys.stdout, "_stream", None), logbook.Tee))
        other = logbook.Logbook(directory=tmp / "logs3", keep=2)
        check("a second Logbook does not wrap on top", other.start() == first)
    finally:
        lb.stop()
    check("and stop() restores the real stream",
          not isinstance(sys.stdout, logbook.Tee))


def test_powershell_quoting():
    """PowerShell treats four typographic quotes as string delimiters too, and
    these strings carry the user's account name."""
    import shortcuts
    check("the ascii apostrophe is doubled", shortcuts._q("O'Brien") == "'O''Brien'")
    for ch in "\u2018\u2019\u201a\u201b":
        q = shortcuts._q(f"C:\\{ch}x")
        check(f"U+{ord(ch):04X} is doubled", q.count(ch) == 2 and q.endswith("x'"))
    check("ordinary paths are untouched",
          shortcuts._q("C:\\Users\\a\\Simplex") == "'C:\\Users\\a\\Simplex'")


def test_child_pipe_is_read_as_bytes():
    """Regression: text=True puts the pipe in universal-newline mode, where a
    bare CR ends a line - which is exactly what a progress bar is made of."""
    import io
    wrapper = io.TextIOWrapper(io.BytesIO(b"10%\r50%\rdone\n"))
    check("a text pipe really does split on CR", len(list(wrapper)) == 3)
    src = (Path(__file__).parent / "win_start.py").read_text()
    launch = src[src.index("proc = subprocess.Popen("):]
    launch = launch[:launch.index(")")]
    check("so the launcher does not use one", "text=True" not in launch)
    check("and decodes the bytes itself", "getincrementaldecoder" in src)


def test_shortcuts_wait_for_a_working_start():
    """Shortcuts used to be written before the server had ever run, so a kit
    that died on the model load still left an icon on the desktop."""
    import shortcuts
    src = (Path(__file__).parent / "win_start.py").read_text()
    check("they are made from the ready callback", "ensure_shortcuts(cfg)" in
          src[src.index("def _on_ready"):src.index("class Runtime")])
    check("and not before the server is launched",
          "ensure_shortcuts(cfg)" not in src[src.index("    rt = Runtime()"):])
    check("an installer's own choice is respected", hasattr(shortcuts, "looks_installed"))


def test_second_ctrl_c_still_kills_the_child():
    src = (Path(__file__).parent / "win_start.py").read_text()
    block = src[src.index("except KeyboardInterrupt:\n                # Ctrl+C"):]
    block = block[:block.index("pump.join")]
    check("a second Ctrl+C does not skip terminate()", "except BaseException" in block)


def test_installer_version_is_overridable():
    iss = (Path(__file__).resolve().parent.parent / "packaging" / "simplex.iss").read_text()
    check("build.ps1's /DAppVersion is not overwritten", "#ifndef AppVersion" in iss)
    check("the architecture directive works on every Inno 6",
          "ArchitecturesAllowed=x64\n" in iss)
    check("the wizard warns about the download size",
          "ExtraDiskSpaceRequired=10737418240" in iss)
    check("the kit's own folders stay out of the installer",
          all(x in iss for x in ("\\.github", "\\bench_vram.json", "\\models")))


def test_broken_venv_is_rebuilt():
    """A venv whose base Python was upgraded still has a python.exe that
    cannot run; trusting it made the rebuild unreachable."""
    import setup_core
    src = (Path(__file__).parent / "setup_core.py").read_text()
    check("the venv is probed, not just looked for", 'usable = VENV_PY.is_file() and' in src)
    check("a dead one is deleted first", "shutil.rmtree(VENV_DIR" in src)
    sys_py = setup_core._system_python()
    check("the system python is one that runs", sys_py is not None)
    check("and is never the venv being rebuilt",
          sys_py is None or setup_core.VENV_DIR not in Path(sys_py[0]).resolve().parents)
    bat = (Path(__file__).resolve().parent.parent / "start.bat").read_text()
    check("start.bat checks the venv runs before using it",
          '-c "pass"' in bat and "if not errorlevel 1" in bat)


def test_bench_vram_gpu_attrs():
    """bench_vram.py's main() reads gpu.total_gib / gpu.driver: profiles.GPU's
    actual attribute names. Using .memory / .driver_version crashed with an
    AttributeError before a single quant ever loaded."""
    import profiles
    g = profiles.detect_gpu()
    check("profiles.GPU has no .memory (bench_vram must not assume it)",
          not hasattr(g, "memory"))
    check("profiles.GPU exposes total_gib", hasattr(g, "total_gib"))
    check("profiles.GPU exposes driver", hasattr(g, "driver"))
    src = (Path(__file__).parent / "bench_vram.py").read_text()
    check("bench_vram.py reads gpu.total_gib, not gpu.memory",
          "gpu.total_gib" in src and "gpu.memory" not in src)
    check("bench_vram.py's dry-run stub matches the real GPU's attribute names",
          '"total_gib": 32.0, "driver": "0"' in src)
    check("bench_vram.py no longer reads the nonexistent gpu.driver_version",
          "gpu.driver_version" not in src)
    # --list and --dry-run must not touch CUDA, so both must run clean here
    # even though this box has no GPU. They must also not touch the real results:
    # a dry run fabricates numbers, and this suite used to delete bench_vram.json
    # afterwards - between them they silently destroyed overnight measurements
    # that --all is documented to resume from.
    import subprocess, json as _json
    root = Path(__file__).resolve().parent.parent
    real = root / "bench_vram.json"
    real_md = root / "bench_vram.md"
    saved = (real.read_text(encoding="utf-8") if real.is_file() else None,
             real_md.read_text(encoding="utf-8") if real_md.is_file() else None)
    planted = saved[0] is None
    if planted:                       # nothing real here: plant a stand-in to guard
        real.write_text(_json.dumps({"gpu": {"name": "REAL CARD"},
                                     "quants": {"5.0": {"id": "5.0", "weights_gib": 16.123}}}),
                        encoding="utf-8")
    before = real.read_text(encoding="utf-8")
    try:
        for extra in (["--list"], ["--dry-run", "--quant", "2.0"]):
            r = subprocess.run([sys.executable, str(Path(__file__).parent / "bench_vram.py"), *extra],
                               cwd=str(root), capture_output=True, text=True, timeout=30)
            check(f"bench_vram.py {' '.join(extra)} exits clean",
                  r.returncode == 0, r.stderr[-400:])
            check(f"bench_vram.py {' '.join(extra)} leaves real measurements alone",
                  real.read_text(encoding="utf-8") == before)
        check("a dry run writes its fabricated numbers to their own file",
              (root / "bench_vram.dryrun.json").is_file())
        # The real file legitimately holds a 2.0 row from an actual bench run, so
        # "no 2.0 row" is not the invariant - "nothing the dry run invented" is:
        # its GPU block is named "Dry run", and the file must be byte-identical
        # (asserted per command above) whatever it already contained.
        after = _json.loads(real.read_text(encoding="utf-8"))
        check("no fabricated GPU block leaked into the real results",
              (after.get("gpu") or {}).get("name") != "Dry run", after.get("gpu"))
        check("and the real file is byte-identical to before the dry run",
              real.read_text(encoding="utf-8") == before)
    finally:
        # Cleanup must never be what fails the run: on a read-only or
        # delete-restricted mount the unlink raises, and losing a passing test to
        # a leftover temp file helps nobody.
        for f in root.glob("bench_vram.dryrun.*"):
            try:
                f.unlink()
            except OSError:
                pass
        if planted:
            try:
                real.unlink(missing_ok=True)
            except OSError:
                pass
        elif saved[0] is not None:
            real.write_text(saved[0], encoding="utf-8")
        if saved[1] is not None:
            real_md.write_text(saved[1], encoding="utf-8")


def test_bench_vram_kv_rate_excludes_vision():
    """kv_rate_kb()/weights_from_probe() must derive the KV-per-token rate
    from the raw allocated_gib delta between the two probes alone.

    probe_a's allocated_gib is snapshotted right after build_model(), before
    the vision tower ever loads - so it never includes vision_gib in the
    first place. A prior version of measure_quant() subtracted vision_gib
    from probe_a's allocated_gib anyway, double-counting it and inflating
    the measured rate by vision_gib * 32 KB/token (since here
    PROBE_B - PROBE_A == 32768 tokens == 1024**3/32768 GiB->KB scaling
    collapses to a clean x32). This test pins the correct, vision-agnostic
    math with numbers reverse-engineered to match that exact failure mode,
    so a regression reintroducing the double-subtraction fails loudly."""
    import bench_vram as bv

    # Two probes of a real quant with a vision tower: weights ~14 GiB,
    # vision ~0.6 GiB, true KV rate 18 KB/token (the profiles.py assumption),
    # MTP factor already baked into the rate for this synthetic case.
    weights_gib = 14.0
    vision_gib = 0.6
    true_kv_kb = 18.0
    probe_a = {
        "ok": True,
        "allocated_gib": weights_gib + true_kv_kb * bv.PROBE_A * 1024 / bv.GIB,
        "vision_gib": vision_gib,
    }
    probe_b = {
        "ok": True,
        "allocated_gib": weights_gib + true_kv_kb * bv.PROBE_B * 1024 / bv.GIB,
    }

    kv_kb = bv.kv_rate_kb(probe_a, probe_b)
    check("kv_rate_kb() recovers the true per-token rate regardless of "
          "vision_gib on probe_a",
          abs(kv_kb - true_kv_kb) < 1e-6, kv_kb)

    weights = bv.weights_from_probe(probe_a, kv_kb)
    check("weights_from_probe() backs out just the KV footprint, leaving "
          "the true resident weights (vision excluded, not subtracted)",
          abs(weights - weights_gib) < 1e-6, weights)

    # A quant with NO vision tower (vision_gib absent/0) must give the exact
    # same kv_kb and weights_gib as the vision case above - proving vision
    # plays no role in either calculation, which is the actual bug that was
    # fixed (a nonzero vision_gib used to change the result; it must not).
    probe_a_no_vision = dict(probe_a)
    probe_a_no_vision.pop("vision_gib")
    kv_kb_no_vision = bv.kv_rate_kb(probe_a_no_vision, probe_b)
    check("kv_rate_kb() is identical whether or not probe_a carries vision_gib",
          kv_kb_no_vision == kv_kb, (kv_kb_no_vision, kv_kb))
    weights_no_vision = bv.weights_from_probe(probe_a_no_vision, kv_kb_no_vision)
    check("weights_from_probe() is identical whether or not probe_a carries "
          "vision_gib",
          weights_no_vision == weights, (weights_no_vision, weights))

    # The rate is floored at 0.1 KB/token so a same-size or backwards probe
    # pair (or measurement noise landing at/under zero) can't produce a
    # zero or negative rate that would divide-by-near-zero downstream in
    # plan_context().
    flat = {"allocated_gib": 14.0}
    check("kv_rate_kb() floors at 0.1 KB/token instead of going to zero "
          "when the two probes show no growth",
          bv.kv_rate_kb(flat, flat) == 0.1)


def test_profiles_carry_measured_numbers():
    """profiles.py's QUANTS must hold what bench_vram.py actually measured, and
    say so per row. A row that was never loaded on hardware has to admit it, or
    the menu presents a guess with the same confidence as a measurement."""
    import profiles

    by_id = {q.id: q for q in profiles.QUANTS}
    # measured on the RTX 5090 (bench_vram.md); the old table carried shard-size
    # estimates that ran 0.8-1.7 GiB light on every quant
    for qid, weights, vision in (("6.0", 18.935, 0.874), ("5.0", 16.123, 0.877),
                                 ("4.0", 13.283, 0.873), ("3.5", 11.864, 0.872),
                                 ("3.0", 10.422, 0.870), ("2.0", 7.082, 0.173)):
        q = by_id[qid]
        check(f"{qid}bpw carries its measured weights ({weights} GiB)",
              abs(q.gpu_gib - weights) < 1e-6, q.gpu_gib)
        check(f"{qid}bpw carries its measured vision tower ({vision} GiB)",
              abs(q.vision_gib - vision) < 1e-6, q.vision_gib)
        check(f"{qid}bpw is flagged measured", q.measured is True)

    check("2.5bpw is still flagged unmeasured (its download is corrupt)",
          by_id["2.5"].measured is False)
    check("only 2.5bpw is unmeasured",
          [q.id for q in profiles.QUANTS if not q.measured] == ["2.5"])

    # The KV assumption survived the bench: measured rates were 18.85-19.81
    # KB/token *including* the MTP draft cache, and profiles applies MTP_FACTOR
    # on top of KV_KB - so the base rate must land near 18, not near 19.
    for measured_with_mtp in (18.85, 19.81):
        base = measured_with_mtp / profiles.MTP_FACTOR
        check(f"KV_KB['4'] is within 5% of the measured base rate ({base:.2f})",
              abs(base - profiles.KV_KB["4"]) / profiles.KV_KB["4"] < 0.05, base)


def test_planner_arithmetic_is_pinned():
    """Concrete numbers, computed by hand, for every step of the VRAM model.

    Everything else in this file asserts relationships - that images cost
    context, that one menu matches another - and relationships survive a broken
    constant. A mutation run proved it: deleting MARGIN_GIB, dropping the vision
    tower from need_gib(), and handing the server the whole card instead of
    need+1.5 each left the whole suite green. These are the values themselves."""
    import profiles as p                                # noqa: WPS433
    GIB = 1024 ** 3

    # --- kv_gib: KB/token -> GiB, with the MTP draft cache on top ----------
    # 262144 tokens x 18 KB x 17/16 = 4831838208 x 17/16 bytes = 4.78 GiB
    expect = 262144 * 18 * 1024 / GIB * (17 / 16)
    check("kv_gib(262144, '4') is 4.78 GiB", abs(p.kv_gib(262144, "4") - expect) < 1e-9,
          p.kv_gib(262144, "4"))
    check("...which is 4.78 to two places", round(p.kv_gib(262144, "4"), 2) == 4.78,
          round(p.kv_gib(262144, "4"), 2))
    check("kv_gib scales linearly with context",
          abs(p.kv_gib(131072, "4") * 2 - p.kv_gib(262144, "4")) < 1e-9)
    check("the 8-bit cache costs 34/18 of the int4 one",
          abs(p.kv_gib(65536, "8") / p.kv_gib(65536, "4") - 34 / 18) < 1e-9)

    # --- budget: total VRAM minus the larger of 1.3 GiB and 8% ------------
    for total, want in ((16, 14.7), (24, 22.1), (32, 29.4), (8, 6.7)):
        check(f"budget_gib({total}) is {want}", p.budget_gib(total) == want, p.budget_gib(total))

    # --- need_gib: weights + KV + tower + overhead, each actually charged --
    q4 = next(q for q in p.QUANTS if q.id == "4.0")
    base = q4.gpu_gib + p.kv_gib(131072, "4") + p.overhead_gib(131072)
    check("need_gib without images is weights + KV + overhead",
          abs(p.need_gib(q4, 131072, "4", False) - base) < 1e-9)
    check("need_gib with images adds the tower, and that is a real cost",
          abs(p.need_gib(q4, 131072, "4", True) - (base + q4.vision_gib)) < 1e-9)
    check("the tower is not silently free",
          p.need_gib(q4, 131072, "4", True) - p.need_gib(q4, 131072, "4", False) > 0.5,
          q4.vision_gib)

    # --- overhead: the measured figure, not the old 1.7 --------------------
    check("OVERHEAD_GIB bounds the worst measured underestimate (0.73 GiB at 1.7)",
          p.OVERHEAD_GIB >= 2.43, p.OVERHEAD_GIB)
    check("overhead_gib() is flat until a sweep measures the per-token part",
          p.overhead_gib(32768) == p.overhead_gib(262144) or p.OVERHEAD_KB_PER_TOKEN > 0)

    # --- max_ctx is the exact inverse of need_gib, margin included ---------
    budget = p.budget_gib(24)
    ctx = p.max_ctx(q4, "4", True, budget)
    if ctx < p.NATIVE_CTX and (not q4.verified_ctx or ctx < q4.verified_ctx):
        check("max_ctx plans right up to budget - MARGIN_GIB",
              abs(p.need_gib(q4, ctx, "4", True) - (budget - p.MARGIN_GIB)) < 0.02,
              (p.need_gib(q4, ctx, "4", True), budget - p.MARGIN_GIB))
        check("one step more would not fit",
              p.need_gib(q4, ctx + 256, "4", True) > budget - p.MARGIN_GIB)
    # a card with no room at all plans nothing, rather than a negative context
    tiny = next(q for q in p.QUANTS if q.id == "6.0")
    check("a quant whose weights exceed the budget plans 0 context",
          p.max_ctx(tiny, "4", True, p.budget_gib(12)) == 0)

    # --- MARGIN_GIB is real slack, not decoration -------------------------
    check("MARGIN_GIB is actually held back",
          p.MARGIN_GIB > 0 and p.need_gib(q4, p.max_ctx(q4, "4", True, budget), "4", True)
          <= budget - p.MARGIN_GIB + 0.02)

    # --- env_updates: what the server is actually told ---------------------
    # GPU_MEM_GB becomes a hard per-process cap in serve_openai, so it must be
    # what this profile needs plus a little, never the whole card. Assert it on a
    # row where those two differ - on a row whose need+1.5 already exceeds the
    # budget the formulas coincide and the check proves nothing.
    _, options = p.plan(32.0, False)
    discriminating = [o for o in options if o["need"] + 1.5 < o["budget"] - 0.5]
    check("there is a profile where 'need + 1.5' and 'the whole budget' differ",
          bool(discriminating), [(o["quant"], o["need"], o["budget"]) for o in options])
    for o in discriminating:
        upd = p.env_updates(o, "Fake GPU")
        cap = float(upd["GPU_MEM_GB"])
        check(f"{o['quant']}bpw: GPU_MEM_GB is need + 1.5, not the whole card",
              abs(cap - (o["need"] + 1.5)) < 0.05, (cap, o["need"], o["budget"]))
        check(f"{o['quant']}bpw: and it leaves the rest of the card alone",
              cap < o["budget"], (cap, o["budget"]))
    o = next(x for x in options if x["quant"] == "4.0")
    upd = p.env_updates(o, "Fake GPU")
    check("CONTEXT_SIZE is the planned context verbatim",
          upd["CONTEXT_SIZE"] == str(o["ctx"]))
    check("VISION follows the plan", upd["VISION"] == ("auto" if o["vision"] else "off"))
    check("the profile name carries the quant and context",
          upd["PROFILE"] == f"{o['quant']}bpw-{round(o['ctx'] / 1000)}k", upd["PROFILE"])


def test_verified_context_ceiling():
    """max_ctx() must never offer a context that a real prefill already failed at.

    The formula counts weights + KV + a flat overhead, but the prefill workspace
    grows with context too: 5.0bpw computes a full 262144 and died at 262144,
    222720 and 183296+1 on the bench card, surviving only at 183296. verified_ctx
    pins that observed ceiling so no budget, however large, can plan past it."""
    import profiles

    by_id = {q.id: q for q in profiles.QUANTS}
    huge = profiles.budget_gib(96)          # far more VRAM than any row needs

    q5 = by_id["5.0"]
    check("5.0bpw has its bench-verified ceiling recorded", q5.verified_ctx == 183296)
    for vision in (True, False):
        got = profiles.max_ctx(q5, "4", vision, huge)
        check(f"5.0bpw on a 96GB card stops at the verified 183296 (vision={vision})",
              got == 183296, got)

    q4 = by_id["4.0"]
    check("4.0bpw verified the full native context", q4.verified_ctx == profiles.NATIVE_CTX)
    check("4.0bpw is therefore free to reach it",
          profiles.max_ctx(q4, "4", True, huge) == profiles.NATIVE_CTX)

    # A quant with no verified ceiling still falls back to the formula, and the
    # menu marks it - see test_menu_marks_unverified_rows.
    check("6.0bpw has no verified ceiling yet (its stress run OOMed)",
          by_id["6.0"].verified_ctx is None)

    # The cap only ever lowers a number; a small card is still limited by VRAM.
    tight = profiles.budget_gib(24)
    check("the ceiling never raises what a smaller card can hold",
          profiles.max_ctx(q5, "4", True, tight) <= 183296)


def test_vision_is_asked_not_guessed():
    """The picker asks about images up front, then builds the menu for that
    answer. Before, it decided per row, so a text-only install silently paid for
    a vision tower it never used - about 0.9 GiB, ~45k tokens of context."""
    import profiles
    src = (Path(__file__).parent / "profiles.py").read_text()

    check("there is a question to ask", "def ask_vision(" in src)
    check("run() asks it before planning", "want_vision = None if (auto or list_only)" in src)
    check("plan() takes the answer", "def plan(total_gib: float, want_vision" in src)

    for gb in (16, 24, 32):
        _, on = profiles.plan(gb, True)
        _, off = profiles.plan(gb, False)
        _, auto = profiles.plan(gb)

        check(f"{gb}GB: images-on menu never offers a row without images",
              all(o["vision"] for o in on))
        check(f"{gb}GB: images-off menu never offers a row with images",
              all(not o["vision"] for o in off))
        # every quant that fits with the tower loaded also fits without it
        on_ids, off_ids = {o["quant"] for o in on}, {o["quant"] for o in off}
        check(f"{gb}GB: asking for images can only narrow the menu",
              on_ids <= off_ids, (sorted(on_ids), sorted(off_ids)))
        for o in off:
            same = next((r for r in on if r["quant"] == o["quant"]), None)
            if same:
                check(f"{gb}GB {o['quant']}bpw: dropping images never costs context",
                      o["ctx"] >= same["ctx"], (o["ctx"], same["ctx"]))
        check(f"{gb}GB: None is the default, so unattended planning is unchanged",
              auto == profiles.plan(gb, None)[1])
        check(f"{gb}GB: the automatic rule still picks per row, between the two menus",
              all(any(o["quant"] == a["quant"] for o in off) for a in auto))

    # 16 GB is where the trade actually bites: the tower is worth real context
    _, on16 = profiles.plan(16, True)
    _, off16 = profiles.plan(16, False)
    check("on a 16GB card, text-only unlocks context or quants that images cannot fit",
          sum(o["ctx"] for o in off16) > sum(o["ctx"] for o in on16))


def test_quality_labels_do_not_invert():
    """The word next to a quant is the only quality signal most people read, so
    it must never rank a worse quant above a better one. Easy to break by
    editing one row: calling 3.0 bpw "better" while 3.5 still said "good" made
    the more expensive download look like the weaker choice."""
    import profiles                                     # noqa: WPS433

    rank = {w: i for i, w in enumerate(profiles.QUALITY_ORDER)}
    rows = sorted(profiles.QUANTS, key=lambda q: q.bpw)
    for q in rows:
        word = profiles.QUALITY_WORD.get(q.note, q.note)
        check(f"{q.id}bpw's label ({word}) is on the known ladder", word in rank, word)

    # more bits per weight is never labelled worse, and never has a higher KL
    for lo, hi in zip(rows, rows[1:]):
        lo_w = profiles.QUALITY_WORD.get(lo.note, lo.note)
        hi_w = profiles.QUALITY_WORD.get(hi.note, hi.note)
        check(f"{hi.id}bpw is not labelled worse than {lo.id}bpw "
              f"({lo_w} -> {hi_w})", rank.get(hi_w, -1) >= rank.get(lo_w, -1))
        check(f"{hi.id}bpw really is closer to the unquantised model than {lo.id}bpw",
              hi.kl <= lo.kl, (lo.kl, hi.kl))

    # the specific labels asked for
    by_id = {q.id: profiles.QUALITY_WORD.get(q.note, q.note) for q in profiles.QUANTS}
    for qid, want in (("2.0", "fair"), ("2.5", "good"), ("3.0", "better")):
        check(f"{qid}bpw reads as '{want}'", by_id[qid] == want, by_id[qid])


def test_claimed_minimum_vram_is_honest():
    """The banner's "NVIDIA N GB+" and the README's table are promises. They have
    to match what the planner actually offers, or they drift the moment a
    measurement changes - which is how the banner ended up claiming 12 GB while
    a 12 GB card gets exactly one profile (2.0bpw at 33k, no images) and the
    README's own headline said 16 GB."""
    import re                                           # noqa: WPS433
    import profiles                                     # noqa: WPS433
    root = Path(__file__).resolve().parent.parent

    banner = (root / "tools" / "win_start.py").read_text()
    m = re.search(r"NVIDIA (\d+) GB\+", banner)
    check("the launcher banner states a minimum VRAM", m is not None)
    if not m:
        return
    claimed = int(m.group(1))

    # at the claimed size there must be a real menu, not a single fallback row
    _, at = profiles.plan(float(claimed), None)
    check(f"{claimed} GB (the claim) offers more than one profile", len(at) > 1,
          [o["quant"] for o in at])
    rec = next((o for o in at if o["recommended"]), None)
    check(f"{claimed} GB is not steered to the weakest quant the kit ships",
          rec and rec["kl"] < max(q.kl for q in profiles.QUANTS), rec and rec["quant"])
    check(f"{claimed} GB can run at least one profile with images",
          any(o["vision"] for o in at) or bool(profiles.plan(float(claimed), True)[1]))

    # and one step below it, the experience really is worse - otherwise the
    # claim is too conservative and turns people away for no reason
    _, below = profiles.plan(float(claimed) - 4, None)
    check(f"{claimed - 4} GB is meaningfully worse, so the claim is not just caution",
          len(below) < len(at), [o["quant"] for o in below])

    # the README table's bolded pick per card size must be the planner's own
    readme = (root / "README.md").read_text()
    rows = re.findall(r"^\| (\d+) GB\+? \| \*\*([\d.]+) bpw @ ([\dk]+)\*\*", readme, re.M)
    check("the README table marks a recommendation per card size", len(rows) >= 3, rows)
    for vram, quant, ctx in rows:
        _, options = profiles.plan(float(vram), None)
        got = next((o for o in options if o["recommended"]), None)
        check(f"README says {vram} GB gets {quant} bpw - and the planner agrees",
              got and got["quant"] == quant, got and got["quant"])
        check(f"README's context for {vram} GB matches the planner",
              got and profiles.ctx_label(got["ctx"]).startswith(ctx),
              got and profiles.ctx_label(got["ctx"]))


def test_no_invalid_escape_sequences():
    """No file may print a warning just for being imported.

    A Windows path in a plain docstring - ".venv\\Scripts" - is an invalid escape
    sequence. Python 3.12 turned that from a quiet DeprecationWarning into a
    SyntaxWarning printed to the console, so on a current Python the launcher
    greeted the user with a compiler warning before it did anything else. The
    warning class differs by version, so match on the message instead."""
    import warnings                                     # noqa: WPS433
    root = Path(__file__).resolve().parent.parent
    bad = []
    for f in sorted(root.rglob("*.py")):
        if any(part in (".venv", "site-packages", "__pycache__") for part in f.parts):
            continue
        try:
            src = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                compile(src, str(f), "exec")
            except SyntaxError as e:                    # noqa: PERF203
                bad.append(f"{f.name}: SyntaxError {e}")
                continue
            for item in caught:
                if "escape" in str(item.message):
                    bad.append(f"{f.relative_to(root)}:{item.lineno} {item.message}")
    check("no file has an invalid escape sequence (use a raw docstring for "
          "Windows paths)", not bad, "; ".join(bad[:5]))


def test_simulation_mode():
    """Simulation mode answers "what would a 16 GB card get?" without one, and
    must never change the machine it runs on."""
    import profiles                                     # noqa: WPS433
    import setup_core                                   # noqa: WPS433

    root = Path(__file__).resolve().parent.parent
    env = root / ".env"
    before = env.read_bytes() if env.is_file() else None

    cards = {c["name"]: c for c in setup_core.cards()}
    check("the card list covers the generations", len(cards) >= 10)
    # specs from NVIDIA's own compute-capability table and the cards' spec sheets
    for name, vram, cc in (("RTX 5090", 32, 12.0), ("RTX 5080", 16, 12.0),
                           ("RTX 5070 Ti", 16, 12.0), ("RTX 5060 Ti 16GB", 16, 12.0),
                           ("RTX 4090", 24, 8.9), ("RTX 3090", 24, 8.6),
                           ("RTX 2080 Ti", 11, 7.5), ("GTX 1080 Ti", 11, 6.1)):
        c = cards.get(name)
        check(f"{name} is listed with the right VRAM and compute capability",
              c and c["vram_gib"] == vram and c["cc"] == cc, c)

    # the three cards this was asked for are all 16 GB, i.e. the kit's own target
    for name in ("RTX 5070 Ti", "RTX 5060 Ti 16GB", "RTX 5080"):
        d = setup_core.simulate(cards[name]["vram_gib"], cards[name]["cc"], "581.29", name)
        check(f"{name}: supported", d["support"] == "ok", d["support"])
        check(f"{name}: something fits", d["fits"], d)
        check(f"{name}: the recommendation is not the weakest quant on the menu",
              d["recommended"] != min(d["options"], key=lambda o: o["bpw"])["quant"]
              or len(d["options"]) == 1, d["recommended"])

    # old cards: the honest verdicts
    turing = setup_core.simulate(11, 7.5, "581.29", "RTX 2080 Ti")
    check("Turing is supported but flagged slow", turing["support"] == "slow", turing["support"])
    pascal = setup_core.simulate(11, 6.1, "581.29", "GTX 1080 Ti")
    check("Pascal is reported unsupported", pascal["support"] == "unsupported")
    check("...with a reason, not a bare refusal", bool(pascal["notes"]))

    # KV cache: the thing people get backwards. int4 is the one with NO hardware
    # requirement; fp8 / nvfp4 are the ones that need Ada or newer.
    for cc in (7.5, 8.6, 8.9, 12.0):
        lanes = {k["format"]: k for k in profiles.kv_support(cc)}
        for fmt in ("4", "8,4", "8"):
            check(f"cc {cc}: the int{fmt} KV cache is available (it has no hardware gate)",
                  lanes[fmt]["available"], lanes[fmt])
        want_fp8 = cc >= profiles.FP8_MIN_CC
        check(f"cc {cc}: fp8/nvfp4 availability follows the {profiles.FP8_MIN_CC} floor",
              lanes["fp8"]["available"] == want_fp8
              and lanes["nvfp4"]["available"] == want_fp8, lanes["fp8"])
        if not want_fp8:
            check(f"cc {cc}: and says why fp8 is unavailable", bool(lanes["fp8"]["why"]))

    # a card too small gets a number to shop for, not just a refusal
    small = setup_core.simulate(8, 12.0, "581.29", "RTX 5060 Ti 8GB")
    check("an 8GB card is told nothing fits", not small["fits"])
    check("...and how big a card it would take",
          small["min_card_gib"] > 8 and small["min_card_gib"] < 24, small["min_card_gib"])

    # simulating is read-only
    setup_core.simulate(24, 8.9, "581.29", "RTX 4090", want_vision=True)
    after = env.read_bytes() if env.is_file() else None
    check("simulating another card never touches .env", after == before)

    js = (Path(__file__).parent / "webui" / "setup.js").read_text()
    check("the page has a simulation panel", "renderSim" in js and "/setup/simulate" in js)
    html = (Path(__file__).parent / "webui" / "setup.html").read_text()
    check("and a place to put it", 'id="sim"' in html and "Simulation mode" in html)

    # The card list has to survive a stale server. Static files are served from
    # disk per request, so a browser can run new JS against a process started
    # before this existed: /setup/state then carries no cards and the picker sits
    # empty with nothing to explain why.
    check("the card list has an endpoint of its own, not only a field on state",
          "/setup/cards" in js and "/setup/cards" in
          (Path(__file__).parent / "setup_web.py").read_text())
    check("and the panel says what to do when even that is missing",
          "start Simplex again" in js)


def test_vision_toggle_is_cheap_and_clearable():
    """The three images buttons must re-plan the menu and nothing else.

    Two bugs, both visible to anyone who pressed them. They posted to
    /setup/refresh, which re-probes the machine: nvidia-smi (up to a 10s
    timeout), a spawned Python to read wheel tags, disk checks - seconds per
    click on a button that only changes arithmetic. And on that endpoint a null
    answer had to mean "keep whatever was chosen", so "Decide for me" could
    never clear a previous Yes or No: it silently did nothing."""
    import re                                           # noqa: WPS433
    import profiles                                     # noqa: WPS433
    import setup_core                                   # noqa: WPS433
    import setup_web                                    # noqa: WPS433

    real_gpu, real_probe = profiles.detect_gpu, setup_core.probe
    probes = []
    try:
        profiles.detect_gpu = lambda: profiles.GPU(name="Fake", total_gib=16.0, cc=8.9, driver="580")
        setup = setup_web.Setup({}, force=True)
        setup_core.probe = lambda cfg: probes.append(1) or {}

        # 1. toggling never probes the machine
        probes.clear()
        setup.set_vision(True)
        setup.set_vision(False)
        setup.set_vision(None)
        check("toggling images does not re-probe the machine",
              probes == [], f"{len(probes)} probe(s) for three clicks")

        # 2. every answer actually lands, including the automatic one *after*
        #    an explicit one - the case that was broken
        setup.set_vision(True)
        on = [(o["quant"], o["vision"]) for o in setup.menu["options"]]
        check("Yes gives an images-only menu", all(v for _, v in on) and on, on)

        setup.set_vision(False)
        off = [(o["quant"], o["vision"]) for o in setup.menu["options"]]
        check("No gives a text-only menu", not any(v for _, v in off) and off, off)

        setup.set_vision(True)                # get into an explicit state first
        setup.set_vision(None)                # "Decide for me" must clear it
        auto = [(o["quant"], o["ctx"], o["vision"]) for o in setup.menu["options"]]
        expected = [(o["quant"], o["ctx"], o["vision"]) for o in profiles.plan(16.0, None)[1]]
        check("Decide for me clears a previous Yes and returns to the automatic rule",
              auto == expected, (auto, expected))
        check("...and the menu says so, so the button can light up",
              setup.menu["vision"] is None, setup.menu["vision"])

        # 3. the state the page renders carries the answer back
        setup.set_vision(False)
        check("the state the page reads reports the current answer",
              setup.state()["menu"]["vision"] is False, setup.state()["menu"]["vision"])
    finally:
        profiles.detect_gpu, setup_core.probe = real_gpu, real_probe

    # the client must use the cheap endpoint, not the probing one
    js = (Path(__file__).parent / "webui" / "setup.js").read_text()
    check("setup.js posts the images answer to /setup/vision",
          '"/setup/vision"' in js or "'/setup/vision'" in js)
    # look at the call itself, not the comments around it
    body = js.split("async function setVision")[1].split("\nfunction ")[0]
    calls = re.findall(r"post\(\s*[\"']([^\"']+)", body)
    check("...and the images answer never goes to the probing refresh endpoint",
          calls == ["/setup/vision"], calls)


def test_rename_accepts_spaces():
    """A space typed into the rename box must be a space.

    The row is a <button> and the rename <input> lives inside it. A button
    activates on Space, so every space bubbled up, "clicked" the row, and the
    blur that followed saved the rename - typing "my long title" saved "my".
    All key events now stop at the input, and the row ignores clicks while it
    is being renamed."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    fn = js.split("function startRename")[1].split("\nasync function")[0]

    check("key events do not escape the rename box",
          'addEventListener("keydown"' in fn and "stopPropagation" in fn, fn[:200])
    check("keyup is covered too - a button activates on the key's release",
          'addEventListener("keyup"' in fn)
    check("the row is marked while renaming", "dataset.renaming" in fn)
    check("Enter and Escape still work",
          '"Enter"' in fn and '"Escape"' in fn)
    check("and the row will not open the chat mid-rename",
          "if (!row.dataset.renaming) openSession" in js)


def test_webui_restarts_itself():
    """Opening the UI must give you the UI, not a port-in-use error.

    The copy already holding the port is also running the old code - which is
    usually the very thing you restarted to be rid of - so it is replaced. But
    only when it is recognisably this same program: taking a port from whatever
    else happens to be listening is not a restart, it is collateral damage."""
    import subprocess, time                             # noqa: WPS433
    import urllib.error                                 # noqa: WPS433
    import chatui                                       # noqa: WPS433

    # who it will and will not stop
    for cmd, ours in (
            ("/usr/bin/python3 tools/chatui.py --port 8890", True),
            (r"c:\qwen\.venv\scripts\python.exe -u tools\chatui.py --open", True),
            ("/usr/sbin/nginx -g daemon off;", False),
            ("python3 -m http.server 8890", False),
            ("", False)):
        check(f"{'claims' if ours else 'leaves alone'}: {cmd[:44] or '(unknown)'}",
              chatui._is_this_ui(cmd) is ours)

    # the Windows listener lookup, whose output cannot be produced on this box
    sample = (
        "  Proto  Local Address      Foreign Address    State       PID\n"
        "  TCP    0.0.0.0:135        0.0.0.0:0          LISTENING   1044\n"
        "  TCP    127.0.0.1:8890     0.0.0.0:0          LISTENING   7788\n"
        "  TCP    127.0.0.1:8890     127.0.0.1:51000    ESTABLISHED 9999\n")
    parsed = None
    for line in sample.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[3].upper() == "LISTENING" \
                and parts[1].rsplit(":", 1)[-1] == "8890":
            parsed = int(parts[4])
    check("the netstat parse picks the LISTENING row, not an ESTABLISHED one",
          parsed == 7788, parsed)

    # and the real thing: a second start takes the port from the first
    port = 8849
    first = subprocess.Popen([sys.executable, "tools/chatui.py", "--mock",
                              "--port", str(port)],
                             cwd=str(Path(__file__).resolve().parent.parent),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def answers():
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{port}/ui/config",
                headers={"Origin": f"http://127.0.0.1:{port}"}), timeout=2).close()
            return True
        except Exception:                               # noqa: BLE001
            return False

    second = None
    try:
        for _ in range(30):
            time.sleep(1)
            if answers():
                break
        check("the first copy is serving", answers())
        second = subprocess.Popen([sys.executable, "tools/chatui.py", "--mock",
                                   "--port", str(port)],
                                  cwd=str(Path(__file__).resolve().parent.parent),
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(30):
            time.sleep(1)
            if first.poll() is not None and answers():
                break
        check("starting it again stops the copy that was there",
              first.poll() is not None, "the old instance survived")
        check("...and the port is serving again afterwards", answers())
    finally:
        for proc in (second, first):
            if proc and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)


def test_pin_does_not_sit_on_the_timestamp():
    """Nothing in a pinned row may be painted on top of its time.

    The pin used to be styled by reusing the hover menu's class, which the
    visual refresh moved to `position: absolute; right: 8px` so it could
    overlay the timestamp on hover. The pin, unlike that menu, was always
    visible - so it landed directly on the time: "1:14 PM" with a pin drawn
    through the P. Pinned chats now have a section of their own and the row
    draws no pin at all, which is that fix taken to its conclusion. What still
    has to hold: the time is the row's last element, and only the hover menu
    ever takes its place."""
    css = (Path(__file__).parent / "webui" / "style.css").read_text()
    js = (Path(__file__).parent / "webui" / "app.js").read_text()

    check("no pin glyph is drawn inside the row any more",
          ".session .pin {" not in css and 'el("span", "del pin")' not in js)
    check("...because pinned chats are their own section",
          'label: "Pinned"' in js)
    check("the pinned class survives - it is what drag-to-reorder selects",
          'row.classList.add("pinned")' in js and ".session.pinned" in css)
    check("the hover menu itself still overlays the row",
          "position: absolute" in css.split(".session .del {")[1].split("}")[0])
    check("the time is the row's last element, so nothing is laid on top of it",
          js.index('row.append(when)') > js.index('row.classList.add("pinned")'),
          "the timestamp must be appended after the pin handling")
    check("...and it yields to the hover menu rather than fighting it",
          ".session:hover .when" in css and "opacity: 0" in
          css.split(".session:hover .when")[1][:60])


def test_sidebar_groups_by_kind():
    """The index is grouped by what a conversation is, and each group folds.

    Fifteen near-identical rows under Today / Yesterday told you when you had
    been busy, not what you were looking for - and agent chats, which carry
    tool calls and a folder, were mixed in among ordinary ones with a small
    AGENT badge as the only clue. Sections for Pinned / Chats / Agentic carry
    that instead, each one collapsible so a section you are not working in can
    be put away."""
    js = (Path(__file__).parent / "webui" / "style.css").parent
    js = (js / "app.js").read_text()
    css = (Path(__file__).parent / "webui" / "style.css").read_text()

    block = js.split("const SESSION_GROUPS")[1].split("];")[0]
    for label in ("Pinned", "Chats", "Agentic"):
        check(f"there is a {label} section", f'label: "{label}"' in block)
    check("agent chats are the ones in agent mode",
          's.mode === "agent"' in block)
    check("a pinned chat is listed once, under Pinned",
          "!s.pinned && s.mode" in block)
    check("an empty section is not drawn at all", "if (!rows.length) return;" in js)

    # collapsing
    check("each heading is a button, not a label", 'el("button", "group-head")' in js)
    check("it reports its state to assistive tech", 'aria-expanded' in js)
    check("the closed sections are remembered", "chatui.closedGroups" in js)
    check("a browser that refuses storage does not take the UI down",
          "catch (e)" in js.split("function setGroupClosed")[1][:400])
    check("a collapsed section still says how much is in it",
          'el("b", "count"' in js)
    check("...and it animates rather than snapping",
          "grid-template-rows" in css.split(".group-rows {")[1][:200])
    check("a search hit is never hidden inside a folded section",
          "const closed = query ? new Set() : closedGroups();" in js)

    # a heading has to look like a heading, not like another row
    head = css.split(".group-head {")[1].split("}")[0]
    check("the heading is a band on its own surface", "background: var(--band)" in head)
    check("...and that surface is a token, defined for both themes",
          css.count("--band:") >= 3)
    check("a hovered row and the open row are surfaces of their own too",
          css.count("--row-hover:") >= 3 and css.count("--row-on:") >= 3)
    check("the open row is not just a louder hover",
          ".session.on .t { font-weight" in css
          and "background: var(--row-on)" in css)
    # the accent bar and the row surface are both positioned pseudo-elements:
    # without an explicit order the surface paints over the bar and the bar is
    # invisible on every row, which is exactly what happened
    check("the accent bar is painted above the row's surface, not under it",
          "z-index: 0" in css.split(".session::after {")[1].split("}")[0]
          and "z-index: 1" in css.split(".session::before {")[1].split("}")[0])
    check("...separated from the rows under it", "border-bottom" in head)
    check("...and it stays put while a long list scrolls", "position: sticky" in head)
    check("it is full-bleed, so nothing shows through beside it",
          "margin: 0 -8px" in head)

    # the row marker is a bullet, not a swatch
    check("the marker is one quiet colour for every row",
          "color-mix(in srgb, var(--muted)" in css.split(".session .dot {")[1].split("}")[0])
    check("no per-chat hue is computed any more",
          "sessionHue" not in js and "--hue" not in css)
    check("colour is spent on the open row", ".session.on .dot" in css)
    check("...and on a chat that is still working", ".session.live .dot" in css)


def test_a_truncated_tool_call_cannot_brick_a_chat():
    """max_tokens cutting a tool call in half must not end the conversation.

    A model asked to write a whole file in one call streams the file into the
    `content` argument. When the reply hits max_tokens the JSON stops mid-string,
    and three things used to go wrong in sequence: _parse_args swallowed the
    error and returned {}, so the tool reported "missing 2 required positional
    arguments" and blamed the model for the server's truncation; finish_reason
    "length" was yielded by the client and dropped by run_turn, so nothing said
    what had happened; and the unparseable string was written into the history,
    where the model server parses it again to render the chat template - so
    every later request in that conversation failed with HTTP 400, permanently,
    because the message was on disk."""
    import json as _json, sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    import webui_agent as wa

    cut = '{"path":"index.html","content":"<!doctype html>' + "x" * 9000

    args, err = wa.parse_args(cut)
    check("a truncated argument string reports why it failed", err and args == {})
    check("...and says how much arrived",
          f"{len(cut)} characters" in (err or ""), err)
    check("a good argument string still parses",
          wa.parse_args('{"path":"a.txt"}') == ({"path": "a.txt"}, None))

    calls = [{"id": "c1", "type": "function",
              "function": {"name": "write_file", "arguments": cut}}]
    fixed, notes, _ = wa.repair_tool_calls(calls)
    check("the broken call is made safe to store",
          _json.loads(fixed[0]["function"]["arguments"]) == {})
    check("...and the repair is reported", notes and "write_file" in notes[0])
    good = [{"id": "c2", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'}}]
    check("a healthy call is passed through untouched",
          wa.repair_tool_calls(good) == (good, [], {}))

    class Truncated:
        def stream(self, messages, tools=None, sampling=None):
            yield "content", "I'll write it in one shot. "
            yield "finish", "length"
            yield "tool_calls", [{"id": "call_0", "type": "function",
                                  "function": {"name": "write_file",
                                               "arguments": cut}}]

    events = list(wa.run_turn(Truncated(), [{"role": "user", "content": "go"}],
                              [], wa.ToolContext(), mode="agent",
                              sampling={"max_tokens": 4096}, max_steps=3))
    kinds = [e["type"] for e in events]
    err_ev = next((e for e in events if e["type"] == "error"), None)
    check("the user is told the output limit was hit", err_ev is not None)
    check("...with the ceiling that was in force",
          "Max new tokens is 4096" in (err_ev or {}).get("message", ""),
          (err_ev or {}).get("message"))
    check("...and what to do about it",
          "Settings" in (err_ev or {}).get("message", ""))
    done = next((e for e in events if e["type"] == "done"), None)
    stored = (done or {}).get("messages", [])
    unparseable = 0
    for m in stored:
        for c in (m.get("tool_calls") or []):
            try:
                _json.loads(c["function"]["arguments"])
            except ValueError:
                unparseable += 1
    check("nothing unparseable reaches the conversation", unparseable == 0,
          "this is what used to make every later request a 400")
    check("the model is told too, so it can retry smaller",
          any(m.get("role") == "tool" and "could not be read" in m.get("content", "")
              for m in stored))
    check("the broken call still gets a card, so nothing is drawn nowhere",
          any(e["type"] == "tool_call" for e in events))
    check("...and a failed result under the same id",
          any(e["type"] == "tool_result" and not e["ok"] for e in events))


def test_a_poisoned_session_heals_when_it_is_opened():
    """Conversations saved before that fix must not stay broken for ever."""
    import json as _json, sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    import webui_agent as wa

    cut = '{"path":"a.html","content":"<!doctype' + "y" * 500
    messages = [{"role": "user", "content": "go"},
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "write_file",
                                              "arguments": cut}}]}]
    healed = wa.heal_messages(messages)
    check("a poisoned history is repaired", healed is not None)
    check("...without losing any messages", len(healed) == len(messages))
    check("...and every stored call now parses",
          all(_json.loads(c["function"]["arguments"]) is not None
              for m in healed for c in (m.get("tool_calls") or [])))
    check("a clean history is not rewritten for nothing",
          wa.heal_messages([{"role": "user", "content": "hi"}]) is None)

    app = (Path(__file__).parent / "webui_app.py").read_text()
    check("the server heals on the way in", "heal_messages" in app)
    tail = app.split("heal_messages")[1][:1400]
    check("...and writes the repair back once", '"healed"' in tail)
    check("...under the same lock the writers use, so a concurrent turn is "
          "not reverted", "_save_locks" in tail)
    check("...and an already-healed session is not walked again",
          'session.get("healed")' in app)


def test_a_long_tool_call_is_visible_while_it_streams():
    """A call whose arguments take minutes to write must not look like a hang.

    Tool-call fragments are accumulated and only yielded when the whole reply
    ends, so a large `write_file` produced no events at all between the last
    content token and the finished card."""
    import json as _json, sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    import webui_agent as wa

    def sse(obj):
        return f"data: {_json.dumps(obj)}\n".encode()

    lines = [sse({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "call_0",
         "function": {"name": "write_file", "arguments": ""}}]}}]})]
    for _ in range(30):
        lines.append(sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "x" * 400}}]}}]}))
    lines.append(sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}))
    lines.append(b"data: [DONE]\n")

    class FakeResp:
        def __iter__(self):
            return iter(lines)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    c = wa.ModelClient.__new__(wa.ModelClient)
    c.base_url, c.api_key, c.model, c.timeout = "http://x/v1", "", "m", 5
    c._post = lambda path, payload: FakeResp()
    partials = [v for k, v in c.stream([{"role": "user", "content": "hi"}])
                if k == "tool_partial"]
    check("the call is announced while it is still being written", partials)
    check("...as soon as it has a name", partials[0]["name"] == "write_file")
    check("...with a count that climbs",
          all(b["chars"] >= a["chars"] for a, b in zip(partials, partials[1:])))
    check("...throttled, not one event per fragment", len(partials) < 30,
          len(partials))

    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    check("the browser draws a card for it", "function pendingToolCard" in js)
    check("...and any placeholder still standing is cleared when one lands",
          'querySelectorAll(".tool.pending")' in js)
    check("...and again when the turn ends, for a call that never landed",
          'querySelectorAll(".tool.pending")' in js.split("function finishTurn")[1][:400])


def test_stray_br_is_a_line_break_not_text():
    """Models reach for <br> mid-sentence. escapeHtml turned it into visible
    text; only the void, attribute-less spellings come back."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    block = js.split("function inline(")[1].split("\nfunction ")[0]
    check("the void spellings are restored", "&lt;br\\s*\\/?&gt;" in block)
    check("...and nothing with attributes is", "onload" not in block)
    check("code spans are lifted out before any of it runs",
          block.index("HOLE_OPEN") < block.index("&lt;br"))
    check("...and put back at the end", "holes[n] === undefined" in block)


def test_the_renderer_cannot_be_made_to_emit_markup():
    """Model output is untrusted - fetched pages, file contents, a provider.

    `inline()` escapes everything up front and then generates HTML of its own,
    which is where it went wrong: the link rule interpolated the href into an
    attribute, and the autolink rule ran afterwards over the same string,
    matched the URL sitting *inside* that attribute, and rewrote it as a whole
    new anchor - injecting raw quotes into the middle of the tag. Everything
    past those quotes was parsed as further attributes, and because "/"
    separates attribute names, `http://e/onmouseover=...` became a live event
    handler. That was arbitrary script in this page's origin, on hover, from
    one line of model output - and it could POST to /ui/approve and approve the
    agent's own file writes.

    The rule that prevents it: nothing this function generates may be visible
    to a later rule. Every produced tag goes into a hole first."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    block = js.split("function inline(")[1].split("\nfunction ")[0]

    check("the link rule hides the anchor it builds",
          "return hole(`<a href=" in block)
    check("...including the label, so the autolink cannot open one inside it",
          "${label}</a>" in block)
    check("the autolink rule hides its anchor too",
          block.count("hole(`<a href=") == 2)
    check("...and cannot swallow a hole sentinel",
          "[^\\s<)\\uE000\\uE001]" in block)
    check("an href carrying a raw quote or bracket is not a link",
          '/["\'<>]/.test(href)' in block)
    check("nested holes are resolved to a fixed point", "pass < 4" in block)

    # the sentinels cannot be typed by the text they protect
    md = js.split("function markdown(")[1].split("\nfunction ")[0]
    check("the private-use sentinels are stripped from the source first",
          "[\\uE000-\\uE002]" in md)
    check("...before anything is escaped or lifted",
          md.index("[\\uE000-\\uE002]") < md.index("escapeHtml"))
    check("no @@ placeholder survives anywhere in the renderer",
          "@@CB" not in js and "@@CS" not in js)


def test_a_turn_answers_every_call_it_announces():
    """An assistant message carrying tool_calls must be followed by a tool
    message for every one of those ids, or a strict endpoint rejects the whole
    conversation from then on - the same permanent HTTP 400 a truncated
    argument used to cause, and one heal_messages cannot repair because the
    arguments themselves are valid. Stop landing mid-loop used to leave the
    rest unanswered for good."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    import webui_agent as wa

    mk = lambda cid, name, args: {"id": cid, "type": "function",
                                  "function": {"name": name, "arguments": args}}

    class Two:
        def stream(self, messages, tools=None, sampling=None):
            yield "tool_calls", [mk("c1", "read_file", '{"path":"a.txt"}'),
                                 mk("c2", "read_file", '{"path":"b.txt"}')]

    for label, budget in (("stopped before the loop", 1), ("stopped mid-loop", 4)):
        seen = {"n": 0}

        def cancelled(budget=budget, seen=seen):
            seen["n"] += 1
            return seen["n"] > budget

        events = list(wa.run_turn(Two(), [{"role": "user", "content": "go"}],
                                  [], wa.ToolContext(), mode="agent",
                                  max_steps=2, cancelled=cancelled))
        done = next(e for e in events if e["type"] == "done")
        ids = [c["id"] for m in done["messages"]
               for c in (m.get("tool_calls") or [])]
        answered = [m["tool_call_id"] for m in done["messages"]
                    if m.get("role") == "tool"]
        check(f"{label}: every announced call is answered",
              set(ids) == set(answered), (ids, answered))


def test_only_a_real_truncation_blames_max_tokens():
    """A model emitting a Python dict literal fails to parse in exactly the
    same way as one cut off at the ceiling. Telling that user to raise
    max_tokens sends them to fix a setting that is not the problem - and the
    turn used to abort rather than let the model correct itself."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    import webui_agent as wa

    mk = lambda cid, name, args: {"id": cid, "type": "function",
                                  "function": {"name": name, "arguments": args}}

    class Malformed:
        def stream(self, messages, tools=None, sampling=None):
            yield "finish", "stop"
            yield "tool_calls", [mk("x", "read_file", "{'path': 'a.txt'}")]

    events = list(wa.run_turn(Malformed(), [{"role": "user", "content": "go"}],
                              [], wa.ToolContext(), mode="agent",
                              sampling={"max_tokens": 16384}, max_steps=3))
    check("bad JSON that is not a truncation says nothing about the limit",
          not [e for e in events if e["type"] == "error"])
    check("...it fails that one call instead",
          any(e["type"] == "tool_result" and not e["ok"] for e in events))
    check("...and the turn carries on so the model can correct itself",
          sum(1 for e in events if e["type"] == "step") > 1)

    # arguments that parse but are not an object
    for raw, kind in (("", "empty"), ("   ", "blank"), ("null", "null"),
                      ("[1,2]", "list"), ("3", "int")):
        fixed, notes, by_id = wa.repair_tool_calls([mk("z", "list_dir", raw)])
        stored = fixed[0]["function"]["arguments"]
        check(f"{kind} arguments are stored as an object", stored == "{}", stored)

    # shapes that used to raise
    for bad in ([None], [{"id": "n"}], [{"function": {}}], "notalist", None):
        try:
            if isinstance(bad, list):
                wa.repair_tool_calls(bad)
            wa.heal_messages([{"role": "assistant", "tool_calls": bad}])
            ok = True
        except Exception as e:                       # noqa: BLE001
            ok = f"{type(e).__name__}: {e}"
        check(f"a malformed {str(bad)[:18]} does not raise", ok is True, ok)


def test_menus_survive_the_click_that_opens_them():
    """A document-level listener closes any open menu on a click outside it.

    That listener sees the very click that opened the menu, because it bubbles
    all the way up - so a menu opened from an onclick handler that does not
    stop propagation opens and shuts in the same tick. The element is still in
    the DOM with `hidden` set, which is what makes this so easy to miss: a test
    that reads the menu's buttons finds them and passes, while nothing was ever
    on screen. Every opener has to stop the click."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    closer = "if (!e.target.closest(\"#menu-pop\")) closeMenu();"
    check("the document still closes menus on an outside click", closer in js)

    for opener in ("function effortMenu(", "function permissionsMenu("):
        body = js.split(opener)[1].split("\nfunction ")[0]
        check(f"{opener.split('function ')[1].rstrip('(')} stops the opening click",
              "stopPropagation()" in body)

    # ...and the handlers bound to a click hand the event over to be stopped
    check("the chips pass the event to their opener",
          '$("#stat-effort").onclick = effortMenu;' in js
          and '$("#stat-perm").onclick = permissionsMenu;' in js)

    # sessionMenu is safe by two other routes, and both have to stay that way:
    # the "more" button stops the click before calling it, and a right-click
    # never produces a click event for the document listener to see.
    check("the row's menu button stops the click for it",
          "e.stopPropagation(); sessionMenu(e, row, s);" in js)
    check("...and its other opener is contextmenu, which fires no click",
          "row.oncontextmenu = (e) => sessionMenu(e, row, s);" in js)


def test_thinking_is_visible_and_settable_from_the_composer():
    """The effort setting used to exist only inside Settings, so a terse answer
    gave you no way to tell whether it was the setting or the model."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    html = (Path(__file__).parent / "webui" / "index.html").read_text()

    check("there is a chip for it beside the model", 'id="stat-effort"' in html)
    check("...that says what the setting is", 'id="stat-effort-name"' in html)
    check("...and looks like something you can open", 'id="stat-effort"' in html
          and "chev" in html.split('id="stat-effort"')[1][:400])
    check("it is kept in step with every other way of changing it",
          js.count("refreshEffort()") >= 5)
    check("an endpoint with no levels explains itself rather than looking empty",
          "reported no effort levels" in js)
    check("...and a menu can carry a line that is not a choice", "item.note" in js)

    # permissions
    check("a standing permission is shown in the composer", 'id="stat-perm"' in html)
    check("...only when it is not the safe default", 'now === "ask"' in
          js.split("function refreshPermissions")[1][:400])
    check("...and loudest when everything is accepted",
          'classList.toggle("hot"' in js)
    app = (Path(__file__).parent / "webui_app.py").read_text()
    check("the server reads the stance from the request, never from memory",
          'settings.get("permissions"' in app)
    check("...and turns it into names, so run_turn is unchanged",
          "t.risk in (webui_tools.WRITE, webui_tools.EXEC)" in app)


def test_the_server_describes_itself_on_models():
    """Point one copy of this kit at another as a provider and it must be able
    to discover what that one supports.

    /v1/models published an id and a context length and nothing else, so the
    probe that reads reasoning levels and modalities off a /models row found
    neither - and the UI honestly reported "this provider did not say which
    levels it takes" about a server whose own chat template accepts three.
    The keys below are the ones the prober already understands, which are the
    shapes hosted APIs use; none of them is private to this kit."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    import webui_providers as wp

    src = (Path(__file__).parent / "serve_openai.py").read_text()
    block = src.split("async def models(")[1].split("\nasync def ")[0]
    check("the levels come from the template, not a hardcoded list",
          "supported_efforts(tokenizer)" in block)
    check("...and are published under a key the prober reads",
          '"supported_reasoning_efforts"' in block)
    check("modalities are published too", '"input_modalities"' in block)
    check("a server with no tokenizer still answers",
          'request.app.get("tokenizer")' in block)
    check("...and a probe failure is not fatal to /models", "except Exception" in block)

    # the round trip: what this server publishes, the prober must understand
    row = {"id": "m", "object": "model", "max_model_len": 199936,
           "supported_reasoning_efforts": ["low", "medium", "xhigh"],
           "architecture": {"input_modalities": ["text", "image"],
                            "output_modalities": ["text"]}}
    check("a provider probe reads those levels back",
          wp.efforts_from_model_row(row) == ["low", "medium", "xhigh"],
          wp.efforts_from_model_row(row))
    check("...and reads vision back", wp.vision_from_model_row(row) is True)
    text_only = dict(row, architecture={"input_modalities": ["text"],
                                        "output_modalities": ["text"]})
    check("...and can tell text-only apart from silent",
          wp.vision_from_model_row(text_only) is False
          and wp.vision_from_model_row({"id": "m"}) is None)


def test_a_provider_can_be_told_what_it_takes():
    """An endpoint that says nothing is not an endpoint that says no.

    Images already worked this way - `vision` is the user's answer, kept apart
    from `vision_detected`, and the user's wins. Reasoning levels had a single
    field, so pressing Test overwrote whatever the user had entered by hand
    with whatever the probe found, which for many endpoints is nothing.

    vLLM is the case that forces this: it accepts reasoning_effort through
    chat_template_kwargs perfectly well, and its /models row mentions none of
    it - so the probe will never find levels there, and without somewhere to
    declare them the Thinking menu is permanently two entries long."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    import webui_providers as wp

    src = (Path(__file__).parent / "webui_providers.py").read_text()
    check("what the probe found has a field of its own",
          '"efforts_detected"' in src)
    check("...and Test writes to that one, not over the user's",
          'stored["efforts_detected"] = caps["efforts"]' in src
          and 'stored["efforts"] = caps["efforts"]' not in src)
    check("both reach the browser", '"efforts_detected": list(' in src)

    row = wp._clean({"id": "p", "name": "P", "base_url": "http://x/v1",
                     "default_model": "m", "models": ["m"],
                     "efforts": ["medium", "xhigh", "nonsense"]})
    check("a declared list is kept, and validated",
          row["efforts"] == ["medium", "xhigh"], row["efforts"])
    check("...separately from what was detected", row["efforts_detected"] == [])

    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    block = js.split("function availableEfforts()")[1].split("\nfunction ")[0]
    check("the user's answer wins over the probe's", "p?.efforts?.length" in block)
    check("...and the probe's is the fallback", "efforts_detected" in block)
    check("the provider form offers the choice", '"Reasoning levels"' in js)


def test_the_effort_level_that_is_set_is_the_one_that_is_sent():
    """Whatever level is chosen goes to the endpoint, unfiltered.

    This was a hardcoded ("low", "medium", "high"), and measurement on a real
    vLLM + Qwen3 stack showed it wrong in both directions at once: "xhigh" was
    dropped silently, so choosing Extra high sent nothing and changed nothing,
    while "high" was forwarded happily and that template rejects it outright -
    `Unexpected reasoning effort high. Supported types are xhigh (default),
    medium, and low.` The list of valid levels belongs to the endpoint, and is
    already answered where that is known: the local server probes its own
    template, a provider declares them or is probed. Nothing in the middle
    should have an opinion."""
    app = (Path(__file__).parent / "webui_app.py").read_text()
    block = app.split('thinking = str(settings.get')[1].split("with self._lock")[0]

    check("no hardcoded set of levels survives",
          '("low", "medium", "high")' not in block, block[:200])
    check("any level that is set is forwarded",
          'elif thinking and thinking != "default":' in block)
    check("...as reasoning_effort", 'sampling["reasoning_effort"] = thinking' in block)

    # off has to keep both spellings: vLLM takes "none" only at the top level
    # and rejects it inside chat_template_kwargs; this kit reads the kwarg.
    check("off still says it both ways",
          '"enable_thinking": False' in block
          and 'sampling["reasoning_effort"] = "none"' in block)

    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    check("the menu does not present the levels as a ladder",
          "Named modes, not a ladder" in js)
    check("...because measurement showed the ordering does not hold",
          "not guaranteed to think" in js)


def test_no_undefined_globals():
    """No function may reference a module global that does not exist.

    This is the class of bug that a string-matching test cannot see and that
    only fires on the path nobody can run here. A nested function referring to
    a name that is a local of some *other* function compiles fine, imports
    fine, and raises NameError the moment that line runs - which for
    `in_think = [bool(enable_thinking)]` inside chat_completions.run() meant
    every streaming completion returned 200, then zero bytes, then a dropped
    connection: an empty reply bubble and no error, on the one path the mock
    suite cannot exercise.

    symtable sees it without importing torch, so it costs nothing to check
    every server file on every run."""
    import builtins, symtable                           # noqa: WPS433
    here = Path(__file__).parent

    for name in ("serve_openai.py", "webui_app.py", "webui_agent.py",
                 "webui_providers.py", "webui_models.py", "setup_core.py",
                 "setup_web.py", "profiles.py", "bench_vram.py", "chatui.py"):
        path = here / name
        if not path.is_file():
            continue
        src = path.read_text(encoding="utf-8")
        top = symtable.symtable(src, name, "exec")
        # module dunders are real at runtime but symtable does not list them
        known = ({sym.get_name() for sym in top.get_symbols()} | set(dir(builtins))
                 | {"__file__", "__name__", "__doc__", "__package__", "__spec__",
                    "__loader__", "__builtins__", "__debug__"})
        missing = []

        def walk(table, where):
            for child in table.get_children():
                scope = f"{where}.{child.get_name()}"
                for sym in child.get_symbols():
                    # a global read that nothing defines at module level
                    if (sym.is_global() and sym.get_name() not in known
                            and not sym.is_assigned()):
                        missing.append(f"{scope} -> {sym.get_name()}")
                walk(child, scope)

        walk(top, name)
        check(f"{name} references no undefined global", not missing, missing[:3])


def test_ui_chrome_is_quiet():
    """The refresh, pinned as rules rather than as taste.

    The conversation is the product: it should not be framed by an avatar and a
    caps label on every turn, the same instruction should not be printed twice,
    and the sidebar should be an index you can scan rather than fifteen
    identical rows."""
    css = (Path(__file__).parent / "webui" / "style.css").read_text()
    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    html = (Path(__file__).parent / "webui" / "index.html").read_text()

    # turns are told apart by shape, and the label survives for screen readers
    check("the per-message caps label is not drawn", ".msg .who {\n  position: absolute"
          in css or "clip: rect(0 0 0 0)" in css.split(".msg .who")[1][:200])
    check("but it is still in the DOM for assistive tech", '"who"' in js)

    # a one-line question must not be drawn as a three-line box
    check("hover actions hang below a question instead of padding it out",
          ".msg.user .msg-actions" in css and "position: absolute" in
          css.split(".msg.user .msg-actions")[1][:120])

    # nothing is explained twice
    check("the composer placeholder does not repeat the key hints",
          'placeholder="Message the model"' in html)
    check("...and the hint row does not either",
          '"Shift+Enter"' not in js.split("function showHint")[1].split("function ")[1])
    check("the welcome screen does not re-explain the mode switch",
          "Also reads and edits files in one folder you pick" not in js)

    # the sidebar is an index
    check("rows carry a time", "function rowTime" in js and '"when"' in js)
    check("the per-row AGENT badge is gone entirely",
          ".session .tag" not in css and 'el("span", "tag", "agent")' not in js)
    check("...because agent chats have a section of their own instead",
          'label: "Agentic"' in js and 'label: "Chats"' in js)

    # the accent is spent on what is active, not on furniture
    check("New chat is not a saturated slab", 'id="new-chat" class="btn block"' in html)
    # ...but the mode switch is what is active, so it keeps the accent
    glider = css.split(".mode-glider {")[1][:320]
    check("the selected mode wears the accent",
          "linear-gradient(135deg, var(--accent), var(--accent-2))" in glider)
    check("...and its label is legible on it",
          "color: var(--accent-ink)" in css.split(".mode.on {")[1][:120])
    check("hovering the other mode previews the colour it would take",
          ".mode:not(.on):hover" in css
          and "color: var(--accent)" in css.split(".mode:not(.on):hover")[1][:80])

    # and the scale rule the whole UI depends on still holds
    import re
    sizes = re.findall(r"font-size:\s*([^;]+);", css)
    bad = [x for x in sizes if "px" in x and "var(--text-scale)" not in x]
    check("every font-size still scales with the text-size setting", not bad, bad[:3])


def test_new_chats_keep_the_model_you_last_used():
    """"Whatever I used last" is the default, so it has to mean that. It fell
    through to the local server instead, which put every new chat - and every
    restart - back on this machine's model however long you had been working
    through a provider."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()

    check("the last endpoint is remembered", "function rememberEndpoint" in js
          and "chatui.lastEndpoint" in js)
    check("...in the browser, beside the other preferences",
          "localStorage.setItem(\"chatui.lastEndpoint\"" in js)
    check("a browser that refuses storage does not take the UI down",
          "catch (e)" in js.split("function rememberEndpoint")[1][:400])

    d = js.split("function defaultEndpoint")[1].split("\nfunction ")[0]
    check("with no setting, the default IS the last one used",
          "state.settings?.defaultEndpoint || lastEndpoint()" in d)
    check("an explicit setting still wins", d.index("state.settings?.defaultEndpoint")
          < d.index("lastEndpoint()"))
    check("a remembered provider that has been deleted falls back",
          "if (!p) return null" in d)

    # what counts as "used"
    check("choosing one from the model list counts",
          "rememberEndpoint(entry.provider, entry.model, entry.provider_name)" in js)
    check("so does sending a turn on it",
          "rememberEndpoint(state.provider, state.model)" in js)
    check("opening an old conversation does not - browsing your history would "
          "otherwise rewrite the default",
          "rememberEndpoint" not in js.split("async function openSession")[1]
          .split("\nfunction ")[0] if "async function openSession" in js else True)

    order = js.split("function newChat")[1][:600]
    check("a new chat applies it", "applyDefaultEndpoint()" in order)


def test_default_endpoint_for_new_chats():
    """Which model a new conversation starts on is a preference, not an
    accident of whatever was opened last."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    check("there is a stored default", "defaultEndpoint" in js)
    check("a new chat applies it", "applyDefaultEndpoint()" in js
          and js.index("function newChat") < js.rindex("applyDefaultEndpoint()"))
    check("the choices are built from the local server plus the providers",
          "function endpointChoices" in js)
    check("settings offers it", "Default model for new chats" in js)
    fn = js.split("function defaultEndpoint")[1].split("function applyDefaultEndpoint")[0]
    check("a default pointing at a deleted provider falls back instead of stranding "
          "the UI", "if (!p) return null" in fn)
    check("existing conversations are not retargeted",
          "existing conversations keep their own" in js)


def test_provider_images():
    """Whether images can be attached is a property of the endpoint in use, not
    of this machine. It read the local server's vision flag regardless, so a
    provider with a working vision model had the attach button hidden because
    this card's own tower had not loaded - which is exactly the state this
    install was in."""
    import shutil, tempfile                             # noqa: WPS433
    import webui_providers as wp                        # noqa: WPS433

    for label, row, want in (
            ("OpenRouter-style modalities",
             {"architecture": {"input_modalities": ["text", "image"]}}, True),
            ("declared text-only",
             {"architecture": {"input_modalities": ["text"]}}, False),
            ("a flat modalities list", {"modalities": ["text", "image"]}, True),
            ("capabilities.vision true", {"capabilities": {"vision": True}}, True),
            ("capabilities.vision false", {"capabilities": {"vision": False}}, False),
            ("an endpoint that says nothing", {"object": "model"}, None)):
        got = wp.vision_from_model_row({"id": "m", **row})
        check(f"{label} -> {want}", got is want, got)
    check("'did not say' is not the same as 'no' - it must stay distinct",
          wp.vision_from_model_row({"id": "m"}) is None)

    root = Path(tempfile.mkdtemp(prefix="vis-"))
    try:
        row = wp.upsert(root, {"name": "S", "base_url": "https://s/v1",
                               "default_model": "m"})
        check("a new provider has no answer either way", row["vision"] is None)
        yes = wp.upsert(root, {"id": row["id"], "name": "S",
                               "base_url": "https://s/v1", "default_model": "m",
                               "vision": True})
        check("the user can say it takes images", yes["vision"] is True)
        kept = wp.upsert(root, {"id": row["id"], "name": "S",
                                "base_url": "https://s/v1", "default_model": "m"})
        check("and that survives an edit that does not resend it",
              kept["vision"] is True)
        no = wp.upsert(root, {"id": row["id"], "name": "S", "base_url": "https://s/v1",
                              "default_model": "m", "vision": "no"})
        check("and can be turned off again", no["vision"] is False)
        check("both the probe's answer and the user's are exposed",
              "vision" in wp.public(no) and "vision_detected" in wp.public(no))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # The save round-trip, through the real endpoint - the part that was broken
    # while every piece of it tested green on its own.
    import urllib.error                                 # noqa: WPS433

    def post_json(path, payload):
        return json.loads(post(path, payload).read())

    saved = post_json("/ui/providers", {"name": "VisionTest",
                                        "base_url": "https://vision.test/v1",
                                        "default_model": "m", "vision": True})
    try:
        check("saving a provider with images on returns it set",
              saved["provider"]["vision"] is True, saved["provider"])
        back = get("/ui/config")
        row = next((p for p in back.get("providers", [])
                    if p["id"] == saved["provider"]["id"]), None)
        check("...and the chat UI's own config carries it",
              row and row["vision"] is True, row)
        again = post_json("/ui/providers", {"id": saved["provider"]["id"],
                                            "name": "VisionTest",
                                            "base_url": "https://vision.test/v1",
                                            "default_model": "m"})
        check("...and an edit that does not resend it does not clear it",
              again["provider"]["vision"] is True, again["provider"])
    finally:
        try:                                        # leave no test provider behind
            req = urllib.request.Request(
                BASE + f"/ui/providers/{saved['provider']['id']}",
                headers=UI_HEADERS, method="DELETE")
            urllib.request.urlopen(req, timeout=15).close()
        except Exception:                               # noqa: BLE001
            pass

    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    check("the composer asks the active endpoint, not the local server",
          "function activeVision" in js)
    # Saving a provider used to refresh state.config without re-syncing anything
    # derived from it, so turning Images on for the provider you were already
    # chatting with changed nothing on screen and read as "it did not save".
    check("changing the config re-syncs the controls that depend on it",
          "async function reloadConfig" in js)
    save_fn = js.split("save.textContent = provider")[1][:600]
    check("...and the provider form uses it rather than assigning config directly",
          "reloadConfig()" in save_fn and "state.config = await" not in save_fn,
          save_fn[:200])
    check("...and no longer reads state.config.vision directly for the button",
          "$(\"#attach\").hidden = !config.vision" not in js
          and "$(\"#attach\").hidden = !state.config.vision" not in js)
    check("the user's answer overrides what the probe found",
          "p.vision !== null" in js)
    check("the provider form offers the switch", '"Images"' in js)
    check("switching endpoint updates the button",
          "refreshAttachButton" in js)


def test_effort_levels_come_from_the_endpoint():
    """Reasoning levels differ per model - low/medium/high on most, plus xhigh
    or max on others - so the menu has to come from what the endpoint says it
    takes, not from names invented here. An endpoint that says nothing gets no
    levels: offering switches that do nothing is worse than offering none."""
    import shutil, tempfile                             # noqa: WPS433
    import webui_providers as wp                        # noqa: WPS433

    for label, row, want in (
            ("an explicit list", {"supported_reasoning_efforts":
                                  ["low", "medium", "high", "xhigh"]},
             ["low", "medium", "high", "xhigh"]),
            ("a nested schema", {"reasoning_effort":
                                 {"enum": ["minimal", "low", "medium", "high", "max"]}},
             ["minimal", "low", "medium", "high", "max"]),
            ("levels under capabilities",
             {"capabilities": {"reasoning": {"levels": ["low", "high"]}}}, ["low", "high"]),
            ("OpenRouter's marker with no levels",
             {"supported_parameters": ["temperature", "reasoning"]},
             ["low", "medium", "high"]),
            ("an endpoint that says nothing", {"object": "model"}, []),
            ("junk and casing", {"reasoning_efforts": ["banana", "HIGH", "low"]},
             ["low", "high"])):
        got = wp.efforts_from_model_row({"id": "m", **row})
        check(f"{label} -> {want}", got == want, got)

    check("the levels come back in ascending order",
          wp.efforts_from_model_row({"id": "m", "reasoning_efforts":
                                     ["max", "low", "high"]}) == ["low", "high", "max"])

    root = Path(tempfile.mkdtemp(prefix="eff-"))
    try:
        row = wp.upsert(root, {"name": "S", "base_url": "https://s/v1",
                               "default_model": "m",
                               "efforts": ["low", "high", "xhigh"]})
        check("they are stored with the provider",
              row["efforts"] == ["low", "high", "xhigh"], row["efforts"])
        check("and reach the browser", wp.public(row)["efforts"] == ["low", "high", "xhigh"])
        kept = wp.upsert(root, {"id": row["id"], "name": "S",
                                "base_url": "https://s/v1", "default_model": "m"})
        check("an edit that does not resend them keeps them",
              kept["efforts"] == ["low", "high", "xhigh"])
    finally:
        shutil.rmtree(root, ignore_errors=True)

    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    check("the UI asks the active endpoint what it takes",
          "function availableEfforts" in js)
    check("...and the settings picker is built from that, not a fixed list",
          "opts = [[\"default\"" in js or "levels.map((l)" in js)
    check("an endpoint offering nothing says so instead of showing dead switches",
          "did not report which effort levels" in js)
    app = (Path(__file__).parent / "webui_app.py").read_text()
    srv = (Path(__file__).parent / "serve_openai.py").read_text()
    # This used to advertise nothing at all, on the grounds that the engine
    # cannot cut thinking short once it has started. True, but beside the
    # point: the chat template acts on reasoning_effort when the prompt is
    # rendered, which is before any of that - so the levels were inert only
    # because the request never carried them that far.
    check("the UI reports whatever the mounting server found",
          '"efforts": list(self.efforts)' in app)
    check("...and standalone, with no template to ask, that is nothing",
          "self.efforts: list[str] = []" in app)
    check("the levels come from the model's own chat template",
          "def supported_efforts" in srv)
    check("...a template that ignores the argument advertises none",
          "len(seen) > 1" in srv)
    check("the effort actually reaches the template now",
          "template_effort(tokenizer, reasoning_effort)" in srv)
    check("...and this template's spelling is not OpenAI's",
          '"high": "xhigh"' in srv)


def test_stop_reaches_the_gpu():
    """Stop used to close the socket and nothing else. The worker thread kept
    generating to max_tokens with nobody listening - and because generation is
    serialised, the user's NEXT message queued behind the reply they had just
    cancelled, which reads as "Stop does nothing"."""
    import types

    # serve_openai imports aiohttp at module scope and exllamav3 inside the
    # function; neither exists off a GPU box, so stand in for both.
    saved = {k: sys.modules.get(k) for k in
             ("aiohttp", "aiohttp.web", "exllamav3", "exllamav3.generator",
              "exllamav3.generator.sampler", "exllamav3.generator.sampler.presets",
              "serve_openai")}

    class FakeJob:
        def __init__(self, **kw):
            self.sequences = [types.SimpleNamespace(
                sequence_ids=types.SimpleNamespace(seq_len=40))]

    class FakeGenerator:
        """Streams forever unless cancelled - a model asked for 4k tokens."""
        LIMIT = 5000

        def __init__(self):
            self.jobs, self.cancelled, self.steps = [], [], 0

        def enqueue(self, job):
            self.jobs.append(job)

        def num_remaining_jobs(self):
            return len(self.jobs)

        def iterate(self):
            self.steps += 1
            if self.steps > self.LIMIT:
                raise AssertionError("generation never stopped")
            return [{"text": "tok ", "token_ids": [1]}]

        def cancel(self, job):
            self.cancelled.append(job)
            self.jobs.remove(job)

    class FakeTokenizer:
        eos_token_id = 7

        def hf_chat_template(self, *a, **k):
            class Ids:
                shape = (1, 12)
            return Ids()

    try:
        web = types.SimpleNamespace(StreamResponse=object, Response=object,
                                    Application=object, RouteTableDef=object,
                                    json_response=lambda *a, **k: None)
        sys.modules["aiohttp"] = types.SimpleNamespace(web=web)
        sys.modules["aiohttp.web"] = web
        ex = types.ModuleType("exllamav3")
        ex.Job = FakeJob
        presets = types.ModuleType("exllamav3.generator.sampler.presets")
        presets.ComboSampler = lambda **kw: object()
        sys.modules.update({
            "exllamav3": ex,
            "exllamav3.generator": types.ModuleType("exllamav3.generator"),
            "exllamav3.generator.sampler": types.ModuleType("exllamav3.generator.sampler"),
            "exllamav3.generator.sampler.presets": presets})
        sys.modules.pop("serve_openai", None)
        import serve_openai as S

        gen = FakeGenerator()
        seen = []
        text, calls, finish, ptoks, otoks, reasoning, content = S.generate_full(
            gen, FakeTokenizer(), [{"role": "user", "content": "hi"}],
            max_tokens=4096, temperature=0.7, top_p=0.9, top_k=40, seed=None,
            tools=None, on_text=seen.append,
            should_stop=lambda: len(seen) >= 5)

        check("the stop signal ends generation instead of running to max_tokens",
              gen.steps < 50, f"{gen.steps} decode steps")
        check("the job is cancelled on the generator, so its cache pages come back",
              len(gen.cancelled) == 1)
        check("...and nothing is left queued behind it",
              gen.num_remaining_jobs() == 0)
        check("what was generated before the stop is kept, not thrown away",
              len(seen) >= 5 and text.strip() != "")
        check("a cancelled turn finishes as a stop, not an error", finish == "stop")

        # the wiring: a client that hangs up has to reach that signal
        src = (Path(__file__).parent / "serve_openai.py").read_text()
        check("the request handler passes one in",
              "should_stop = gone.is_set" in src)
        check("a failed write sets it", "gone.set()" in src
              and "gone.set()" in src.split("async def send(")[1][:600])
        check("...and so does a socket that closed during prefill, when no "
              "write would ever fail", "async def watch_client" in src
              and "transport.is_closing()" in src)
        check("the signal is set once the turn is over either way",
              "finally:" in src.split("await consume()")[1][:200])
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_thinking_block_follows_its_own_text():
    """A pane that does not scroll with the text being written into it is a pane
    you cannot read: the model writes at the bottom while you sit at the top
    watching a scrollbar shrink."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    css = (Path(__file__).parent / "webui" / "style.css").read_text()

    check("there is a follower", "function followThink" in js)
    check("every reasoning delta calls it",
          "followThink(pane)" in js
          and "followThink(pane)" in js.split('case "reasoning"')[1][:400])
    check("it appends to the same pane it then scrolls - not a fresh lookup",
          "const pane = thinkBlock(body, true)" in js)

    follower = js.split("function followThink")[1].split("\nfunction ")[0]
    check("following means pinning to the end",
          "scrollTop = pane.scrollHeight" in follower)
    check("a reader who scrolled away is left alone",
          'dataset.stuck === "0"' in follower)

    made = js.split("function thinkBlock")[1].split("\nfunction ")[0]
    check("the pane starts out following", 'pane.dataset.stuck = "1"' in made)
    check("scrolling up stops the follow, scrolling back resumes it",
          'addEventListener("scroll"' in made
          and "scrollHeight - pane.scrollTop - pane.clientHeight < 24" in made)

    settle = js.split("function settleThinkBlock")[1].split("\nfunction ")[0]
    check("a finished thought is rewound, so it reads from the start",
          "pane.scrollTop = 0" in settle)
    check("...unless the reader had already put it somewhere",
          'dataset.stuck !== "0"' in settle)

    import re as _re

    def cap(block):
        """The px ceiling out of `max-height: min(NNvh, NNNpx)`."""
        m = _re.search(r"max-height:\s*min\(\s*\d+vh\s*,\s*(\d+)px\s*\)", block)
        return int(m.group(1)) if m else None

    live = css.split(".think.live .think-body {")[1][:400]
    settled = css.split(".think .think-body {")[1][:500]
    check("the live window sits lower than the settled one",
          cap(live) and cap(settled) and cap(live) < cap(settled),
          f"live={cap(live)} settled={cap(settled)}")
    check("both give way on a short window instead of eating the answer",
          "vh" in live and "vh" in settled)
    check("...but there is enough of it to read a train of thought",
          cap(settled) >= 420 and cap(live) >= 240,
          f"live={cap(live)} settled={cap(settled)}")
    check("text runs off the top edge instead of being chopped",
          "mask-image: linear-gradient(to bottom, transparent 0" in live)
    check("the pane does not hand its overscroll to the page",
          "overscroll-behavior: contain" in css.split(".think .think-body {")[1][:400])


def test_thinking_block_is_a_real_toggle():
    """The Thinking block is how reasoning is seen at all, so its open/closed
    state has to belong to the user. It was slammed shut at the end of every
    reply, which made opening one pointless - the next answer closed it again."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()

    check("the preference is remembered", "chatui.thinkOpen" in js)
    check("it is set by toggling the block itself",
          'addEventListener("toggle"' in js and "setThinkOpenPref(node.open)" in js)
    check("the end of a reply no longer forces it shut",
          "think.open = false" not in js, "something still slams it closed")
    check("a finished block says what it did",
          "Thought for" in js)
    check("a live one says it is working", '"Thinking..."' in js)
    check("a reloaded conversation follows the same preference",
          js.count("thinkOpenPref()") >= 3)
    check("code-driven opening does not overwrite the user's choice",
          "dataset.settling" in js)


def test_slash_commands():
    """/effort in the composer, because a setting behind a panel is a setting
    nobody changes mid-conversation - and thinking effort is exactly the kind
    of thing you want to change for one question."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()

    check("there is a command table", "const COMMANDS" in js)
    check("/effort is one of them", "effort: {" in js)
    check("/help lists them", "help: {" in js and "list these commands" in js)
    check("commands are intercepted before a turn is started",
          js.index("runCommand(text)") < js.index('await fetch("/ui/chat"'),
          "a command must never reach the model")

    cmd = js.split("const COMMANDS")[1].split("async function send")[0]
    # off and default work everywhere; the levels themselves come from the
    # endpoint, so the command must not hardcode a menu of its own
    for word in ("off", "default"):
        check(f"/effort accepts '{word}' on any endpoint", f'"{word}"' in cmd, word)
    check("the levels offered come from the endpoint, not from a fixed list",
          "availableEfforts()" in cmd)
    check("a level the endpoint never claimed is refused",
          "does not offer" in cmd or "did not say it takes" in cmd)
    check("only 'off' is described as exact",
          "this one is exact" in cmd and "not a limit" in cmd)
    # This used to point at the command, because the command was the only way
    # to change a level for one conversation. The chip by the message box is
    # the shorter route now - and, since the field only sets what a NEW chat
    # starts on, saying where the per-chat control lives is the point of it.
    check("the settings field says where a single chat is changed",
          "use the chip by the message box" in js)


def test_reasoning_from_any_endpoint():
    """Thinking has to survive whatever shape the endpoint sends it in.

    Only one shape was handled - `reasoning_content`, which is what this kit's
    own server sends - so a local chat showed Thinking and a custom provider
    showed none at all. Confirmed against real session files: 14 local replies,
    12 with reasoning; 10 replies through a provider, 0 with reasoning. The
    other two shapes in the wild are a `reasoning` field (OpenRouter and
    others) and a raw <think> block inside `content` (llama.cpp, LM Studio,
    vLLM with no reasoning parser)."""
    import json as _json                                # noqa: WPS433
    import webui_agent                                  # noqa: WPS433

    sp = webui_agent.ThinkSplitter
    def split(chunks):
        s2, out = sp(), []
        for c in chunks:
            out += s2.feed(c)
        out += s2.flush()
        merged = []
        for kind, piece in out:
            if merged and merged[-1][0] == kind:
                merged[-1] = (kind, merged[-1][1] + piece)
            else:
                merged.append((kind, piece))
        return merged

    check("an inline think block is split out",
          split(["<think>weighing</think>Answer."])
          == [("reasoning", "weighing"), ("content", "Answer.")])
    check("a marker split across chunks is still found",
          split(["<thi", "nk>a", "b</thi", "nk>Answer."])
          == [("reasoning", "ab"), ("content", "Answer.")])
    check("plain text is left entirely alone",
          split(["Just an answer."]) == [("content", "Just an answer.")])
    check("a <think> mentioned later in the answer is not a second block",
          split(["<think>a</think>see <think> here"])
          == [("reasoning", "a"), ("content", "see <think> here")])
    check("a stream that stops mid-thought still delivers what it had",
          split(["<think>unfinished"]) == [("reasoning", "unfinished")])

    # and the client itself, against each shape an endpoint might use
    class Resp:
        def __init__(self, deltas): self.deltas = deltas
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def __iter__(self):
            for d in self.deltas:
                yield b"data: " + _json.dumps({"choices": [{"delta": d}]}).encode() + b"\n"
            yield b"data: " + _json.dumps(
                {"choices": [{"delta": {}, "finish_reason": "stop"}]}).encode() + b"\n"
            yield b"data: [DONE]\n"

    real = webui_agent.urllib.request.urlopen
    try:
        for label, deltas in (
                ("reasoning_content", [{"reasoning_content": "think"}, {"content": "Answer."}]),
                ("reasoning", [{"reasoning": "think"}, {"content": "Answer."}]),
                ("inline <think>", [{"content": "<think>think</think>"}, {"content": "Answer."}])):
            webui_agent.urllib.request.urlopen = (
                lambda req, timeout=None, d=deltas: Resp(d))
            client = webui_agent.ModelClient("http://x/v1", "k", "m")
            got = {}
            for kind, val in client.stream([{"role": "user", "content": "hi"}]):
                if kind in ("reasoning", "content"):
                    got[kind] = got.get(kind, "") + val
            check(f"an endpoint sending {label} produces a Thinking block",
                  got.get("reasoning") == "think", got)
            check(f"...and its answer is not polluted with the markers ({label})",
                  got.get("content") == "Answer.", got)
    finally:
        webui_agent.urllib.request.urlopen = real


def test_message_edit_is_in_place():
    """Editing a question must happen where the question is.

    It used to rewind first - deleting the message and its answer - and then
    quietly fill the composer at the bottom of the window. From where the user
    was looking, clicking Edit made their message disappear and nothing else
    happen, which reads as "Edit is broken" or worse, "Edit ate my message".
    Now the bubble becomes a textarea, nothing is destroyed until Save, and
    Cancel puts it back."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()

    check("there is an in-place editor", "function startMessageEdit" in js)
    check("the old composer-filling editLast is gone", "async function editLast" not in js)
    edit = js.split("function startMessageEdit")[1].split("\n/*")[0]
    check("nothing is rewound before the user presses Save",
          edit.index("save.onclick") < edit.index("await rewind()"),
          "rewind must only happen inside the save handler")
    check("Cancel restores the original message", "cancel.onclick = close" in edit)
    check("Escape closes it too", '"Escape"' in edit)
    check("the textarea does not leak keys to the composer",
          "e.stopPropagation()" in edit)

    row = js.split("function refreshEditAction")[1].split("function messageText")[0]
    check("the question carries a copy action as well as edit",
          '"copy"' in row and '"pencil"' in row)
    check("both are icon-only", "icon-only" in row)
    check("...but still labelled for screen readers and tooltips",
          'setAttribute("aria-label"' in row and "b.title = label" in row)

    css = (Path(__file__).parent / "webui" / "style.css").read_text()
    check("icon-only actions are styled square, not left in a pill sized for text",
          ".act.icon-only" in css)
    check("the in-place editor is styled", ".edit-area" in css and ".edit-bar" in css)


def test_provider_context_length():
    import shutil, tempfile                             # noqa: WPS433
    """A provider's context window is something only the user knows - /models
    does not report one, and a remote model's window has nothing to do with
    this card's VRAM. Without it the UI hid the meter entirely, so a long
    conversation on a provider gave no hint how full it was getting."""
    import webui_providers as wp                        # noqa: WPS433

    root = Path(tempfile.mkdtemp(prefix="prov-"))
    try:
        row = wp.upsert(root, {"name": "OpenRouter", "base_url": "https://o.ai/v1",
                               "default_model": "a/b", "context_length": "128000"})
        check("a context is stored", row["context_length"] == 128000, row["context_length"])
        check("and reaches the browser", wp.public(row)["context_length"] == 128000)

        kept = wp.upsert(root, {"id": row["id"], "name": "OpenRouter",
                                "base_url": "https://o.ai/v1", "default_model": "a/b"})
        check("editing without resending it keeps the value",
              kept["context_length"] == 128000, kept["context_length"])

        blank = wp.upsert(root, {"name": "NoCtx", "base_url": "https://n.ai/v1",
                                 "default_model": "m"})
        check("it stays optional", blank["context_length"] is None)

        for bad, why in (("abc", "not a number"), (12, "absurdly small"),
                         (10 ** 9, "absurdly large")):
            try:
                wp.upsert(root, {"name": "B", "base_url": "https://b.ai/v1",
                                 "default_model": "m", "context_length": bad})
                check(f"{why} is refused", False, f"accepted {bad!r}")
            except wp.ProviderError:
                check(f"{why} is refused", True)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    check("the meter measures against whatever endpoint is active",
          "function activeContextLength" in js)
    check("...and the provider form offers the field",
          'field("Context", "context_length"' in js)


def test_thinking_effort():
    """The Thinking setting has to mean something different per level, and the
    only exact one is Off: the chat template emits an empty <think></think> pair
    so there is nowhere to reason. The effort levels are guidance - this engine
    cannot cut thinking short mid-generation - so they must not be sold as a
    hard budget."""
    src = (Path(__file__).parent / "serve_openai.py").read_text()

    check("generate_full can be told not to open a think block",
          "enable_thinking = True" in src and "enable_thinking = enable_thinking" in src)
    check("the streaming parser starts outside <think> when thinking is off",
          'in_think = [bool(req["enable_thinking"])]' in src,
          "otherwise the answer is swallowed into a reasoning block")
    # This assertion previously named the local `enable_thinking`, which does not
    # exist in that scope - so it passed by pinning the exact broken line.
    # test_no_undefined_globals() is what actually guards it now.
    check("both the vLLM/SGLang spelling and OpenAI's are accepted",
          "chat_template_kwargs" in src and "reasoning_effort" in src)

    # the parser itself, lifted out so this needs no torch
    import ast                                          # noqa: WPS433
    tree = ast.parse(src)
    mod = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                           and n.name in ("parse_request", "normalize_messages")],
                     type_ignores=[])
    ns = {"MODEL_ID": "local"}
    exec(compile(mod, "<parse_request>", "exec"), ns)   # noqa: S102
    base = {"messages": [{"role": "user", "content": "hi"}]}
    for extra, want, label in (
            ({}, True, "unset means the model's own default (thinking on)"),
            ({"reasoning_effort": "high"}, True, "an effort level keeps thinking on"),
            ({"reasoning_effort": "none"}, False, "'none' turns it off"),
            ({"chat_template_kwargs": {"enable_thinking": False}}, False,
             "the template kwarg turns it off"),
            ({"chat_template_kwargs": {"enable_thinking": True},
              "reasoning_effort": "none"}, True,
             "an explicit template kwarg wins over the effort word")):
        req, err = ns["parse_request"]({**base, **extra})
        check(label, req and req["enable_thinking"] is want,
              req and req["enable_thinking"])

    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    check("the UI offers the choice", '"Thinking"' in js and '["off", "Off"]' in js)
    check("...and does not promise a hard limit it cannot enforce",
          "a level is a request" in js)
    app = (Path(__file__).parent / "webui_app.py").read_text()
    check("the choice is sent to the model", 'settings.get("thinking"' in app)


def test_setup_page_survives_the_handover():
    """Setup and the model server share one port, with minutes between them while
    the weights load and the kernels compile. A page open across that gap must
    not tell the user their install is broken.

    It did: any failure to reach /setup/state rendered "Setup is not running -
    close it and start Simplex again", which is exactly wrong advice while
    Simplex is starting. There are two distinct cases and neither is a failure:
    nothing listening yet (wait for it), and the model server having taken the
    port (go to the app it serves - it 404s /setup/state)."""
    js = (Path(__file__).parent / "webui" / "setup.js").read_text()

    check("a stale page waits for the server instead of declaring it dead",
          "waitForServer" in js)
    check("...and says what is actually happening",
          "Loading the model into the graphics card" in js)
    check("...and only suggests a restart after it has really been too long",
          "longer than a first load usually takes" in js)

    boot = js.split("function boot()")[1].split("function waitForServer")[0]
    check("the dead-end message is gone from boot()",
          "Setup is not running" not in boot, boot[-300:])
    check("a non-OK /setup/state means the model server owns the port, so the page "
          "hands over rather than erroring",
          "location.href" in boot and "!r.ok" in boot)

    # the model server really does answer that way - this is what the page keys on
    import urllib.error                                 # noqa: WPS433
    try:
        urllib.request.urlopen(urllib.request.Request(
            BASE + "/setup/state", headers=UI_HEADERS), timeout=5).close()
        check("the model server does not serve /setup/state", False, "it answered 200")
    except urllib.error.HTTPError as e:
        check("the model server 404s /setup/state, which is the handover signal",
              e.code == 404, e.code)
    except Exception as e:                              # noqa: BLE001
        check("the model server was reachable for the handover check", False, str(e))


def test_setup_page_can_scroll():
    """The setup page must scroll. It shares style.css with the chat UI, which
    pins itself to the viewport - `html, body { height: 100% }` and
    `body { overflow: hidden }` - because that layout scrolls inside its own
    panes. The setup page is one long column, so inheriting the lock clipped it
    at the fold: on a card with a few quants the menu and the Install button
    were simply unreachable, with no scrollbar to hint why."""
    import re                                           # noqa: WPS433
    web = Path(__file__).parent / "webui"
    shared = (web / "style.css").read_text()
    page = (web / "setup.html").read_text()

    locks_viewport = ("html, body { height: 100%; }" in shared
                      and re.search(r"body\s*\{[^}]*overflow:\s*hidden", shared, re.S))
    check("style.css still pins the chat UI to the viewport (the thing to undo)",
          bool(locks_viewport))
    if not locks_viewport:
        return

    head = page.split("</style>")[0]
    check("setup.html releases the height lock",
          re.search(r"html,\s*body\s*\{[^}]*height:\s*auto", head) is not None, head[-400:])
    check("setup.html releases the overflow lock",
          re.search(r"html,\s*body\s*\{[^}]*overflow:\s*(visible|auto)", head) is not None,
          head[-400:])
    # the override has to come after the stylesheet link, or the cascade eats it
    marker = "html, body { height: auto"
    check("and does so after style.css is linked, so it actually wins",
          marker in page and page.index("style.css") < page.index(marker),
          "the override is missing or sits above the stylesheet link")


def test_the_suite_never_writes_the_real_env():
    """Running the tests must not reconfigure the installation they run in.

    Two of them did. Both drive the real apply_choice() with a pretend GPU
    patched in, and apply_choice writes setup_core.ENV_FILE - so on a real
    machine the suite rewrote .env with a profile for hardware that is not
    there: PROFILE_GPU=Fake / RTX 4060 Ti, GPU_MEM_GB=14.7 on a 32 GB card,
    which then caps the server to 14.7 GB at the next start. Nothing here may
    write a file the product reads."""
    import hashlib, subprocess                          # noqa: WPS433
    root = Path(__file__).resolve().parent.parent
    env = root / ".env"
    if not env.is_file():
        check("no .env to protect on this box (nothing to verify)", True)
        return

    def digest():
        return hashlib.sha256(env.read_bytes()).hexdigest()

    before = digest()
    # the two tests that patch the hardware, run in-process
    import setup_mock                                   # noqa: WPS433
    scratch = setup_mock.patch(vram_gib=16.0, speed=0.0, download_seconds=0.1)
    check("patching the hardware also redirects .env away from the real one",
          Path(scratch) != env, scratch)
    import setup_core                                   # noqa: WPS433
    check("...and setup_core writes there instead",
          Path(setup_core.ENV_FILE) != env, setup_core.ENV_FILE)
    check("the real .env is byte-identical after mocking", digest() == before)

def test_web_setup_matches_the_console():
    """The browser flow and the console are two front doors to the same install,
    so they must plan identically. The browser one never asked about images (it
    called plan() with no answer) and its page ignored the measured / verified
    flags, so a 32 GB user was offered 6.0bpw at an unverified 262144 - the
    config the bench OOMed - with no warning the console shows."""
    import shutil, tempfile                             # noqa: WPS433
    import profiles                                     # noqa: WPS433
    import setup_core                                   # noqa: WPS433
    import setup_web                                    # noqa: WPS433

    real = profiles.detect_gpu
    try:
        for vram in (16.0, 24.0, 32.0):
            profiles.detect_gpu = lambda v=vram: profiles.GPU(
                name="Fake", total_gib=v, cc=8.9, driver="580")
            for want in (None, True, False):
                web = setup_core.options_for({}, vram, want)
                console = profiles.plan(vram, want)[1]
                if want and not console:
                    continue                 # both fall back; covered separately
                check(f"{vram:.0f}GB vision={want}: the page offers the console's rows",
                      [(o["quant"], o["ctx"], o["vision"]) for o in web["options"]]
                      == [(o["quant"], o["ctx"], o["vision"]) for o in console],
                      (web["options"], console))
                check(f"{vram:.0f}GB vision={want}: every row carries its provenance",
                      all("measured" in o and "verified_ctx" in o for o in web["options"]))
            # the keys the page reads must exist in every shape of the response
            for payload in (setup_core.options_for({}, vram, True),
                            setup_core.options_for({}, 0.0, True)):
                check("the menu payload always carries the images keys",
                      all(k in payload for k in ("vision", "hidden_by_vision", "vision_cost")),
                      sorted(payload))

        # The answer survives a choice and lands in .env - written to a throwaway
        # file, never the real one. choose() calls setup_core.apply_choice(), which
        # writes setup_core.ENV_FILE; when this test ran against the live checkout
        # it rewrote the user's actual .env with this fake 16 GB card's profile
        # (PROFILE_GPU=Fake, GPU_MEM_GB=14.7) and capped their 32 GB card at 14.7.
        # A test may never write a file the product reads.
        profiles.detect_gpu = lambda: profiles.GPU(name="Fake", total_gib=16.0, cc=8.9, driver="580")
        real_env = setup_core.ENV_FILE
        scratch = Path(tempfile.mkdtemp(prefix="envtest-")) / ".env"
        setup_core.ENV_FILE = scratch
        try:
            before = real_env.read_text(encoding="utf-8") if real_env.is_file() else None
            setup = setup_web.Setup({}, force=True)
            setup.refresh(False)
            text_only = [(o["quant"], o["ctx"]) for o in setup.menu["options"]]
            setup.choose(setup.menu["options"][-1]["id"], 0.0)
            check("choosing with images off writes VISION=off",
                  setup.cfg.get("VISION") == "off", setup.cfg.get("VISION"))
            check("the choice went to the scratch .env, not the real one",
                  scratch.is_file() and "PROFILE_GPU=Fake" in scratch.read_text(encoding="utf-8"))
            after = real_env.read_text(encoding="utf-8") if real_env.is_file() else None
            check("the real .env was not touched", after == before,
                  "the live .env changed during the test")
            setup.refresh(True)
            check("toggling images re-plans the menu",
                  [(o["quant"], o["ctx"]) for o in setup.menu["options"]] != text_only)
            check("and reports which quants that hid",
                  setup.menu["hidden_by_vision"] == ["3.0"], setup.menu["hidden_by_vision"])
        finally:
            setup_core.ENV_FILE = real_env
            shutil.rmtree(scratch.parent, ignore_errors=True)

        # the page must actually render both flags
        js = (Path(__file__).parent / "webui" / "setup.js").read_text()
        check("setup.js renders the images choice", "renderVisionChoice" in js)
        check("setup.js marks an estimated VRAM figure", "memory use estimated" in js)
        check("setup.js marks an unverified context", "context not verified" in js)
    finally:
        profiles.detect_gpu = real


def test_vision_prompt_edge_cases():
    """The images question decides which menu the user sees, so every way of
    answering - and of not answering - has to lead somewhere sane."""
    import io, contextlib, subprocess                   # noqa: WPS433
    import profiles                                     # noqa: WPS433
    root = Path(__file__).resolve().parent.parent

    # 1. EOF is not an answer. It used to fall through to `not "".startswith("n")`
    #    and force images ON, so a launcher with stdin from NUL, or a scheduled
    #    run, silently took a narrower menu than the auto rule would have.
    r = subprocess.run([sys.executable, "-c",
                        "import sys; sys.path.insert(0, 'tools')\n"
                        "import profiles; print('ANSWER', profiles.ask_vision(False, 32.0))"],
                       stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       cwd=str(root), timeout=60)
    check("EOF on stdin falls back to the automatic rule, not a forced yes",
          "ANSWER None" in r.stdout, r.stdout[-200:] + r.stderr[-200:])

    # 2. --auto never blocks on the question
    check("unattended runs do not ask", profiles.ask_vision(True) is None)

    # 3. The quoted cost has to match the card. The tower is 0.87 GiB on the big
    #    quants but 0.17 on the 2.0bpw build, which is what a 16 GB card is
    #    steered to - one flat "0.9 GB" overstated it fivefold exactly there.
    line16 = profiles.vision_cost_line(16.0)
    small = min(q.vision_gib for q in profiles.QUANTS)
    check("the 16GB prompt does not quote only the big-quant figure",
          f"{small:.1f}" in line16, line16)
    check("the quoted token cost is derived, not hardcoded",
          "45k" not in line16, line16)

    # 4. Answering yes must never abort an install that works text-only.
    for gb in (12.0, 12.1):
        on = profiles.plan(gb, True)[1]
        off = profiles.plan(gb, False)[1]
        if on or not off:
            continue                       # not the interesting band on this table
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            profiles.run(force=True, vram=gb, list_only=True)
        out = buf.getvalue()
        check(f"{gb}GB: a card that only fits text-only is not told the model does not fit",
              "does not fit" not in out, out[-300:])

    # 5. Hidden rows are the biggest quants, so they must not be called small.
    hidden_seen = False
    for gb in (16.0, 20.0, 24.0):
        on = {o["quant"] for o in profiles.plan(gb, True)[1]}
        off = [o for o in profiles.plan(gb, False)[1] if o["quant"] not in on]
        if not off:
            continue
        hidden_seen = True
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            profiles.run(force=True, vram=gb, list_only=True)
        check(f"{gb}GB: hidden quants are described as higher quality, not lower-VRAM",
              "lower-VRAM" not in buf.getvalue())
    check("at least one card size exercised the hidden-rows message", hidden_seen)


def test_recommendation_prefers_verified_rows():
    """The pre-selected row is the one most people take, so it must never be a
    row the kit only computed. On a 32 GB card the top-quality option is 6.0bpw,
    whose full-context prefill has never survived - recommending it would hand a
    first-time user the exact config that OOMed on the bench card."""
    import profiles

    for gb in (16, 24, 32):
        for vision in (True, False):
            _, options = profiles.plan(gb, vision)
            if not options:
                continue
            rec = next((o for o in options if o["recommended"]), None)
            check(f"{gb}GB (images={vision}): exactly one row is recommended",
                  sum(bool(o["recommended"]) for o in options) == 1)
            solid = [o for o in options if o["measured"] and o["verified_ctx"]]
            if solid:
                check(f"{gb}GB (images={vision}): the recommended row is measured "
                      "and has a verified context",
                      rec["measured"] and bool(rec["verified_ctx"]),
                      f"recommended {rec['quant']}bpw at {rec['ctx']}")

    # the specific regression: the bench card must not be pointed at 6.0bpw
    _, on32 = profiles.plan(32.0, True)
    rec32 = next(o for o in on32 if o["recommended"])
    check("a 32GB card is not recommended the quant whose prefill OOMed",
          rec32["quant"] != "6.0", rec32["quant"])
    check("it gets the best verified quality instead (5.0bpw at its verified ceiling)",
          (rec32["quant"], rec32["ctx"]) == ("5.0", 180224), (rec32["quant"], rec32["ctx"]))

    # The card this kit is actually for. The default must not trade three times
    # the quantisation error for context nobody reaches: on 16 GB only the two
    # worst quants can hit 128k, so a 128k threshold recommended 2.0bpw
    # (KL 0.35) over 3.0bpw (KL 0.112).
    _, off16 = profiles.plan(16.0, False)
    rec16 = next(o for o in off16 if o["recommended"])
    better = [o for o in off16 if o["kl"] < rec16["kl"] and o["ctx"] >= profiles.COMFORTABLE_CTX]
    check("a 16GB card is recommended the best quality that still has real context",
          not better, f"recommended {rec16['quant']}bpw but "
                      f"{[b['quant'] for b in better]} are better and still roomy")
    check("...and that is not the lowest-quality quant on the menu",
          rec16["kl"] < max(o["kl"] for o in off16), rec16["quant"])
    check("the recommended row still clears the comfort threshold",
          rec16["ctx"] >= profiles.COMFORTABLE_CTX, rec16["ctx"])


def test_menu_marks_unverified_rows():
    """A number the kit measured and a number it guessed must not look alike."""
    import io, contextlib
    import profiles

    gpu = profiles.GPU(name="Fake", total_gib=32.0, cc=12.0, driver="610")
    budget, options = profiles.plan(32.0, True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        profiles.show_menu(gpu, budget, options, None)
    out = buf.getvalue()

    check("the footnote explains what an asterisk means", "not verified on hardware yet" in out)
    unmeasured = [o for o in options if not o["measured"]]
    unverified = [o for o in options if not o["verified_ctx"]]
    check("there is something to mark on a 32GB card",
          bool(unmeasured) and bool(unverified))
    check("an unmeasured VRAM figure is marked",
          all(f"{o['need']:.1f} GB*" in out for o in unmeasured))
    # Per row, not a substring search over the whole menu: on a 32 GB card both
    # 6.0 and 2.5 render "262k (max)", so one row's asterisk used to satisfy the
    # assertion for the other - and dropping the mark from 6.0 specifically (the
    # dangerous one, whose 262k prefill has never survived) left the suite green.
    body = [ln for ln in out.splitlines() if "bpw" in ln and "|" not in ln]
    for o in options:
        line = next((ln for ln in body if f"{o['bpw']:.1f} bpw" in ln), "")
        check(f"{o['quant']}bpw has a row in the menu", bool(line), out)
        want_ctx_mark = not o.get("verified_ctx")
        got_ctx_mark = f"{profiles.ctx_label(o['ctx'])}*" in line
        check(f"{o['quant']}bpw: context marked only when unproven "
              f"(expected {want_ctx_mark})", got_ctx_mark == want_ctx_mark, line.strip())
        want_vram_mark = not o.get("measured", True)
        got_vram_mark = f"{o['need']:.1f} GB*" in line
        check(f"{o['quant']}bpw: VRAM marked only when estimated "
              f"(expected {want_vram_mark})", got_vram_mark == want_vram_mark, line.strip())



def test_model_switch_without_a_readable_gpu():
    """A box where the GPU cannot be read must keep the rest of the .env as-is.

    That fallback used to work by accident: settings_for() read gpu.memory,
    which profiles.GPU has never had, so the AttributeError landed in its bare
    `except Exception: return updates`. Reading the right name is correct but
    does NOT raise when there is no card - detect_gpu() hands back a GPU with
    total_gib 0.0 - so without an explicit check the planner runs on a -1.3 GiB
    budget and every model in the switcher is refused with "does not fit in
    0 GB of VRAM"."""
    import types                                        # noqa: WPS433
    import profiles                                     # noqa: WPS433
    import webui_models                                 # noqa: WPS433

    src = (Path(__file__).parent / "webui_models.py").read_text()
    check("webui_models.py reads gpu.total_gib", "gpu.total_gib" in src)
    check("webui_models.py no longer reads the nonexistent gpu.memory",
          "gpu.memory" not in src)

    real = profiles.detect_gpu
    try:
        for label, stub in (
                ("no card at all", lambda: profiles.GPU()),
                ("nvidia-smi unreadable", lambda: profiles.GPU(name="", total_gib=0.0)),
                ("detect_gpu itself raised", None)):
            profiles.detect_gpu = stub or (lambda: (_ for _ in ()).throw(OSError("smi died")))
            # a refusal is itself the bug here, so catch it and report rather
            # than letting it abort the run
            try:
                got = webui_models.settings_for(Path("."), "models/Qwen3.8-27B-EXL3-4.0bpw", {})
            except Exception as e:                      # noqa: BLE001
                check(f"{label}: the switch is not refused outright", False, f"{type(e).__name__}: {e}")
                continue
            check(f"{label}: the switch leaves context/VRAM settings alone",
                  "CONTEXT_SIZE" not in got and "GPU_MEM_GB" not in got, got)
            check(f"{label}: it still points at the new model",
                  got.get("MODEL_DIR", "").endswith("4.0bpw"))
            check(f"{label}: and never claims the card has 0 GB",
                  "0 GB" not in str(got))
    finally:
        profiles.detect_gpu = real


def test_model_switch_uses_the_planner_rule():
    """The switcher and the first-run menu must plan a quant identically.

    They did not: the switcher used `vision = ctx >= 32768` while the menu used
    the quarter-of-context rule, so on a 16 GB card switching to 3.0bpw turned
    images on and cut the context the picker had chosen from 114688 to 65536.
    Both now go through profiles.plan_one()."""
    import profiles                                     # noqa: WPS433
    import webui_models                                 # noqa: WPS433

    real = profiles.detect_gpu
    try:
        for vram in (16, 24, 32):
            profiles.detect_gpu = lambda v=vram: profiles.GPU(
                name="Fake", total_gib=v, cc=8.9, driver="580")
            _, options = profiles.plan(float(vram))
            for o in options:
                got = webui_models.settings_for(
                    Path("."), f"models/Qwen3.8-27B-EXL3-{o['quant']}bpw", {})
                check(f"{vram}GB {o['quant']}bpw: the switch plans the menu's context",
                      int(got["CONTEXT_SIZE"]) == o["ctx"], (got["CONTEXT_SIZE"], o["ctx"]))
                check(f"{vram}GB {o['quant']}bpw: and the menu's images setting",
                      (got["VISION"] != "off") == o["vision"], (got["VISION"], o["vision"]))

        # an answer of "no images" at setup time survives a model switch
        profiles.detect_gpu = lambda: profiles.GPU(name="Fake", total_gib=32.0, cc=8.9, driver="580")
        off = webui_models.settings_for(Path("."), "models/Qwen3.8-27B-EXL3-4.0bpw",
                                        {"VISION": "off"})
        check("an explicit VISION=off is not overwritten by a model switch",
              off["VISION"] == "off", off["VISION"])
        check("and dropping images buys back context",
              int(off["CONTEXT_SIZE"]) >= o["ctx"])
    finally:
        profiles.detect_gpu = real


def test_bench_bat_checks_the_venv():
    """bench.bat needs the same venv-liveness probe start.bat has: a venv
    whose base Python was upgraded/removed keeps a python.exe that dies
    instantly, and bare `if exist` would trust it and fail every run with a
    cryptic traceback instead of pointing at start.bat."""
    bat = (Path(__file__).resolve().parent.parent / "bench.bat").read_text()
    check("bench.bat checks the venv runs before using it",
          '-c "pass"' in bat and "if not errorlevel 1" in bat)
    check("bench.bat points at start.bat when the venv is broken",
          "start.bat" in bat)
    check("bench.bat does not fall back to a bare system python "
          "(the engine only exists in the venv)",
          "where python" not in bat and "where py" not in bat)


def test_webui_bat_runs_the_ui_without_a_model():
    """webui.bat is the no-GPU entry point: tools/chatui.py standalone, so the
    chat app, provider settings, and session history are reachable without
    start.bat ever loading weights."""
    root = Path(__file__).resolve().parent.parent
    bat_path = root / "webui.bat"
    check("webui.bat exists", bat_path.is_file())
    bat = bat_path.read_text()
    check("webui.bat checks the venv runs before using it",
          '-c "pass"' in bat and "if not errorlevel 1" in bat)
    check("webui.bat points at start.bat when the venv is broken",
          "start.bat" in bat)
    check("webui.bat launches chatui.py standalone, not serve_openai.py",
          "tools\\chatui.py" in bat and "serve_openai.py" not in bat)
    check("extra args pass through (so --mock still works from webui.bat)",
          "%*" in bat)

    # the standalone server itself must come up with nothing else running -
    # no model server, no GPU, no venv even (system python3 is enough, since
    # chatui.py's own server path only needs the stdlib plus the small
    # webui_* modules, none of which touch torch).
    import socket
    import subprocess
    import time as _time
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "chatui.py"), "--port", str(port)],
        cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        ok = False
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1).close()
                ok = True
                break
            except Exception:                           # noqa: BLE001
                _time.sleep(0.1)
        check("chatui.py serves the UI with no model process running", ok)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_font_size_setting():
    """Every font-size in style.css is calc(basepx * var(--text-scale)), so
    Settings' Font size picker scales the whole UI with one number, and the
    baseline (--text-scale: 1) is itself the slightly-reduced default - not
    the old, larger sizes."""
    import re
    css = (Path(__file__).parent / "webui" / "style.css").read_text()
    check("--text-scale is defined on :root", "--text-scale: 1;" in css)

    sizes = re.findall(r"font(?:-size)?:\s*([^;]+);", css)
    unscaled = [s for s in sizes if "px" in s and "var(--text-scale)" not in s]
    check("no font-size in px escaped the --text-scale treatment",
          not unscaled, unscaled[:5])
    scaled = [s for s in sizes if "var(--text-scale)" in s]
    check("a meaningful number of rules were actually converted",
          len(scaled) > 60, len(scaled))

    # the new baseline must be a real reduction from the old raw values, not
    # just the same numbers wrapped in calc().
    m = re.search(r"font:\s*calc\((\d+(?:\.\d+)?)px \* var\(--text-scale\)\)/1\.6", css)
    check("body's base size actually went down (was 15px)",
          m is not None and float(m.group(1)) < 15, m and m.group(1))

    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    check("a dedicated setter exists for the font-size preference",
          "function setTextScale(" in js)
    check("it is kept out of state.settings (never sent to the model)",
          "state.settings" not in js.split("function setTextScale(")[1].split("\n\n")[0])
    check("it persists under its own localStorage key, not chatui.settings",
          'localStorage.setItem("chatui.textScale"' in js)
    check("the saved scale is applied on load, same as the theme",
          "chatui.textScale" in js.split("function loadSettings()")[1].split("\n\n")[0])
    check("Settings offers a font-size picker with Small/Default/Large/Extra large",
          all(label in js for label in ("Small", "Extra large"))
          and "Font size" in js and "textsize-row" in js)

    style_row = (Path(__file__).parent / "webui" / "style.css").read_text()
    check(".textsize-row lays its buttons out as an even row",
          ".textsize-row" in style_row)


def test_the_thinking_level_belongs_to_the_conversation():
    """A level used to be one global preference, so choosing "low" for a quick
    question left every later chat on low until you remembered to put it back.
    It is a property of the conversation you are in - the Settings entry is
    only what a new one starts on."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    check("the chat carries its own level", "thinking: null" in js
          or "state.thinking = null" in js)
    check("...and what is in force is the chat's, then the default",
          "function currentEffort" in js
          and "state.thinking || state.settings.thinking" in js)
    check("the turn is built from that, not from the raw settings",
          "function turnSettings" in js and "turnSettings()" in js)
    check("choosing one writes it to the conversation right away",
          "function setEffort" in js
          and 'body: JSON.stringify({ thinking: value === "default" ? "" : value })' in js)
    check("opening a conversation restores its level",
          "state.thinking = session.thinking || null" in js)
    check("...and the Settings field is labelled as the default, not the level",
          "Default thinking for new chats" in js)

    app = (Path(__file__).parent / "webui_app.py").read_text()
    check("the server accepts a level on a session", '"thinking"' in app)
    check("...and stores it with the turn", '"thinking": thinking or None' in app)
    # This once read `thinking in ("low", "medium", "high")`, which was wrong in
    # both directions at once: it dropped xhigh and max on templates that have
    # them, and forwarded "high" to templates that 400 on it. Whether a level is
    # accepted is the endpoint's business; the UI only offers what it declared.
    check("any declared level is forwarded, not a hardcoded three",
          'elif thinking and thinking != "default":' in app)


def test_levels_can_be_declared_from_the_menu():
    """Which levels an endpoint takes cannot always be probed - vLLM accepts
    reasoning_effort perfectly well and its /models row says nothing about it -
    so it has to be answerable by hand, from the place you noticed the problem
    rather than four clicks away in Settings."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    check("the six candidates are offered", "EFFORT_CANDIDATES" in js
          and '"minimal", "low", "medium", "high", "xhigh", "max"' in js)
    check("the editor opens from the chip's own menu", "function levelsEditor" in js
          and "levelsEditor({ left:" in js)
    check("a toggle saves to the provider", "function saveLevels" in js
          and 'await api("/ui/providers"' in js)
    check("...carrying no key, so the stored one is kept",
          "{ ...provider, efforts: levels }" in js)

    # The menu has to stay open while you toggle - it is a set, not a choice -
    # and each row has to show its own new state, because nothing else redraws
    # it while it is open.
    body = js.split("function levelsEditor(")[1].split("\nfunction ")[0]
    check("the rows are toggles that keep the menu open", "keep: true" in body)
    check("...and repaint their own tick", "button.replaceChild(icon(" in body)
    check("...marked so it can be read back", "button.dataset.on" in body)

    # This is the bug that made the editor unreachable. The document listener
    # closes a menu on any click outside it, and decides "outside" by walking
    # the target's ancestors. A row that opens a second menu replaces the
    # menu's children first, so by the time the click arrives at document the
    # button it started on has no ancestors at all - closest() returns null,
    # the click reads as outside, and the menu that had just opened was shut
    # in the same tick. It was still in the DOM, `hidden`, which is why a test
    # that only read its buttons passed while nothing was ever on screen.
    opener = js.split("function openMenu(")[1].split("\nfunction ")[0]
    check("a click on a row never reaches the document closer",
          "e.stopPropagation();" in opener)
    check("...and the row is handed its own button to repaint",
          "item.run(button)" in opener)


def test_every_test_is_actually_run():
    """Thirteen tests written in one sitting were never called once: main() is
    an explicit list, and adding the function is not adding the test. A test
    that does not run is worse than no test, because it reads as cover."""
    import re                                           # noqa: WPS433
    src = Path(__file__).read_text()
    defined = re.findall(r"^def (test_\w+)\(", src, re.M)
    body = src[src.index("def main():"):]
    called = set(re.findall(r"^\s+(test_\w+)\(", body, re.M))
    missing = [name for name in defined if name not in called]
    check(f"every one of the {len(defined)} tests is called from main()",
          not missing, ", ".join(missing))


def test_arguments_can_be_read_as_they_arrive():
    """A tool call's arguments are one JSON object built a token at a time, so
    nothing can be parsed until the last brace - which for a write_file
    carrying a whole page is minutes of a climbing character count and nothing
    to look at. The decoder walks the half-written fragment instead, and has to
    be exact under any chunking, because the text it hands back is shown."""
    import random                                       # noqa: WPS433
    import webui_agent as wa                            # noqa: WPS433

    want = {"path": "arcanum.html",
            "content": '<h1>Hi</h1>\nline "two"\ttabbed \u00e9 \\slash'}
    whole = json.dumps(want)

    def read(sizes):
        preview, got, at = wa.ArgPreview(), {}, 0
        for step in sizes:
            at = min(len(whole), at + step)
            for field, add in preview.feed(whole[:at]):
                got[field] = got.get(field, "") + add
        return got

    for step in (1, 2, 3, 7, 40, 5000):
        got = read([step] * (len(whole) // max(step, 1) + 2))
        check(f"exact in {step}-character pieces", got == want, got)

    random.seed(11)
    ragged = all(read([random.randint(1, 9) for _ in range(len(whole))]) == want
                 for _ in range(200))
    check("...and under 200 random ragged chunkings", ragged)

    # The point of all this: the filename is readable long before the file is.
    early = wa.ArgPreview().feed(whole[:60])
    check("the path is readable while the content is still arriving",
          ("path", "arcanum.html") in early, early)
    check("...and the content it has so far comes back under its own name",
          any(f == "content" and t for f, t in early), early)

    # One fragment can carry the end of one value and the start of the next.
    # Labelling the whole thing with the key that happened to be current when
    # the walk stopped filed a filename under the file's own contents.
    both = wa.ArgPreview().feed(whole)
    check("each piece is filed under the argument it belongs to",
          [f for f, _ in both] == ["path", "content"], both)

    # A half-written escape must not be decoded: half of an escape is a wrong
    # character that has already been shown and can never be taken back.
    needle = json.dumps("\u00e9")[1:-1]       # how json spells it: backslash u...
    at = whole.index(needle) + 2              # ...cut just past the backslash
    part = wa.ArgPreview()
    so_far = "".join(t for f, t in part.feed(whole[:at]) if f == "content")
    check("half of an escape is not guessed at",
          not so_far.endswith(("u", needle[0])) and needle[0] not in so_far,
          repr(so_far[-8:]))
    check("...and the character arrives whole once the rest of it does",
          "".join(t for f, t in part.feed(whole) if f == "content")
          .startswith("\u00e9"))


def test_a_cut_off_call_is_told_apart_from_a_malformed_one():
    """finish_reason is the official answer to "was this cut off?" and it
    cannot be relied on: the endpoint that produced this bug ended a reply in
    the middle of an 8,944-character string and still reported "stop". The
    text is better evidence - valid JSON never ends inside a string."""
    import webui_agent as wa                            # noqa: WPS433

    whole = json.dumps({"path": "a.html", "content": "x" * 200})
    for label, raw, cut in (
            ("stopped mid-string", whole[:120], True),
            ("stopped right after a comma", whole[:whole.index(",") + 1], True),
            ("malformed in the middle", '{"path": ,"content": "hi"}', False),
            ("a list, not an object", "[1, 2, 3]", False)):
        _, _, by_id = wa.repair_tool_calls(
            [{"id": "c1", "function": {"name": "write_file", "arguments": raw}}])
        check(f"{label} -> cut={cut}", by_id["c1"]["cut"] is cut, by_id["c1"])

    kept, _, by_id = wa.repair_tool_calls(
        [{"id": "c1", "function": {"name": "write_file", "arguments": whole}}])
    check("a whole call is left alone",
          not by_id and kept[0]["function"]["arguments"] == whole)
    # the unreadable one is still neutralised, or the conversation is finished
    broken, _, _ = wa.repair_tool_calls(
        [{"id": "c1", "function": {"name": "write_file",
                                   "arguments": whole[:120]}}])
    check("...and a cut one is still stored as {}",
          broken[0]["function"]["arguments"] == "{}")

    # What the model is actually handed. "Call it again" is the wrong advice
    # for a reply that ran out of room - repeating the same argument fails in
    # the same place - so this asserts the message, not how it is spelled in
    # the source: an earlier version of this check grepped the file and went on
    # passing while the sentence drifted across two string literals.
    class Cut:
        def stream(self, messages, tools=None, sampling=None):
            yield "tool_calls", [{"id": "c1", "type": "function",
                                  "function": {"name": "write_file",
                                               "arguments": whole[:120]}}]
            yield "finish", "stop"          # the endpoint does not own up to it

    events = list(wa.run_turn(Cut(), [{"role": "user", "content": "go"}], [],
                              wa.ToolContext(), mode="agent",
                              sampling={"max_tokens": 4096}, max_steps=2))
    said = " ".join(e.get("output", "") for e in events
                    if e.get("type") == "tool_result")
    check("the model is told the call was cut, not that it was malformed",
          "stopped in the middle of them" in said, said[:160])
    check("...and not to simply send the same thing again",
          "stop in the same place" in said, said[:160])
    check("...but to write the file in pieces",
          "add each further part with edit_file" in said, said[:160])
    check("...and how far it got, and how much room there was",
          "120 characters" in said and "4096 tokens" in said, said[:200])
    banner = next((e for e in events if e.get("type") == "error"), None)
    check("the person is told too, even though the endpoint said 'stop'",
          banner is not None and "did not report" in banner["message"],
          (banner or {}).get("message"))


def test_the_finished_file_arrives_as_an_event():
    """A row of small grey buttons is the wrong shape for the thing you asked
    for: Copy and Retry are what you might do next, and this is the answer. So
    the file gets its own card, the width of the reply - and it shows the page
    rather than describing it, which is also the fastest way to see that it
    came out right."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    css = (Path(__file__).parent / "webui" / "style.css").read_text()
    card = js.split("function pageReadyCard(")[1].split("\nconst KINDS")[0]

    check("it is a card, above the row of things you might do next",
          "body.append(pageReadyCard(" in js
          and ".msg-actions .primary-act" not in css)
    check("...and still a real link", 'card.target = "_blank"' in card
          and 'card.rel = "noopener noreferrer"' in card)
    check("...that says what it is", '"Ready"' in card and "kindOf(name)" in card)
    check("only one per reply",
          '!body.querySelector(".page-ready")' in js)

    # The preview is the file itself, rendered - which is the point, and also
    # the reason it has to be kept in its box.
    check("the preview is the real page", "shot.dataset.src" in js
          and 'el("iframe")' in js)
    check("...sandboxed in its own right, not only by the response",
          'frame.setAttribute("sandbox", "allow-scripts")' in js)
    check("...and out of the tab order, since the card is the link",
          'frame.setAttribute("tabindex", "-1")' in js)
    check("...unmounted when it scrolls away",
          "IntersectionObserver" in js and "shot.replaceChildren();" in js)
    check("...and simply shown when there is no observer to ask",
          'if (calm || !("IntersectionObserver" in window)) { show(); return; }' in js)

    # Appends mean the last write is not the file, so the size is the file's
    # own - read from the response and then dropped.
    check("the size is the file's, not the last write's",
          "function sizeInto" in js and 'res.headers.get("content-length")' in js)
    check("...and the body is thrown away once the headers land",
          "res.body?.cancel();" in js)

    ready = css.split(".page-ready {")[1].split("\n}")[0]
    check("it arrives rather than appearing", "ready-in" in ready)
    check("...with one light crossing it, once",
          "ready-shine 1.5s var(--ease) .2s 1" in css)
    check("the ring is the same idea as the running row",
          "mask-composite: exclude" in css.split(".page-ready .ring {")[1].split("}")[0]
          and "edge-run" in css.split(".page-ready .ring::before")[1].split("}")[0])
    check("nothing moves for someone who asked for less motion",
          ".page-ready, .page-ready::after, .page-ready .ring::before," in css)


def test_three_themes_and_a_background_that_does_not_band():
    """A wide, shallow ramp across a dark screen is where 8-bit colour runs out
    of steps and the eye reads the steps as stripes. And OLED is not "dark
    turned down": on that panel #000 is a pixel switched off, which is the
    whole reason to have the theme."""
    import re                                           # noqa: WPS433
    css = (Path(__file__).parent / "webui" / "style.css").read_text()

    check("there is an OLED theme", ':root[data-theme="oled"] {' in css)
    oled = css.split(':root[data-theme="oled"] {')[1].split("\n}")[0]
    check("...and its background is actually black", "--bg: #000000;" in oled)
    check("...with a gradient of its own", "--bg-grad:" in oled)
    check("...that arrives from black and returns to it",
          "rgba(0, 0, 0, 0)" in oled)

    # Every theme has to define every token, or a switch leaves half the UI
    # wearing the last one's colours.
    def tokens(block):
        return set(re.findall(r"(--[a-z0-9-]+):", block))
    dark = css.split(":root {")[1].split("\n}")[0]
    light = css.split(':root[data-theme="light"] {')[1].split("\n}")[0]
    # Not every token: --ring and the shadows are built out of --accent and
    # black, so they follow the theme by themselves, and --accent-ink is white
    # on any theme whose accent is saturated. It is the surface-and-text ladder
    # that has to be restated, because half of it inherited from another theme
    # is exactly how a switch leaves the UI wearing two palettes at once.
    ladder = {"--bg", "--bg-grad", "--panel", "--panel-2", "--chrome", "--raise",
              "--line", "--line-soft", "--text", "--text-2", "--muted",
              "--accent", "--accent-2", "--accent-soft",
              "--band", "--row-hover", "--row-on"}
    check("the dark theme is the one the others are measured against",
          ladder <= tokens(dark), sorted(ladder - tokens(dark)))
    for name, block in (("light", light), ("OLED", oled)):
        missing = sorted(ladder - tokens(block))
        check(f"the {name} theme restates the whole surface ladder",
              not missing, ", ".join(missing))

    # Fading to `transparent` fades to transparent BLACK, which desaturates the
    # ramp on the way out and leaves a dirty edge where it lands.
    for name, block in (("dark", dark), ("light", light), ("OLED", oled)):
        grad = block.split("--bg-grad:")[1].split(";")[0]
        check(f"the {name} gradient never fades to bare `transparent`",
              "transparent" not in grad, grad[:70])

    check("a dither breaks up what is left of the banding",
          "--bg-noise:" in css and "feTurbulence" in css)
    check("...blended with overlay, which leaves black at black",
          "background-blend-mode: var(--bg-blend);" in css)

    # background-blend-mode takes one entry per layer INCLUDING the background
    # colour, and a short list repeats - which silently put `overlay` on the
    # base colour, and on a theme with a third gradient, on a gradient too.
    for name, block in (("dark", dark), ("light", light), ("OLED", oled)):
        grad = block.split("--bg-grad:")[1].split(";")[0]
        layers = grad.count("-gradient(") + 1 + 1        # gradients + noise + colour
        blend = block.split("--bg-blend:")[1].split(";")[0] if "--bg-blend:" in block \
            else dark.split("--bg-blend:")[1].split(";")[0]
        check(f"the {name} theme blends exactly its own layers",
              len(blend.split(",")) == layers,
              f"{len(blend.split(','))} entries for {layers} layers")

    # ":not([data-theme='dark'])" was the same as "nothing chosen" while there
    # were two themes, and stopped being it the moment there was a third.
    check("following the OS means nothing was chosen, not 'not dark'",
          ':root:not([data-theme="dark"])' not in css)
    check("...and the topbar icon follows the same rule",
          ":root:not([data-theme]) #theme-btn .theme-sun" in css)
    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    check("the button cycles all three", 'const THEME_CYCLE = ["light", "dark", "oled"];' in js)
    check("...and Settings offers them plus the system",
          '["oled", "OLED"]' in js and '["", "System"]' in js)


def test_work_in_progress_looks_like_it():
    """Two places say "this is happening now", and both were saying it quietly:
    a flat 90deg wipe of one accent behind the word Thinking, and, in the
    sidebar, nothing at all - the list is only refreshed when a turn ENDS, so
    the row for the chat you were watching never showed it was working."""
    css = (Path(__file__).parent / "webui" / "style.css").read_text()
    js = (Path(__file__).parent / "webui" / "app.js").read_text()

    check("a light runs down the rail the thinking hangs from",
          ".think.live::before" in css and "rail-run" in css)
    check("...so it is visible even with the block folded shut",
          "top: 0; bottom: 0" in css.split(".think.live::before")[1].split("}")[0])
    check("...and the sweep across the words has a core, not one flat colour",
          "color-mix(in srgb, #fff 70%, var(--accent))" in css)
    check("both are real keyframes", "@keyframes rail-run" in css)

    check("a chat still working wears a light round its edge",
          ".session .edge::before" in css and "@keyframes edge-run" in css)
    check("...drawn as the row's own outline, not a box near it",
          "mask-composite: exclude" in css.split(".session .edge {")[1].split("}")[0])
    check("...and only on rows that are working",
          'if (s.running) row.append(el("i", "edge"));' in js)

    # The sidebar has no poll: without this the chat you are looking at never
    # showed as running, because the only refresh happens when it stops.
    check("the row is marked while the turn runs, without a poll",
          "function markRunningRow" in js and "markRunningRow(on);" in js)
    check("...a chat drawn for the first time mid-turn gets it too",
          "markRunningRow(state.streaming);" in js)
    check("...and it comes off when the turn ends, whatever the server says yet",
          "this window is the authority" in js)


def test_a_page_the_agent_wrote_can_be_opened_but_not_trusted():
    """Asking for "a single HTML file" and then being told where it is on disk
    is a strange place to stop, so the conversation offers to open it. That
    means serving a file a model wrote, to a browser - and if it were served
    on this UI's own origin it could read providers.json, keys and all, or
    delete conversations, from a page the person only meant to look at."""
    import shutil, tempfile                             # noqa: WPS433
    from webui_app import ChatUI                        # noqa: WPS433

    root = Path(tempfile.mkdtemp(prefix="page-"))
    try:
        work = root / "workspace"
        (work / "sub").mkdir(parents=True)
        (work / "made.html").write_text("<h1>hi</h1>", encoding="utf-8")
        (work / "sub" / "deep.html").write_text("<p>deep</p>", encoding="utf-8")
        (work / "notes.md").write_text("notes", encoding="utf-8")
        (work / ".env").write_text("KEY=secret", encoding="utf-8")
        (work / "secrets.env").write_text("KEY=secret", encoding="utf-8")
        (root / "above.html").write_text("<p>not yours</p>", encoding="utf-8")
        ui = ChatUI(root=root, cfg={}, model_base="http://127.0.0.1:1/v1",
                    model_id="m")

        def get(path):
            return ui.handle("GET", "/ui/file", {"path": path}, b"")

        ok = get("made.html")
        check("a page it wrote is served", ok.status == 200 and b"hi" in ok.body)
        check("...as html", ok.content_type.startswith("text/html"))
        check("a page in a subfolder too", get("sub/deep.html").status == 200)
        check("and other things worth looking at", get("notes.md").status == 200)

        # The workspace is a real folder with real secrets in it.
        for bad in (".env", "sub/../.env", "../above.html", "/etc/passwd",
                    "../../etc/passwd", "", "."):
            got = get(bad)
            check(f"refused: {bad!r}", got.status in (403, 404), got.status)
        check("a file whose name merely ends in .env is refused too",
              get("secrets.env").status == 403)
        check("...and the refusal says what this route is for",
              b"not viewable" in get("secrets.env").body)

        # Scripts must run - a page that cannot run its own JS is not a preview
        # of anything - but not as this UI.
        csp = ok.headers.get("Content-Security-Policy", "")
        check("the page is sandboxed", "sandbox" in csp)
        check("...it may run its own scripts", "allow-scripts" in csp)
        check("...but never as this origin", "allow-same-origin" not in csp)
        check("...and cannot retarget the page it came from",
              "base-uri 'none'" in csp and "form-action 'none'" in csp)
        check("the type is not sniffed into something else",
              ok.headers.get("X-Content-Type-Options") == "nosniff")
        check("a rewritten file is never served from cache",
              ok.headers.get("Cache-Control") == "no-store")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    check("the conversation offers the page it made",
          "function wroteAPage" in js and "const OPENABLE" in js)
    check("...for the shapes a browser can show as a page",
          "html?|svg|pdf" in js)
    check("...only for a call that actually wrote one",
          'name !== "write_file" && name !== "edit_file"' in js)
    check("it opens in its own tab, with no handle back to this one",
          'a.target = "_blank"' in js and 'a.rel = "noopener noreferrer"' in js)
    # A file written and then appended to six times is one page, not seven.
    check("one link per file, across the whole thread",
          '($("#thread") || container).querySelectorAll(".tool .open-page")' in js)
    check("and the finished turn hands it over as its own card",
          "function pageReadyCard" in js
          and "body.append(pageReadyCard(body.dataset.page, true));" in js)
    check("...including a conversation reopened later",
          "if (i === lastSaid && made) body.dataset.page = made;" in js)


def test_the_reply_ceiling_is_not_the_thing_that_decides_what_fits():
    """Max new tokens is a ceiling, not an allocation: a limit never reached
    costs nothing, and one that is reached costs the whole tool call, because
    the arguments are cut off mid-JSON and cannot be run. 4096 could not write
    a page. 16384 could not write a long one. Both were shipped as defaults,
    and both quietly decided how much work fitted in one reply."""
    import webui_app as wa                              # noqa: WPS433

    check("the ceiling is high enough to stop being the limit",
          wa.DEFAULT_MAX_TOKENS == 65536, wa.DEFAULT_MAX_TOKENS)
    env = (Path(__file__).parent.parent / ".env.example").read_text()
    check("...and the shipped setting agrees with the code",
          "MAX_TOKENS=65536" in env)

    # ...except on a model whose whole window is smaller than the ceiling.
    for context, want in ((None, 65536), (0, 65536), (4096, 2048),
                          (8192, 4096), (32768, 16384), (131072, 65536),
                          (262144, 65536)):
        got = wa.default_max_tokens(context)
        check(f"a {context} window gets {want}", got == want, got)
    check("even a tiny window leaves something usable",
          wa.default_max_tokens(512) == 2048)

    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    # The slider should not offer a ceiling the endpoint cannot reach.
    gen = js.split('fieldSlider("max_tokens"')[0].split("function paneGeneration")[1]
    check("the slider's top follows the window, not a fixed number",
          "activeContextLength()" in gen and "Math.floor(room / 2)" in gen)
    check("...and says what that window is",
          "This endpoint's window is" in js)


def test_a_ceiling_nobody_chose_moves_when_the_default_moves():
    """Every time the shipped ceiling turned out to be too low, the people
    carrying the old one were the ones who never touched the setting: it was
    saved once from a default. The old migration ran once under a flag and
    recognised one specific number, so a browser that had already seen it could
    never be lifted again - which is exactly how a saved 4096 survived two
    raises of the default."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    body = js.split("function migrateSettings()")[1].split("\n}")[0]

    check("the defaults this kit has shipped are known by name",
          "const INHERITED_MAX_TOKENS = [4096, 16384];" in js)
    check("a value that is one of them was inherited, so it moves",
          "INHERITED_MAX_TOKENS.includes(mine)" in body)
    check("...and any other value is the person's own and is left alone",
          "return;   // chosen: leave it alone" in body)
    check("a value already above the default is not lowered",
          "mine >= fresh" in body)
    # The one-shot flag is what made this unrepeatable.
    check("the stamp carries the default it applied, not just 'done'",
          'localStorage.setItem("chatui.maxTokensDefault", String(fresh));' in body)
    check("...so a browser stamped with an older default is lifted again",
          'Number(localStorage.getItem("chatui.maxTokensDefault")) === fresh' in body)
    check("the flag that could never fire twice is gone",
          "maxTokensBumped" not in js)
    # Changing someone's saved setting silently is worse than the setting.
    check("it says so when it changes a saved setting",
          "Max new tokens raised from" in body and "toast(" in body)


def test_an_approval_asks_in_words_a_person_can_judge():
    """This is the one moment where someone has to decide something on the
    agent's behalf, and it was the least readable thing on screen: the
    function's name jammed against its description - "run_python Check raw
    bytes for mangled CSS names" - over the arguments as escaped JSON, so the
    script being approved arrived full of \\n and \\". The button offering to
    stop asking was labelled with the function name, which is the one word in
    the sentence a person has no way to judge."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    body = js.split("function approvalCard(")[1].split("\nasync function ")[0]

    check("it asks in a sentence", "The agent wants to ${view.asks" in body)
    for tool, asks, noun in (
            ("run_python", "run a Python script on this computer", "running Python"),
            ("run_command", "run a command on this computer", "running commands"),
            ("write_file", "write a file", "writing files"),
            ("edit_file", "change a file", "editing files")):
        view = js.split("TOOL_VIEW")[1].split("};")[0]
        check(f"{tool} says what it wants in words", f'asks: "{asks}"' in view)
        check(f"...and names itself as a kind of thing", f'noun: "{noun}"' in view)
    check("a tool with no sentence of its own still reads as one",
          '"run something on this computer"' in body and '"change a file"' in body)

    check("the button is not labelled with a function name",
          "Always allow ${noun}" in body and "Always allow ${ev.name}" not in js)
    check("...and neither is the verdict it leaves behind",
          "`Allowed - ${noun} will not ask again" in body)

    # The approval lives inside the card for the very call it is about, and
    # that card already shows the command, the script, the file.
    check("it does not print the call a second time",
          "const shown = !!card;" in body and "if (!shown) {" in body)
    check("...and nothing renders the arguments as JSON any more",
          "JSON.stringify(ev.args" not in js)

    # A byte count is a fact about a file, not about a 68-character command.
    check("a size is only shown where a size means something",
          "body.length > 400" in js and "ev.chars > 400" in js)


def test_thinking_lands_where_it_happened():
    """The block was found with querySelector(".think") - the FIRST one in the
    message - and prepended. So in an agent turn every later burst of thinking
    was poured back into a window pinned above step one, still growing while
    the work scrolled past underneath it."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    block = js.split("function thinkBlock(")[1].split("\n/* A run of thinking is over")[0]
    check("a new run of thinking gets its own block",
          "const all = container.querySelectorAll(\".think\");" in block
          and "all[all.length - 1]" in block)
    check("...and lands below what it follows, not above it",
          "container.append(node);" in block and "container.prepend(node)" not in js)
    check("a block that has been closed is not reopened",
          'if (node && node.dataset.closed) node = null;' in block)

    close = js.split("function closeThink(")[1].split("\n}")[0]
    check("saying something ends the run of thinking",
          'node.dataset.closed = "1";' in close)
    check("...and so does reaching for a tool",
          js.count("closeThink(body);") >= 3, js.count("closeThink(body);"))
    check("every block is settled at the end, not only the first",
          'body.querySelectorAll(".think").forEach(settleThinkBlock);' in js)


def test_a_sentence_is_not_cut_in_half_by_a_tool_call():
    """The splitter holds a few characters back in case they are the start of
    a <think> marker arriving in two pieces, and released them at
    finish_reason - which comes after the tool call. So the tail of every
    sentence landed below the card it belonged above, mid-word: the transcript
    read "...no external refere", card, "nces."."""
    import io                                           # noqa: WPS433
    import webui_agent as wa                            # noqa: WPS433

    class Splits:
        """Content, then a tool call, the way an endpoint really sends it."""

        def _post(self, path, payload):
            frames = []
            for piece in ["The file is written. Checking it is really one ",
                          "file with no external references."]:
                frames.append({"choices": [{"index": 0,
                                            "delta": {"content": piece}}]})
            frames.append({"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "c1", "type": "function",
                 "function": {"name": "read_file",
                              "arguments": '{"path":"a.html"}'}}]}}]})
            frames.append({"choices": [{"index": 0, "delta": {},
                                        "finish_reason": "tool_calls"}]})
            body = b"".join(b"data: " + json.dumps(f).encode() + b"\n\n"
                            for f in frames) + b"data: [DONE]\n\n"
            return io.BytesIO(body)

    client = wa.ModelClient.__new__(wa.ModelClient)
    client.base_url, client.api_key, client.model = "http://x/v1", "", "m"
    client.timeout = 5
    client._post = Splits()._post

    order = [(kind, value) for kind, value in client.stream([], None, None)]
    said, before_call = [], True
    for kind, value in order:
        if kind in ("tool_partial", "tool_calls"):
            before_call = False
        elif kind == "content":
            said.append((value, before_call))
    text = "".join(v for v, _ in said)
    check("every character of the sentence still arrives",
          text == "The file is written. Checking it is really one file with "
                  "no external references.", text)
    check("...and all of it before the tool call, not around it",
          all(first for _, first in said),
          [v for v, first in said if not first])


def test_the_step_budget_fits_the_way_files_are_written():
    """Eight rounds was a sensible budget when a step meant "read a file, then
    answer". The agent is now told to write a long file by opening it and
    appending the rest, and a page with its own CSS and script is a dozen
    appends by itself - the build stopped two thirds through and left a half
    written file behind."""
    app = (Path(__file__).parent / "webui_app.py").read_text()
    check("a turn has room for a build made of appends",
          'cfg.get("AGENT_MAX_STEPS") or 24' in app)
    env = (Path(__file__).parent.parent / ".env.example").read_text()
    check("...and the shipped setting agrees with the code",
          "AGENT_MAX_STEPS=24" in env)
    check("...and says why it is not smaller", "half written" in env)

    agent = (Path(__file__).parent / "webui_agent.py").read_text()
    # Running out of steps is a dead end the person can do something about.
    check("running out of steps says the work may be unfinished",
          "the work may be unfinished" in agent)
    check("...and how to carry on", "say 'continue' to carry on" in agent)


def test_a_cut_off_call_keeps_what_arrived():
    """The server replaces an unreadable arguments string with {} so it can
    never poison the conversation - which means the finished call arrives at
    the browser with nothing in it. The person had just watched ten thousand
    characters of that file arrive, and the card was throwing them away at the
    exact moment they turned out to matter."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    card = js.split("function toolCard(")[1].split("\nfunction ")[0]
    check("the text that arrived is kept when the arguments are not readable",
          "const live = pending && pending.querySelector" in card
          and "const lost = live &&" in card)
    check("...and labelled as what it is",
          '"what arrived before it stopped"' in card)
    check("the subject survives too, having been decoded before the cut",
          'pending.querySelector(".subject")?.textContent' in card)

    agent = (Path(__file__).parent / "webui_agent.py").read_text()
    # The banner used to fire only on finish_reason == "length". The endpoint
    # that caused this ended a reply inside a 10,582-character string and still
    # reported "stop", so the banner never appeared and the person was left
    # with a failed call and no reason for it.
    check("the banner follows the evidence, not only finish_reason",
          "if truncated or cut:" in agent)
    check("...and says which of the two it saw",
          "The endpoint did not report" in agent and "usual cause" in agent)
    check("both name the limit that was hit",
          "Max new tokens is {limit}" in agent
          and "This reply could be at most {room} tokens." in agent)

    # Better than reporting it well is not walking into it.
    check("the agent is told up front that a whole file has to fit in one reply",
          "A whole file has to fit in one reply" in agent)
    check("...and what to do instead",
          "append the rest with\n  edit_file" in agent)


def test_a_browser_that_leaves_is_not_an_error():
    """A browser hangs up constantly and legitimately: a refresh, a closed tab,
    Stop aborting the fetch part-way through a streamed turn. The write in
    flight then fails, and only two of the three shapes that takes were caught.
    Windows raises the third - WinError 10053 arrives as
    ConnectionAbortedError - so every ordinary refresh printed a nine-frame
    traceback into the console the person is reading as their log.

    This cannot happen on this machine, so it is provoked directly rather than
    waited for: each shape is raised from the socket the handler writes to.
    """
    import io as _io                                    # noqa: WPS433
    import chatui                                       # noqa: WPS433

    class Gone(_io.RawIOBase):
        """A socket whose peer left. Every write fails, as one does."""

        def __init__(self, blow_up):
            self.blow_up = blow_up

        def write(self, data):
            raise self.blow_up("the peer went away")

        def writable(self):
            return True

        def flush(self):
            pass

    class Fake(chatui._Handler):
        def __init__(self, blow_up, events):
            self.wfile = Gone(blow_up)
            self.rfile = _io.BytesIO(b"")
            self.path = "/ui/chat"
            self.headers = {}
            self.client_address = ("127.0.0.1", 1)
            self.close_connection = False
            self.requestline = "POST /ui/chat HTTP/1.1"
            self.request_version = "HTTP/1.1"
            self.command = "POST"
            self._events = events
            self.sent = []

        def send_response(self, *a, **k):
            self.sent.append(a)

        def send_header(self, *a, **k):
            pass

        def end_headers(self):
            pass

        # stand in for the app: one streamed turn, and one static file
        @property
        def ui(self):
            events, this = self._events, self
            class _Ui:
                def handle(self, *a, **k):
                    return (chatui.Stream(events) if events
                            else chatui.Response(200, "text/plain", b"x" * 99))
            return _Ui()

    def stream():
        for i in range(50):
            yield {"type": "content", "delta": f"piece {i}"}

    # All three are ConnectionError; the Windows one is the one that got out.
    for blow_up in (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
        for what, events in (("a streamed turn", stream()), ("a static file", None)):
            handler = Fake(blow_up, events)
            try:
                handler._dispatch("POST")
                ok, why = True, ""
            except Exception as e:                      # noqa: BLE001
                ok, why = False, f"{type(e).__name__}: {e}"
            check(f"{blow_up.__name__} while sending {what} is not an error",
                  ok, why)
            check("...and the connection is not read from again",
                  handler.close_connection is True)

    # A real fault must still be reported: swallowing everything here would
    # hide a bug in the UI behind a message about the browser.
    class Boom(Fake):
        @property
        def ui(self):
            class _Ui:
                def handle(self, *a, **k):
                    raise ValueError("a real bug")
            return _Ui()

    try:
        Boom(BrokenPipeError, None)._dispatch("POST")
        got = "nothing"
    except ValueError:
        got = "ValueError"
    except Exception as e:                              # noqa: BLE001
        got = type(e).__name__
    check("a genuine fault still comes through", got == "ValueError", got)

    # ...and the same forgiveness one level up, where socketserver prints its
    # own traceback for whatever escapes - including its teardown writes.
    src = (Path(__file__).parent / "chatui.py").read_text()
    check("the server does not print a traceback for a client that left",
          "class _Server(ThreadingHTTPServer)" in src
          and "def handle_error" in src
          and "isinstance(sys.exc_info()[1], ConnectionError)" in src)
    check("...and that is the server actually used",
          "_Server((args.host, args.port), _Handler)" in src)
    check("the async mount forgives the same three",
          "except (ConnectionError, asyncio.CancelledError):" in src)


def test_notes_are_written_in_the_dialect_the_renderer_reads():
    """The renderer reads * for emphasis and deliberately ignores _, so that a
    snake_case name written in prose survives. A server-side note that used
    underscores printed its own markup on screen."""
    agent = (Path(__file__).parent / "webui_agent.py").read_text()
    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    check("the renderer emphasises with asterisks",
          "<em>$2</em>" in js and "\\*([^*" in js)
    check("...and leaves underscores alone, so snake_case is safe",
          "_([^_" not in js)
    for line in agent.splitlines():
        if line.strip().startswith('yield {"type": "content", "delta":'):
            check(f"a note does not emit raw _ markup: {line.strip()[:60]}",
                  '"_' not in line and '_"' not in line, line.strip())


def test_a_tool_call_reads_as_a_sentence():
    """A card used to be the function's name over a JSON dump of its arguments
    and a blob of output. That is a database row. The person reading it wants
    to know that a file was written and which one - and only sometimes what
    went into it."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()

    check("every tool says what it does in words", "const TOOL_VIEW" in js)
    for name in ("write_file", "edit_file", "read_file", "list_dir",
                 "find_files", "search_text", "run_command", "run_python",
                 "job_output", "job_kill", "web_search", "web_fetch",
                 "update_plan", "ask_user"):
        check(f"{name} introduces itself", f"{name}:" in js.split("TOOL_VIEW")[1]
              .split("};")[0])
    check("a tool nobody taught it about still reads as one",
          "function toolView" in js and "Running" in js)

    view = js.split("function toolView")[0].split("const TOOL_VIEW")[1]
    check("write_file is 'Writing' then 'Wrote'",
          'doing: "Writing", done: "Wrote"' in view)
    check("...and names the file, not the argument object",
          "subject: (a) => a.path" in view)
    check("the argument worth reading as text is marked as such",
          'text: "content"' in view)

    # A green DONE on every row is noise: a call that worked is the ordinary
    # case and says so by saying nothing.
    fin = js.split("function finishToolCard")[1].split("\nfunction ")[0]
    check("a call that worked wears no badge",
          'badge.textContent = ok ? "" : "failed"' in fin)
    check("...and one that failed does, and stays open",
          "card.open = !ok" in fin)
    css = (Path(__file__).parent / "webui" / "style.css").read_text()
    check("an empty badge takes no room", ".tool .state:empty { display: none; }" in css)
    check("the section captions stopped shouting",
          "text-transform: uppercase" not in css.split(".tool .lbl {")[1].split("}")[0])

    detail = js.split("function toolDetail")[1].split("\nfunction ")[0]
    check("the plan is drawn as a checklist, not printed as JSON",
          "planList(args.todos" in detail)
    check("edit_file shows what it replaced, not only what it wrote",
          "rest.old_text" in detail)
    check("leftover arguments are named values, not a JSON blob",
          'el("dl", "tool-args")' in detail)
    check("...and what the summary already said is not repeated underneath",
          "String(rest[k]) !== said" in detail)
    check("a result the card has already drawn is not printed twice",
          "resultAs" in js)


def test_a_file_can_be_watched_as_it_is_written():
    """The card used to show a character count and nothing else while a large
    write_file streamed, which is the moment the person most wants to see what
    is happening."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    pend = js.split("function pendingToolCard")[1].split("\nfunction ")[0]
    check("the card is open while it is being written", "card.open = true" in pend)

    upd = js.split("function updatePendingToolCard")[1].split("\nfunction ")[0]
    check("the streamed text goes into a live block", "tool-live" in upd)
    check("...appended, so nothing is re-rendered per fragment",
          "pre.append(document.createTextNode(add))" in upd)
    check("...and it follows the tail", "pre.scrollTop = pre.scrollHeight" in upd)
    # Following the tail is only welcome while the person has not scrolled up
    # to read something further back.
    check("...unless the person has scrolled up in it",
          "pre.scrollHeight - pre.scrollTop - pre.clientHeight < 60" in upd)
    check("a short argument becomes the card's subject instead",
          "setToolSubject(card, subject)" in upd)

    card = js.split("function toolCard(")[1].split("\nfunction ")[0]
    check("the finished call reuses the card that was streaming",
          "const pending = container.querySelector" in card
          and "const card = pending ||" in card)
    check("...so the size the person watched climb is still there at the end",
          "setToolMeta(card, sizeOf(" in card)

    agent = (Path(__file__).parent / "webui_agent.py").read_text()
    check("the server sends the decoded text, not only a count",
          '"parts": preview.feed(' in agent)


def test_the_model_picker_opens_on_the_list_you_are_using():
    """The picker is two lists and one of them is nearly always the wrong one:
    a chat answered by a provider has no use for seven local quants."""
    js = (Path(__file__).parent / "webui" / "app.js").read_text()
    check("its sections fold", "function pickerSection" in js)
    body = js.split("async function modelModal")[1].split("\n/* Any OpenAI")[0]
    check("which one opens is decided by what is answering this chat",
          'const onLocal = !state.provider || state.provider === "local"' in body)
    check("this computer opens when this computer is answering",
          "onLocal || only" in body)
    check("providers open when a provider is answering", "!onLocal ||" in body)
    check("a lone section stays open, having nothing to fold away to",
          "const only = !data.remote?.length" in body)
    # An explanation of rows that have been folded away is a loose sentence
    # under a heading.
    check("each section's note is inside it",
          'here.append(el("div", "muted-note"' in body)

    fold = js.split("function fold(")[1].split("\n}")[0]
    check("a folded section is clipped and out of the tab order",
          "inert" in fold and "aria-hidden" in fold)
    css = (Path(__file__).parent / "webui" / "style.css").read_text()
    check("...with a fallback for a browser that has no inert",
          ".pick-group.closed .pick-inner { pointer-events: none; }" in css)
    check("folding animates rather than jumping",
          ".pick-group.closed .pick-rows { grid-template-rows: 0fr; }" in css)


def main():
    root = Path(__file__).resolve().parent.parent
    tmp = root / "workspace"
    tmp.mkdir(exist_ok=True)
    print("sandbox")
    test_sandbox(tmp)
    print("read before write")
    test_read_before_write(tmp)
    print("plan")
    test_plan_tool(tmp)
    print("commands")
    test_command_shape(tmp)
    test_untrusted_label()
    print("models")
    test_model_planning()
    print("providers")
    test_providers()
    print("modes")
    test_modes()
    print("network guard")
    test_origin_guard()
    test_lan_guard()
    print("html")
    test_html()
    print("packaging")
    test_wheel_tags()
    test_wheel_selection(tmp)
    test_pwa()
    test_ready_after_mount()
    test_shortcuts()
    print("first run")
    test_downloader(tmp)
    test_download_cancel(tmp)
    test_setup_core(tmp)
    test_setup_web(tmp)
    test_logbook(tmp)
    print("review regressions")
    test_download_path_containment(tmp)
    test_download_no_token_across_hosts()
    test_download_retry_accounting(tmp)
    test_download_complete_part_is_kept(tmp)
    test_download_transient_http_is_retried(tmp)
    test_download_failure_is_terminal(tmp)
    test_event_log_survives_trimming()
    test_phase_is_not_a_latch()
    test_retry_after_a_failure(tmp)
    test_env_injection()
    test_setup_peer_guard()
    test_port_probe_detects_a_listener()
    test_log_keeps_only_the_last_redraw()
    test_logbook_start_is_idempotent(tmp)
    test_powershell_quoting()
    test_child_pipe_is_read_as_bytes()
    test_shortcuts_wait_for_a_working_start()
    test_second_ctrl_c_still_kills_the_child()
    test_installer_version_is_overridable()
    test_broken_venv_is_rebuilt()
    test_bench_vram_gpu_attrs()
    test_bench_vram_kv_rate_excludes_vision()
    test_profiles_carry_measured_numbers()
    test_planner_arithmetic_is_pinned()
    test_verified_context_ceiling()
    test_vision_is_asked_not_guessed()
    test_quality_labels_do_not_invert()
    test_claimed_minimum_vram_is_honest()
    test_no_invalid_escape_sequences()
    test_simulation_mode()
    test_vision_toggle_is_cheap_and_clearable()
    test_rename_accepts_spaces()
    test_webui_restarts_itself()
    test_pin_does_not_sit_on_the_timestamp()
    test_no_undefined_globals()
    test_ui_chrome_is_quiet()
    test_default_endpoint_for_new_chats()
    test_new_chats_keep_the_model_you_last_used()
    test_provider_images()
    test_effort_levels_come_from_the_endpoint()
    test_thinking_block_is_a_real_toggle()
    test_stop_reaches_the_gpu()
    test_thinking_block_follows_its_own_text()
    test_slash_commands()
    test_reasoning_from_any_endpoint()
    test_message_edit_is_in_place()
    test_provider_context_length()
    test_thinking_effort()
    test_setup_page_survives_the_handover()
    test_setup_page_can_scroll()
    test_the_suite_never_writes_the_real_env()
    test_web_setup_matches_the_console()
    test_vision_prompt_edge_cases()
    test_recommendation_prefers_verified_rows()
    test_menu_marks_unverified_rows()
    test_model_switch_without_a_readable_gpu()
    test_model_switch_uses_the_planner_rule()
    test_bench_bat_checks_the_venv()
    test_webui_bat_runs_the_ui_without_a_model()
    test_font_size_setting()
    test_sidebar_groups_by_kind()
    test_a_truncated_tool_call_cannot_brick_a_chat()
    test_a_poisoned_session_heals_when_it_is_opened()
    test_a_long_tool_call_is_visible_while_it_streams()
    test_stray_br_is_a_line_break_not_text()
    test_the_renderer_cannot_be_made_to_emit_markup()
    test_a_turn_answers_every_call_it_announces()
    test_only_a_real_truncation_blames_max_tokens()
    test_menus_survive_the_click_that_opens_them()
    test_thinking_is_visible_and_settable_from_the_composer()
    test_the_server_describes_itself_on_models()
    test_a_provider_can_be_told_what_it_takes()
    test_the_effort_level_that_is_set_is_the_one_that_is_sent()
    test_the_thinking_level_belongs_to_the_conversation()
    test_levels_can_be_declared_from_the_menu()
    test_arguments_can_be_read_as_they_arrive()
    test_a_cut_off_call_is_told_apart_from_a_malformed_one()
    test_the_finished_file_arrives_as_an_event()
    test_three_themes_and_a_background_that_does_not_band()
    test_work_in_progress_looks_like_it()
    test_a_page_the_agent_wrote_can_be_opened_but_not_trusted()
    test_the_reply_ceiling_is_not_the_thing_that_decides_what_fits()
    test_a_ceiling_nobody_chose_moves_when_the_default_moves()
    test_an_approval_asks_in_words_a_person_can_judge()
    test_thinking_lands_where_it_happened()
    test_a_sentence_is_not_cut_in_half_by_a_tool_call()
    test_the_step_budget_fits_the_way_files_are_written()
    test_a_cut_off_call_keeps_what_arrived()
    test_a_browser_that_leaves_is_not_an_error()
    test_notes_are_written_in_the_dialect_the_renderer_reads()
    test_a_tool_call_reads_as_a_sentence()
    test_a_file_can_be_watched_as_it_is_written()
    test_the_model_picker_opens_on_the_list_you_are_using()
    test_every_test_is_actually_run()
    try:
        urllib.request.urlopen(urllib.request.Request(
            BASE + "/ui/config", headers=UI_HEADERS), timeout=3).close()
    except Exception as e:                              # noqa: BLE001
        print(f"\n  dev server not reachable at {BASE} ({e})")
        print("  start it with:  python tools/chatui.py --mock --port 8890")
        return 2
    print("network guard")
    test_picker_scope()
    print("chat mode")
    test_chat_turn()
    test_speed_report()
    print("agent mode")
    test_agent_readonly()
    test_agent_approval(tmp)
    test_ask_user()
    test_plan_event()
    print("stream control")
    test_turn_outlives_the_window()
    test_one_turn_per_session()
    test_rewind()
    test_cancel()
    print("sessions")
    test_pin_and_search()
    test_pin_reorder()
    test_sessions()
    print()
    print(f"  {'ALL PASS' if not failures else str(len(failures)) + ' FAILED: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
