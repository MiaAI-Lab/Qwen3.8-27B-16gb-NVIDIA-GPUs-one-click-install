---
name: Bug report
about: The kit is not behaving as expected, produces an error, or numbers do not match the README.
title: ""
labels: "bug"
assignees: ""
---

<!-- Thank you for using this model kit!

     If you are looking for support, please check the README first,
     or reach out on X:
      * https://x.com/MiaAI_lab

     If you have found a bug, then fill out the template below.
-->

---

## Environment

<!-- Fill in what applies to your setup. The README's tables list every knob and its default. -->

- OS: <!-- e.g. Windows 11, Ubuntu 24.04 -->
- GPU / VRAM: <!-- e.g. RTX 4080 16 GB, RTX 4090 24 GB -->
- Driver: <!-- `nvidia-smi` first line -->
- Python: <!-- 3.11 / 3.12 / 3.13, 64-bit -->
- Node: <!-- `node -v`, or "not installed" -->
- How you started: <!-- e.g. windows\START-HERE.bat, windows\start.bat, ./linux/setup.sh, ./linux/start.sh, simplex start -->
- Quant / model id: <!-- e.g. 2.5 bpw, qwen3.8-27b-exl3-2.5bpw — from `simplex status` or GET /v1/models -->
- Relevant settings: <!-- e.g. CONTEXT_SIZE, GPU_MEM_GB, CACHE_QUANT, VISION, DRAFT, PORT, UI -->

---

## Steps to Reproduce

<!-- Full steps so that we can reproduce the problem. -->

1. <!-- e.g. `windows\START-HERE.bat` — what it printed up to the failure -->
2. ... <!-- the request or action that shows the bug -->
3. ... <!-- e.g. "curl /v1/models returns a different served name than configured" -->

**Expected results:** <!-- what did you expect to happen? -->

**Actual results:** <!-- what did you actually see happen? -->

---

### Additional context

Add anything else: a minimal failing request, JSON responses, and so on.

<details>
<summary>Minimal reproduction sample</summary>

<!--
      If the bug is about model output or API behavior, attach a minimal reproducible
      request below between the lines with the backticks.
-->

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-exl3-2.5bpw",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 256
  }'
```

</details>

<details>
  <summary>Logs</summary>

<!--
      Paste the log output below between the lines with the backticks, and mention
      whether it came from the launcher window, logs/, or a client.

      Common culprits worth checking before filing:
        * Weights missing or incomplete -> run setup again; downloads are resumable.
        * Model crawls on Windows -> free VRAM (the preflight lists what is holding it).
        * "Images: off" -> lower CONTEXT_SIZE, or pick a smaller quant.
        * Insufficient VRAM in split -> lower CONTEXT_SIZE or GPU_MEM_GB.
        * Chat UI missing -> Node 22.19+; /v1 still serves without it.
-->

```

```

</details>

<!--
      Consider also attaching screenshots and/or videos to better illustrate the issue.

      You can upload them directly on GitHub.
      Beware that video file size is limited to 10MB.
-->
