/* Built-in chat UI. No framework, no bundler, no CDN: this file is served
   from the same process as the model, so it has to work offline. */

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const icon = (name, cls) => {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", cls ? `ico ${cls}` : "ico");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
};

const state = {
  config: null,
  sessionId: null,
  mode: "chat",
  workspace: "",
  streaming: false,
  controller: null,
  attachments: [],
  provider: "local",     // which endpoint answers this chat
  model: "",
  settings: {},          // filled from the server's defaults, then localStorage
};

const SUGGESTIONS = {
  chat: [
    ["globe", "What changed in local LLM tooling this month?"],
    ["bolt", "Explain KV cache quantization in plain terms"],
    ["copy", "Rewrite this paragraph to be half as long"],
  ],
  agent: [
    ["folder", "Summarise every file in this folder"],
    ["bolt", "Find the TODOs in the code here"],
    ["shield", "Write a README for this project"],
  ],
};

/* ----------------------------------------------------------- markdown -- */

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function inline(s) {
  return s
    .replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (whole, label, href) =>
      (/^(https?:|mailto:|[./#])/i.test(href.trim())
        ? `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`
        : `${label} (${href})`))       /* javascript:, data:, file: stay inert */
    .replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g,
             '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>');
}

function codeBlock(lang, code) {
  return `<div class="code-wrap"><div class="code-head"><span>${lang || "code"}</span>` +
    '<button class="copy-btn" type="button">' +
    '<svg class="ico"><use href="#i-copy"/></svg>copy</button></div>' +
    `<pre><code>${code}</code></pre></div>`;
}

function markdown(src) {
  const blocks = [];
  let text = escapeHtml(src || "").replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    blocks.push(codeBlock(lang, code.replace(/\n$/, "")));
    return `@@CB${blocks.length - 1}@@`;
  });

  const open = text.match(/```(\w*)\n?([\s\S]*)$/);
  if (open) {
    blocks.push(codeBlock(open[1], open[2].replace(/\n$/, "")));
    text = text.slice(0, open.index) + `@@CB${blocks.length - 1}@@`;
  }

  const lines = text.split("\n");
  const out = [];
  let list = null, para = [], quote = [];

  const flushPara = () => {
    if (para.length) { out.push(`<p>${inline(para.join(" "))}</p>`); para = []; }
  };
  const flushList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  const flushQuote = () => {
    if (quote.length) {
      out.push(`<blockquote>${inline(quote.join(" "))}</blockquote>`); quote = [];
    }
  };
  const flushAll = () => { flushPara(); flushList(); flushQuote(); };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^@@CB\d+@@$/.test(line.trim())) { flushAll(); out.push(line.trim()); continue; }
    if (!line.trim()) { flushAll(); continue; }

    const head = line.match(/^(#{1,6})\s+(.*)$/);
    if (head) { flushAll(); out.push(`<h${head[1].length}>${inline(head[2])}</h${head[1].length}>`); continue; }
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(line.trim())) { flushAll(); out.push("<hr>"); continue; }
    if (/^&gt;\s?/.test(line)) { flushPara(); flushList(); quote.push(line.replace(/^&gt;\s?/, "")); continue; }

    if (line.includes("|") && /^\s*\|?[-: |]+\|[-: |]+$/.test(lines[i + 1] || "")) {
      flushAll();
      const cells = (row) => row.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      let html = "<table><thead><tr>" +
        cells(line).map((c) => `<th>${inline(c)}</th>`).join("") + "</tr></thead><tbody>";
      i += 2;
      while (i < lines.length && lines[i].includes("|")) {
        html += "<tr>" + cells(lines[i]).map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>";
        i++;
      }
      i--;
      out.push(html + "</tbody></table>");
      continue;
    }

    const item = line.match(/^\s*([-*+]|\d+[.)])\s+(.*)$/);
    if (item) {
      flushPara(); flushQuote();
      const kind = /\d/.test(item[1]) ? "ol" : "ul";
      if (list !== kind) { flushList(); out.push(`<${kind}>`); list = kind; }
      out.push(`<li>${inline(item[2])}</li>`);
      continue;
    }
    flushList(); flushQuote();
    para.push(line.trim());
  }
  flushAll();
  // a model writing about this file can emit the placeholder itself
  return out.join("\n").replace(/@@CB(\d+)@@/g,
                                (whole, n) => (blocks[n] === undefined ? whole : blocks[n]));
}

/* --------------------------------------------------------------- api --- */

/* Every request carries this marker. A page on another site cannot set a
   custom header without a preflight, and the server has no CORS headers to
   pass one - so this is what makes cross-site POSTs to the agent impossible. */
const UI_HEADERS = { "Content-Type": "application/json", "X-Simplex-UI": "1" };

async function api(path, options) {
  const opts = { ...(options || {}) };
  opts.headers = { ...UI_HEADERS, ...(opts.headers || {}) };
  const res = await fetch(path, opts);
  if (!res.ok) {
    let message = res.statusText;
    try { message = (await res.json()).error.message; } catch (e) { /* not JSON */ }
    throw new Error(message);
  }
  return res.json();
}

function toast(message, bad) {
  const box = el("div", `toast${bad ? " bad" : ""}`);
  box.append(icon(bad ? "shield" : "check"), el("span", null, message));
  $("#toasts").append(box);
  setTimeout(() => {
    box.classList.add("out");
    setTimeout(() => box.remove(), 260);
  }, 2600);
}

/* ------------------------------------------------------------ layout --- */

const compact = () => matchMedia("(max-width: 900px)").matches;

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode").forEach((b) => {
    const on = b.dataset.mode === mode;
    b.classList.toggle("on", on);
    b.setAttribute("aria-selected", String(on));
  });
  $(".modes").classList.toggle("agent", mode === "agent");
  $("#ws-change").hidden = mode !== "agent";
  showHint();
  renderChips();
}

function showHint(extra) {
  const hint = $("#hint");
  hint.replaceChildren();
  if (extra) {
    hint.append(el("span", null, extra));
    return;
  }
  const tools = (state.config?.tools?.[state.mode] || []).map((t) => t.name);
  hint.append(el("span", null, state.mode === "agent"
    ? "Agent works inside the folder above and asks before writing or running"
    : "Chat searches the web when the answer could have changed"));
  if (tools.length) hint.append(el("span", "k", `${tools.length} tools`));
  // The Enter / Shift+Enter chips lived here and in the placeholder directly
  // above them. Everybody already knows how a chat box works; saying it twice
  // is what makes a UI feel fussy. It stays as the textarea's title for anyone
  // who does want it.
}

/* The context window of whatever this chat is actually talking to. For the
   kit's own server that is what it loaded; for a provider it is whatever the
   user typed when adding it, because no endpoint reports its own window and a
   remote model's has nothing to do with this card. Null when unknown, which is
   the one case where hiding the meter is honest. */
function activeContextLength() {
  if (state.provider && state.provider !== "local") {
    const p = (state.config?.providers || []).find((x) => x.id === state.provider);
    return p?.context_length || null;
  }
  return state.config?.context_length || null;
}

/* Which endpoint this chat talks to. The kit's own server is "local"; anything
   else is a provider the user configured, and switching to it needs no
   restart because nothing is loaded into this card. */
function setEndpoint(provider, model, providerName) {
  state.provider = provider || "local";
  state.model = model || "";
  const row = $("#stat-model");
  const remote = state.provider !== "local";
  // the chip carries an icon and a chevron either side of the name, so only
  // the name node is rewritten - textContent on the button would eat both
  ($("#stat-model-name") || row).textContent = model || state.config?.model || "-";
  row.title = remote ? `${providerName || state.provider} - ${model}`
    : (state.config?.can_switch_model ? "Switch model" : model || "");
  $("#stat-remote").hidden = !remote;
  $("#stat-remote").textContent = remote ? (providerName || state.provider) : "";
  // A remote model has its own window, so the local number would be a lie - but
  // if the user told us what it is when adding the provider, use that instead of
  // giving up and showing a dash.
  refreshAttachButton();
  const total = activeContextLength();
  if (total) {
    $("#stat-context").textContent = `${Math.round(total / 1024)}k tokens`;
  } else {
    $("#stat-context").textContent = "-";
    $("#context-meter").hidden = true;
  }
}

function shortPath(path, keep = 2) {
  const parts = String(path).split(/[\\/]+/).filter(Boolean);
  if (parts.length <= keep) return path;
  return `...${path.includes("\\") ? "\\" : "/"}${parts.slice(-keep).join(
    path.includes("\\") ? "\\" : "/")}`;
}

function setWorkspace(path) {
  state.workspace = path;
  $("#ws-path").textContent = shortPath(path);
  $("#ws-change").title = `Agent folder: ${path}`;
}

/* Set while a stored conversation is being rebuilt. Messages created in that
   window are marked .still and skip every entry animation: they are not
   arriving, they are being put back. */
let restoring = false;

function newMessage(role) {
  $("#welcome")?.remove();
  const wrap = el("div", `msg ${role}${restoring ? " still" : ""}`);
  const who = el("div", "who");
  who.append(el("span", "avatar", role === "user" ? "You"[0] : "AI"),
             el("span", null, role === "user" ? "You" : "Assistant"));
  wrap.append(who);
  const body = el("div", "body");
  wrap.append(body);
  $("#thread").append(wrap);
  return body;
}

function nearBottom() {
  const t = $("#thread");
  return t.scrollHeight - t.scrollTop - t.clientHeight < 180;
}

function scrollDown(force, instant) {
  const t = $("#thread");
  if (!force && !nearBottom()) return;
  // during a stream the smooth animation never catches up with the new
  // content, and the measurement of "near the bottom" drifts with it.
  // `instant` is for a conversation being restored: it was already at the
  // bottom when you left it, so scrolling there is a transition to nowhere.
  t.style.scrollBehavior = (instant || state.streaming) ? "auto" : "";
  t.scrollTop = t.scrollHeight;
}

/* --------------------------------------------------------- rendering --- */

function typingDots(container) {
  if (container.querySelector(".typing")) return;
  const dots = el("div", "typing");
  dots.append(el("i"), el("i"), el("i"));
  container.append(dots);
}

/* Whether thinking blocks start open. It is the user's choice, remembered:
   toggling any one of them sets the preference for the next. Defaults to open,
   because someone who has never seen the model think should see it once. */
function thinkOpenPref() {
  try { return localStorage.getItem("chatui.thinkOpen") !== "0"; }
  catch (e) { return true; }
}

function setThinkOpenPref(open) {
  try { localStorage.setItem("chatui.thinkOpen", open ? "1" : "0"); }
  catch (e) { /* private mode: it just will not persist */ }
}

/* A thinking pane that does not follow its own text is a pane you cannot read:
   the model writes at the bottom while you sit at the top watching a scrollbar
   shrink. So it sticks to the tail - until you scroll up, which means you are
   reading something and want to be left alone. Scrolling back to the end opts
   in again. */
function followThink(pane) {
  if (!pane || pane.dataset.stuck === "0") return;
  pane.scrollTop = pane.scrollHeight;
}

function thinkBlock(container, live) {
  let node = container.querySelector(".think");
  if (!node) {
    node = el("details", "think");
    node.open = thinkOpenPref();
    const sum = el("summary");
    sum.append(icon("down", "chev"), el("span", "label", "Thinking..."));
    const pane = el("div", "think-body");
    pane.dataset.stuck = "1";
    pane.addEventListener("scroll", () => {
      pane.dataset.stuck =
        pane.scrollHeight - pane.scrollTop - pane.clientHeight < 24 ? "1" : "0";
    });
    node.append(sum, pane);
    // remember which way the user left it, for every block after this one
    node.addEventListener("toggle", () => {
      if (node.dataset.settling) return;      // not a click, just the code
      setThinkOpenPref(node.open);
    });
    node.dataset.started = String(Date.now());
    container.prepend(node);
  }
  node.classList.toggle("live", !!live);
  return node.querySelector(".think-body");
}

/* The block stops being live: name what it did and how long it took, so it
   reads as a thing you can open rather than a leftover spinner. The open or
   closed state is the user's - it used to be slammed shut on every reply, so
   opening one was pointless. */
function settleThinkBlock(body) {
  const node = body?.querySelector?.(".think") || body;
  if (!node || !node.classList?.contains("think")) return;
  node.classList.remove("live");
  const label = node.querySelector(".label");
  if (label) {
    const started = Number(node.dataset.started || 0);
    const secs = started ? Math.max(1, Math.round((Date.now() - started) / 1000)) : 0;
    label.textContent = secs ? `Thought for ${secs}s` : "Thinking";
  }
  node.dataset.settling = "1";
  node.open = thinkOpenPref();
  delete node.dataset.settling;
  // it stopped being a live tail and became something to read from the start -
  // but only if the reader was not already somewhere of their own choosing
  const pane = node.querySelector(".think-body");
  if (pane && pane.dataset.stuck !== "0") pane.scrollTop = 0;
}

/* Re-rendering the whole answer on every token is quadratic and destroys the
   user's selection; once a frame is indistinguishable and cheap. The raw
   Markdown lives beside the node, not in a DOM attribute. */
const RAW = new WeakMap();

function rawOf(node) {
  return RAW.get(node) ?? node.dataset.raw ?? node.textContent ?? "";
}

function appendDelta(node, delta) {
  RAW.set(node, rawOf(node) + delta);
  node.classList.add("live");
  if (node.__frame) return;
  node.__frame = requestAnimationFrame(() => {
    node.__frame = 0;
    node.innerHTML = markdown(rawOf(node));
    scrollDown();
  });
}

function flushContent(body) {
  body.querySelectorAll(".stream-content").forEach((node) => {
    if (node.__frame) { cancelAnimationFrame(node.__frame); node.__frame = 0; }
    node.innerHTML = markdown(rawOf(node));
    node.classList.remove("live");
  });
}

function contentBlock(container) {
  let node = container.querySelector(".stream-content:last-of-type");
  if (!node || node.dataset.closed) {
    node = el("div", "stream-content live");
    container.append(node);
  }
  return node;
}

function toolCard(container, call) {
  const card = el("details", "tool");
  card.dataset.id = call.id;
  const summary = el("summary");
  summary.append(el("span", "name", call.name), el("span", "arg", call.label || ""),
                 el("span", "state run", "running"));
  const io = el("div", "io");
  io.append(el("div", "lbl", "arguments"),
            el("pre", null, JSON.stringify(call.args, null, 2)));
  card.append(summary, io);
  container.append(card);
  return card;
}

function finishToolCard(card, ok, output, ms) {
  if (!card) return;
  const badge = card.querySelector(".state");
  badge.className = `state ${ok ? "ok" : "err"}`;
  badge.textContent = ok ? (ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : "done") : "failed";
  const io = card.querySelector(".io");
  io.append(el("div", "lbl", "result"), el("pre", null, output));
  if (!ok) card.open = true;
}

function renderPlan(items) {
  const bar = $("#plan-bar");
  const list = bar.querySelector(".plan-list");
  if (!items || !items.length) { bar.hidden = true; list.replaceChildren(); return; }
  const done = items.filter((i) => i.status === "completed").length;
  bar.hidden = false;
  bar.querySelector(".plan-count").textContent = `${done}/${items.length}`;
  bar.querySelector(".plan-bar-track i").style.width =
    `${Math.round((done / items.length) * 100)}%`;
  list.replaceChildren();
  items.forEach((item) => {
    const li = el("li", item.status === "completed" ? "done"
      : item.status === "in_progress" ? "now" : "");
    const mark = el("span", "mark");
    if (item.status === "completed") mark.append(icon("check"));
    if (item.status === "in_progress") mark.append(el("i"));
    li.append(mark, el("span", null, item.content));
    list.append(li);
  });
}

/* ask_user: the model needs an answer from the person, not permission. */
function questionCard(container, ev) {
  const box = el("div", "question");
  box.dataset.call = ev.id;
  const head = el("h4");
  head.append(icon("ask"), el("span", null, ev.header || "A question for you"));
  box.append(head, el("div", "q", ev.question));

  const answer = async (text) => {
    box.classList.add("done");
    box.replaceChildren(head, el("div", "q", ev.question),
                        el("div", "verdict yes", `You answered: ${text}`));
    try {
      await fetch("/ui/answer", {
        method: "POST", headers: UI_HEADERS,
        body: JSON.stringify({ id: ev.id, text }),
      });
    } catch (e) { toast("Could not send that answer", true); }
  };

  if (ev.options?.length) {
    const options = el("div", "options");
    ev.options.forEach((option) => {
      const btn = el("button", "opt");
      btn.append(el("span", null, option.label));
      if (option.description) btn.append(el("small", null, option.description));
      btn.onclick = () => answer(option.label);
      options.append(btn);
    });
    box.append(options);
  }
  const free = el("div", "free");
  const input = el("input");
  input.placeholder = ev.options?.length ? "or type your own answer"
    : "type your answer";
  const send = el("button", "btn primary", "Send");
  const submit = () => { if (input.value.trim()) answer(input.value.trim()); };
  send.onclick = submit;
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); });
  free.append(input, send);
  box.append(free);
  container.append(box);
  scrollDown(true);
  setTimeout(() => input.focus(), 60);
}

function markResolved(body, id, label) {
  body.querySelectorAll(`[data-call="${id}"]`).forEach((card) => {
    if (card.classList.contains("done")) return;
    card.classList.add("done");
    const row = card.querySelector(".row") || card.querySelector(".free");
    if (row) {
      const verdict = el("div", "verdict yes");
      verdict.append(icon("check"), el("span", null, label));
      row.replaceChildren(verdict);
    }
    card.querySelectorAll(".options .opt").forEach((b) => { b.disabled = true; });
  });
}

function approvalCard(container, ev, cards) {
  /* The request belongs to a tool call that is already on screen, so it goes
     inside that card - otherwise the result would appear above the question
     that produced it. */
  const card = cards?.get(ev.id);
  if (card) {
    card.open = true;
    container = card.querySelector(".io");
  }
  const box = el("div", "approval");
  box.dataset.call = ev.id;
  const head = el("h4");
  head.append(icon("shield"),
    el("span", null, ev.risk === "exec"
      ? "The agent wants to run something on this machine"
      : "The agent wants to change a file"));
  box.append(head, el("div", "sub", `${ev.name}  ${ev.label || ""}`),
             el("pre", null, JSON.stringify(ev.args, null, 2)));

  const row = el("div", "row");
  const decide = async (decision, label, yes) => {
    box.classList.add("done");
    const verdict = el("div", `verdict ${yes ? "yes" : "no"}`);
    verdict.append(icon(yes ? "check" : "trash"), el("span", null, label));
    row.replaceChildren(verdict);
    try {
      await fetch("/ui/approve", {
        method: "POST", headers: UI_HEADERS,
        body: JSON.stringify({ id: ev.id, decision }),
      });
    } catch (e) { toast("Could not send that decision", true); }
  };
  const allow = el("button", "btn primary", "Allow once");
  allow.onclick = () => decide("allow", "Allowed", true);
  const always = el("button", "btn", `Always allow ${ev.name}`);
  always.onclick = () => decide("always",
    `Allowed - ${ev.name} will not ask again this session`, true);
  const deny = el("button", "btn danger", "Deny");
  deny.onclick = () => decide("deny", "Denied", false);
  row.append(allow, always, deny);
  box.append(row);
  container.append(box);
  scrollDown(true);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (err) {
    const tmp = el("textarea");
    tmp.value = text;
    document.body.append(tmp);
    tmp.select();
    try { document.execCommand("copy"); } catch (e) { /* nothing else to try */ }
    tmp.remove();
  }
}

async function consumeStream(res, body, cards, stats) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop();
    for (const part of parts) {
      const line = part.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      let event;
      try {
        event = JSON.parse(line.slice(5).trim());
      } catch (err) { continue; }        // one bad frame is not the whole reply
      handleEvent(event, body, cards, stats);
    }
  }
}

function finishTurn(body, stats) {
  setStreaming(false);
  $("#speed").classList.remove("live");
  body.querySelector(".typing")?.remove();
  const think = body.querySelector(".think");
  if (think) settleThinkBlock(think);
  flushContent(body);
  announce("Reply finished");
  messageActions(body, stats);
  refreshEditAction();
  loadSessions();
}

/* The turn belongs to the conversation, not to this window. Reopening a chat
   that is still working replays it from the beginning and carries on live. */
async function attachTurn(sid) {
  let res;
  try {
    res = await fetch("/ui/attach", {
      method: "POST", headers: UI_HEADERS,
      body: JSON.stringify({ session_id: sid, from: 0 }),
    });
  } catch (e) { return false; }
  if (!res.ok) return false;

  const body = newMessage("assistant");
  typingDots(body);
  const cards = new Map();
  const stats = {};
  setStreaming(true);
  rememberEndpoint(state.provider, state.model);
  state.controller = new AbortController();
  try {
    await consumeStream(res, body, cards, stats);
  } catch (e) {
    if (e.name !== "AbortError") body.append(el("div", "err-box", String(e.message || e)));
  } finally {
    finishTurn(body, stats);
  }
  return true;
}

/* Closing a chat leaves its turn running; only Stop ends one. */
function detach() {
  state.controller?.abort();
  setStreaming(false);
}

/* Drop the last exchange server-side, then either resend it (retry) or put it
   back in the composer (edit). */
async function rewind() {
  if (!state.sessionId || state.streaming) return null;
  let got;
  try {
    got = await api("/ui/rewind", {
      method: "POST", headers: UI_HEADERS,
      body: JSON.stringify({ session_id: state.sessionId }),
    });
  } catch (e) { toast(e.message, true); return null; }
  // The server drops everything from the last question onward, which is more
  // than two bubbles once tools are involved - so re-render what it kept
  // instead of guessing.
  try {
    const session = await api(`/ui/sessions/${state.sessionId}`);
    renderStoredMessages(session.messages || []);
  } catch (e) { /* the transcript is already correct server-side */ }
  return got.content;
}

async function retryLast() {
  const content = await rewind();
  if (content != null) send(content);
}

/* A finished answer gets a quiet action row - visible on hover only, so it
   never competes with the text. */
/* Only the newest question can be edited - rewinding further would throw away
   answers the user may still want - so exactly one Edit button exists. */
function refreshEditAction() {
  document.querySelectorAll(".msg.user .msg-actions").forEach((n) => n.remove());
  const last = [...document.querySelectorAll(".msg.user")].pop();
  if (!last || state.streaming) return;
  const body = last.querySelector(".body");
  const row = el("div", "msg-actions");

  // Icon-only, with a real label for anyone who cannot see the icon.
  const act = (name, label, run) => {
    const b = el("button", "act icon-only");
    b.type = "button";
    b.append(icon(name));
    b.title = label;
    b.setAttribute("aria-label", label);
    b.onclick = run;
    row.append(b);
    return b;
  };

  const copy = act("copy", "Copy this message", async () => {
    await copyText(messageText(last));
    copy.classList.add("done");
    copy.replaceChildren(icon("check"));
    setTimeout(() => {
      copy.classList.remove("done");
      copy.replaceChildren(icon("copy"));
    }, 1600);
  });

  act("pencil", "Edit this message and ask again", () => startMessageEdit(last));
  body.append(row);
}

/* The text of a message bubble, without the action row. */
function messageText(msg) {
  const body = msg.querySelector(".body");
  if (!body) return "";
  const clone = body.cloneNode(true);
  clone.querySelectorAll(".msg-actions, .edit-box").forEach((n) => n.remove());
  return clone.innerText.trim();
}

/* Edit in place.

   This used to rewind immediately and drop the text into the composer at the
   bottom of the window: the question vanished from the transcript, the input
   far below quietly filled, and from where the user was looking nothing had
   happened - or their message had just been eaten. Nothing is destroyed here
   until Save is pressed, and the editing happens where the message is. */
function startMessageEdit(msg) {
  if (msg.querySelector(".edit-box")) return;
  const body = msg.querySelector(".body");
  const original = messageText(msg);
  const keep = [...body.childNodes];
  const box = el("div", "edit-box");
  const area = el("textarea", "edit-area");
  area.value = original;
  area.rows = Math.min(12, Math.max(2, original.split("\n").length + 1));

  const bar = el("div", "edit-bar");
  const hint = el("span", "edit-hint", "Sending again replaces the answer below");
  const cancel = el("button", "btn small", "Cancel");
  cancel.type = "button";
  const save = el("button", "btn small primary", "Save and resend");
  save.type = "button";
  bar.append(hint, cancel, save);
  box.append(area, bar);

  const close = () => {
    box.remove();
    keep.forEach((n) => body.append(n));
    refreshEditAction();
  };
  cancel.onclick = close;
  area.onkeydown = (e) => {
    e.stopPropagation();                     // the composer listens globally
    if (e.key === "Escape") close();
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) save.click();
  };
  save.onclick = async () => {
    const text = area.value.trim();
    if (!text) { toast("An empty message cannot be sent", true); return; }
    save.disabled = cancel.disabled = true;
    // only now is anything thrown away
    const got = await rewind();
    if (got == null) { save.disabled = cancel.disabled = false; return; }
    send(text);
  };

  keep.forEach((n) => n.remove());
  body.append(box);
  area.focus();
  area.setSelectionRange(area.value.length, area.value.length);
}

function messageActions(body, stats) {
  if (body.querySelector(".msg-actions")) return;
  const raw = [...body.querySelectorAll(".stream-content")]
    .map(rawOf).join("\n\n").trim();
  if (!raw) return;
  const row = el("div", "msg-actions");
  const copy = el("button", "act");
  copy.append(icon("copy"), el("span", null, "Copy"));
  copy.onclick = async () => {
    await copyText(raw);
    copy.classList.add("done");
    copy.replaceChildren(icon("check"), el("span", null, "Copied"));
    setTimeout(() => {
      copy.classList.remove("done");
      copy.replaceChildren(icon("copy"), el("span", null, "Copy"));
    }, 1600);
  };
  row.append(copy);
  const retry = el("button", "act");
  retry.append(icon("retry"), el("span", null, "Retry"));
  retry.title = "Answer this again";
  retry.onclick = () => {
    if (body.closest(".msg") === $("#thread").lastElementChild) retryLast();
    else toast("Only the last answer can be retried");
  };
  row.append(retry);
  if (stats?.usage?.completion_tokens) {
    const bits = [`${stats.usage.completion_tokens.toLocaleString()} tokens`];
    if (stats.tok_s) bits.push(`${stats.tok_s} tok/s`);
    if (stats.seconds) bits.push(`${stats.seconds}s`);
    row.append(el("span", "msg-stat", bits.join("  ·  ")));
  }
  body.append(row);
}

/* ------------------------------------------------------------ sending -- */

const TEXT_EXT = /\.(txt|md|markdown|csv|tsv|json|ya?ml|toml|ini|cfg|log|sql|py|js|ts|jsx|tsx|html?|css|sh|bat|ps1|rs|go|java|c|h|cpp|rb|php|xml|env)$/i;
const MAX_DOC_CHARS = 120000;

function isTextFile(file) {
  return file.type.startsWith("text/") || TEXT_EXT.test(file.name)
    || file.type === "application/json";
}

function userContent(text) {
  const docs = state.attachments.filter((a) => a.kind === "text");
  const images = state.attachments.filter((a) => a.kind === "image");
  let body = text;
  if (docs.length) {
    body += "\n\n" + docs.map((d) =>
      `File: ${d.name}\n\`\`\`\n${d.text}\n\`\`\``).join("\n\n");
  }
  if (!images.length) return body;
  return [{ type: "text", text: body },
    ...images.map((a) => ({ type: "image_url", image_url: { url: a.url } }))];
}

function contentText(content) {
  if (typeof content === "string") return content;
  return (content || []).filter((p) => p.type === "text")
    .map((p) => p.text).join("\n");
}

/* Which reasoning levels the endpoint in use will accept.

   These differ per model - low/medium/high on most, plus xhigh or max on some -
   so they come from what the endpoint said when it was tested, not from a list
   invented here. An endpoint that says nothing gets no levels, and the UI says
   so rather than offering switches that do nothing. "off" is separate: it is
   done in the prompt template, so it works regardless of what the server takes.
*/
function availableEfforts() {
  if (state.provider && state.provider !== "local") {
    const p = (state.config?.providers || []).find((x) => x.id === state.provider);
    return p?.efforts || [];
  }
  return state.config?.efforts || [];
}

/* Whether the endpoint in use accepts images.

   This used to read the local server's vision flag no matter who the chat was
   talking to, so a provider with a perfectly good vision model had the attach
   button hidden because this machine's own tower had not loaded. The user's
   answer wins over the probe's, because plenty of OpenAI-compatible servers
   publish no modality information at all. */
/* The endpoint a new conversation starts on.

   Without this, a new chat inherited whichever endpoint you happened to be on
   last, which is fine until the one you actually want is not the one you last
   opened. Stored as provider+model so it survives a restart, and falls back to
   the local server whenever the saved choice no longer exists - a provider can
   be deleted, and a menu that points at nothing is worse than a default. */
/* "Whatever I used last" has to actually mean that. It used to fall through to
   the local server, so a new chat - and every restart - quietly put you back on
   this machine's model even though you had spent the afternoon on a provider.
   Kept per browser, like the theme, and written only when you *use* an endpoint:
   picking one, or sending a turn on it. Opening an old conversation to read it
   does not count, or browsing your history would rewrite the default. */
function rememberEndpoint(provider, model, name) {
  try {
    localStorage.setItem("chatui.lastEndpoint", JSON.stringify({
      provider: provider || "local", model: model || "", name: name || "" }));
  } catch (e) { /* private mode: it just will not persist */ }
}

function lastEndpoint() {
  try {
    const was = JSON.parse(localStorage.getItem("chatui.lastEndpoint") || "null");
    return was && was.provider ? was : null;
  } catch (e) { return null; }
}

function defaultEndpoint() {
  // an explicit setting wins; with none, fall back to the last one used
  const want = state.settings?.defaultEndpoint || lastEndpoint();
  if (!want || !want.provider) return null;
  if (want.provider === "local") {
    return { provider: "local", model: state.config?.model || "", name: "" };
  }
  const p = (state.config?.providers || []).find((x) => x.id === want.provider);
  if (!p) return null;
  return { provider: p.id, model: want.model || p.default_model, name: p.name };
}

function applyDefaultEndpoint() {
  const d = defaultEndpoint();
  if (d) setEndpoint(d.provider, d.model, d.name);
  else setEndpoint("local", state.config?.model || "");
}

/* Every endpoint a new chat could start on, local first. */
function endpointChoices() {
  const out = [{ provider: "local", model: state.config?.model || "",
                 label: `This computer - ${state.config?.model || "local model"}` }];
  (state.config?.providers || []).forEach((p) => {
    out.push({ provider: p.id, model: p.default_model,
               label: `${p.name} - ${p.default_model}` });
  });
  return out;
}

function activeVision() {
  if (state.provider && state.provider !== "local") {
    const p = (state.config?.providers || []).find((x) => x.id === state.provider);
    if (!p) return false;
    if (p.vision !== null && p.vision !== undefined) return !!p.vision;
    return !!p.vision_detected;
  }
  return !!state.config?.vision;
}

/* Keep the composer in step with whatever the chat is pointed at. */
/* Replace the server config and re-sync everything derived from it.

   Anything that reads state.config has to be told when it changes, or it keeps
   showing the old answer. Turning Images on for the provider you are already
   chatting with saved correctly and changed nothing on screen - the attach
   button is only recomputed when the endpoint switches - so the setting looked
   like it had not saved at all. */
async function reloadConfig() {
  state.config = await api("/ui/config");
  refreshAttachButton();
  showContext(state.lastPromptTokens);
  const total = activeContextLength();
  $("#stat-context").textContent = total
    ? `${Math.round(total / 1024)}k tokens` : "-";
  return state.config;
}

function refreshAttachButton() {
  const attach = $("#attach");
  if (attach) attach.hidden = !activeVision();
}

const EFFORT_LABELS = {
  minimal: "minimal", low: "low", medium: "medium",
  high: "high", xhigh: "extra high", max: "max",
};

/* Typed commands.

   Settings has the same switches, but a setting you have to open a panel to
   change is a setting you forget exists - and thinking effort is something you
   want to change for one question, not once a month. Commands are handled
   entirely in the browser: nothing is sent to the model, so a typo costs a
   toast rather than a turn.

   Each entry: run(args) -> a line to show, or null if it handled its own. */
const COMMANDS = {
  effort: {
    help: "/effort [off|<level>|default] - how hard the model thinks",
    run: (arg) => {
      const levels = availableEfforts();
      const choices = ["off", ...levels, "default"];
      const now = state.settings.thinking || "default";
      const shown = now === "default" ? "the model's default"
        : (now === "off" ? "off" : (EFFORT_LABELS[now] || now));
      if (!arg) {
        return levels.length
          ? `Thinking is ${shown}. This endpoint takes: ${choices.join(", ")}.`
          : `Thinking is ${shown}. This endpoint did not say which levels it `
            + "takes, so only /effort off and /effort default are reliable. "
            + "Press Test on the provider to ask it again.";
      }
      const want = arg.toLowerCase();
      if (want === "default" || want === "auto" || want === "reset") {
        delete state.settings.thinking;
        saveSettings();
        return "Thinking left to the model to decide, as it was before.";
      }
      if (want === "off" || want === "none") {
        state.settings.thinking = "off";
        saveSettings();
        return "Thinking off. The model answers straight away - this one is exact.";
      }
      if (!levels.includes(want)) {
        return levels.length
          ? `This endpoint does not offer "${arg}". It takes: ${choices.join(", ")}.`
          : `This endpoint did not say it takes "${arg}". Only off and default `
            + "are reliable here.";
      }
      state.settings.thinking = want;
      saveSettings();
      return `Thinking set to ${EFFORT_LABELS[want] || want}. This is a request, `
           + "not a limit: the model decides how much it actually needs.";
    },
  },
  help: {
    help: "/help - list these commands",
    run: () => Object.values(COMMANDS).map((c) => c.help).join("\n"),
  },
};

/* True if the text was a command and has been dealt with. */
function runCommand(text) {
  const m = /^\/([a-z]+)\s*(.*)$/i.exec(text.trim());
  if (!m) return false;
  const cmd = COMMANDS[m[1].toLowerCase()];
  if (!cmd) {
    toast(`No command called /${m[1]}. Try /help.`, true);
    return true;
  }
  const said = cmd.run(m[2].trim());
  if (said) toast(said);
  return true;
}

async function send(preset) {
  const input = $("#input");
  const ready = preset && typeof preset !== "string";       // a retry payload
  const text = ready ? contentText(preset) : (preset || input.value).trim();
  if (!text || state.streaming) return;
  // a command is for this page, not for the model
  if (!ready && text.startsWith("/") && runCommand(text)) {
    input.value = "";
    input.style.height = "auto";
    return;
  }
  if (!preset) {                 // a suggestion chip leaves the draft alone
    input.value = "";
    input.style.height = "auto";
  }

  const content = ready ? preset : userContent(text);
  const sent = newMessage("user");
  renderUserBody(sent, content);
  state.attachments = [];
  $("#attachments").replaceChildren();
  scrollDown(true);

  const body = newMessage("assistant");
  typingDots(body);
  announce("Working on a reply");
  const cards = new Map();
  const stats = {};
  setStreaming(true);
  rememberEndpoint(state.provider, state.model);
  state.controller = new AbortController();

  try {
    const res = await fetch("/ui/chat", {
      method: "POST",
      headers: UI_HEADERS,
      signal: state.controller.signal,
      body: JSON.stringify({
        session_id: state.sessionId, mode: state.mode, content,
        workspace: state.workspace, settings: state.settings,
        provider: state.provider, model: state.model,
      }),
    });
    if (!res.ok) {
      let message = `HTTP ${res.status}`;
      try { message = (await res.json()).error.message; } catch (e) { /* not JSON */ }
      if (res.status === 409) {           // a turn is already running here
        body.closest(".msg").remove();
        toast("This conversation is still working - watching it instead");
        await attachTurn(state.sessionId);
        return;
      }
      throw new Error(message);
    }

    await consumeStream(res, body, cards, stats);
  } catch (err) {
    if (err.name !== "AbortError") {
      body.querySelector(".typing")?.remove();
      body.append(el("div", "err-box", String(err.message || err)));
      scrollDown(true);
    }
  } finally {
    finishTurn(body, stats);
  }
}

function handleEvent(ev, body, cards, stats) {
  switch (ev.type) {
    case "session":
      state.sessionId = ev.id;
      if (ev.workspace) setWorkspace(ev.workspace);
      if (ev.provider) setEndpoint(ev.provider, ev.model, ev.provider_name);
      break;
    case "reasoning": {
      body.querySelector(".typing")?.remove();
      const pane = thinkBlock(body, true);
      pane.append(document.createTextNode(ev.delta));
      followThink(pane);
      scrollDown();
      break;
    }
    case "content": {
      body.querySelector(".typing")?.remove();
      body.querySelector(".think")?.classList.remove("live");
      appendDelta(contentBlock(body), ev.delta);
      break;
    }
    case "tool_call": {
      body.querySelector(".typing")?.remove();
      const open = body.querySelector(".stream-content:last-of-type");
      if (open) { open.dataset.closed = "1"; open.classList.remove("live"); }
      body.querySelector(".think")?.classList.remove("live");
      cards.set(ev.id, toolCard(body, ev));
      scrollDown();
      break;
    }
    case "approval":
      approvalCard(body, ev, cards);
      break;
    case "question": {
      body.querySelector(".typing")?.remove();
      const card = cards.get(ev.id);
      questionCard(card ? card.querySelector(".io") : body, ev);
      if (card) card.open = true;
      break;
    }
    case "plan":
      renderPlan(ev.items);
      break;
    case "tool_result":
      finishToolCard(cards.get(ev.id), ev.ok, ev.output, ev.ms);
      // if this was replayed, its card may still be showing buttons
      markResolved(body, ev.id, "Answered");
      typingDots(body);
      scrollDown();
      break;
    case "error":
      body.querySelector(".typing")?.remove();
      body.append(el("div", "err-box", ev.message));
      scrollDown(true);
      break;
    case "done":
      if (stats) Object.assign(stats, ev);
      showContext(ev.usage?.prompt_tokens);
      notifyIfAway();
      if (ev.tok_s) showSpeed(ev.tok_s, false);
      if (ev.usage?.completion_tokens) {
        const rate = ev.tok_s ? `  ·  ${ev.tok_s} tok/s` : "";
        showHint(`${ev.usage.prompt_tokens.toLocaleString()} prompt + `
          + `${ev.usage.completion_tokens.toLocaleString()} generated tokens${rate}`);
        setTimeout(() => { if (!state.streaming) showHint(); }, 8000);
      }
      break;
    default:
      break;
  }
}

function setStreaming(on) {
  state.streaming = on;
  /* A turn just started: start sampling now rather than waiting out the idle
     interval, or the whole answer can finish before the first busy poll. */
  if (on) { sample = null; scheduleSpeedSample(120); }
  else clearTimeout(speedTimer);
  $("#send").hidden = on;
  $("#stop").hidden = !on;
  $(".brand-mark i").classList.toggle("busy", on);
}

async function stop() {
  if (!state.sessionId) { state.controller?.abort(); setStreaming(false); return; }
  try {
    await fetch("/ui/cancel", {
      method: "POST", headers: UI_HEADERS,
      body: JSON.stringify({ session_id: state.sessionId }),
    });
  } catch (e) { /* the stream is going away anyway */ }
  state.controller?.abort();
  setStreaming(false);
}

/* ----------------------------------------------------------- sessions -- */

/* The time on a sidebar row. Inside today that is a clock time, further back a
   weekday or a date - what a row needs to be told apart from the fourteen like
   it. The group headers say which day; this says which one within it. */
function rowTime(seconds) {
  if (!seconds) return "";
  const then = new Date(seconds * 1000);
  const now = new Date();
  if (then.toDateString() === now.toDateString()) {
    return then.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }
  if ((now - then) / 86400000 < 7) {
    return then.toLocaleDateString([], { weekday: "short" });
  }
  return then.toLocaleDateString([], { month: "short", day: "numeric" });
}

/* The list is grouped by what a conversation *is*, not by when it happened:
   Agent chats have tool calls, approvals and a folder, and mixing them into
   the same run of rows as ordinary chats is what made the per-row "AGENT"
   badge necessary in the first place. With a section for each, the badge is
   redundant and gone. Day headings went with it - every row already carries
   its own time on the right, and two levels of heading in a 292px column is
   one too many. */
const SESSION_GROUPS = [
  { key: "pinned", label: "Pinned", pick: (s) => !!s.pinned },
  { key: "chat", label: "Chats", pick: (s) => !s.pinned && s.mode !== "agent" },
  { key: "agent", label: "Agentic", pick: (s) => !s.pinned && s.mode === "agent" },
];

function closedGroups() {
  try {
    return new Set(JSON.parse(localStorage.getItem("chatui.closedGroups") || "[]"));
  } catch (e) { return new Set(); }
}

function setGroupClosed(key, closed) {
  const set = closedGroups();
  if (closed) set.add(key); else set.delete(key);
  try { localStorage.setItem("chatui.closedGroups", JSON.stringify([...set])); }
  catch (e) { /* private mode: sections just reopen next time */ }
}

function conversationMarkdown(session) {
  const lines = [`# ${session.title || "Conversation"}`,
                 `_${new Date((session.updated || 0) * 1000).toLocaleString()} `
                 + `· ${session.mode || "chat"} mode_`, ""];
  (session.messages || []).forEach((m) => {
    if (m.role === "user") {
      lines.push("## You", "", contentText(m.content), "");
    } else if (m.role === "assistant") {
      if (m.content) lines.push("## Assistant", "", m.content, "");
      (m.tool_calls || []).forEach((c) => {
        lines.push(`> called \`${c.function.name}\``, "");
      });
    }
  });
  return lines.join("\n");
}

function download(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/markdown" }));
  const a = el("a");
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function exportSession(id, title) {
  try {
    const session = await api(`/ui/sessions/${id}`);
    const slug = (title || "conversation").toLowerCase()
      .replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 50);
    download(`${slug || "conversation"}.md`, conversationMarkdown(session));
    toast("Saved as Markdown");
  } catch (e) { toast(e.message, true); }
}

/* Right-click on a conversation. Everything that used to crowd the row lives
   here now, so the list stays quiet until you ask. */
function openMenu(x, y, items) {
  const menu = $("#menu-pop");
  menu.replaceChildren();
  items.forEach((item) => {
    if (item === "-") { menu.append(el("hr")); return; }
    const button = el("button", item.danger ? "danger" : "");
    button.setAttribute("role", "menuitem");
    button.append(icon(item.icon), el("span", null, item.label));
    button.onclick = () => { closeMenu(); item.run(); };
    menu.append(button);
  });
  menu.hidden = false;
  // keep it on screen
  const box = menu.getBoundingClientRect();
  menu.style.left = `${Math.min(x, window.innerWidth - box.width - 8)}px`;
  menu.style.top = `${Math.min(y, window.innerHeight - box.height - 8)}px`;
  setTimeout(() => menu.querySelector("button")?.focus(), 20);
}

function closeMenu() { $("#menu-pop").hidden = true; }

async function togglePin(session) {
  try {
    const got = await api(`/ui/sessions/${session.id}`, {
      method: "POST", body: JSON.stringify({ pinned: !session.pinned }),
    });
    toast(got.pinned ? "Pinned to the top" : "Unpinned");
    loadSessions();
  } catch (e) { toast(e.message, true); }
}

async function deleteSession(session) {
  await fetch(`/ui/sessions/${session.id}`, { method: "DELETE", headers: UI_HEADERS });
  if (session.id === state.sessionId) newChat(); else loadSessions();
  toast("Conversation deleted");
}

function sessionMenu(event, row, session) {
  event.preventDefault();
  openMenu(event.clientX, event.clientY, [
    { icon: "pencil", label: "Rename", run: () => startRename(row, session) },
    { icon: "pin", label: session.pinned ? "Unpin" : "Pin to top",
      run: () => togglePin(session) },
    { icon: "download", label: "Save as Markdown",
      run: () => exportSession(session.id, session.title) },
    "-",
    { icon: "trash", label: "Delete", danger: true,
      run: () => deleteSession(session) },
  ]);
}

function startRename(row, session) {
  const label = row.querySelector(".t");
  const input = el("input", "rename");
  input.value = session.title;
  let closed = false;
  const finish = async (save) => {
    if (closed) return;          // Enter fires, then blur fires on the same input
    closed = true;
    input.replaceWith(label);
    delete row.dataset.renaming;
    if (!save || !input.value.trim() || input.value === session.title) return;
    try {
      await api(`/ui/sessions/${session.id}`, {
        method: "POST", headers: UI_HEADERS,
        body: JSON.stringify({ title: input.value.trim() }),
      });
      loadSessions();
    } catch (e) { toast(e.message, true); }
  };
  // This input lives inside the row, and the row is a <button>. A button
  // activates on Space, so every space typed here bubbled up, "clicked" the
  // row, and the blur that followed saved the rename - a space behaved exactly
  // like Enter, which made multi-word titles impossible to type. Keep all key
  // events inside the input; Enter and Escape are handled here explicitly.
  const swallow = (e) => e.stopPropagation();
  input.addEventListener("keydown", (e) => {
    swallow(e);
    if (e.key === "Enter") finish(true);
    if (e.key === "Escape") finish(false);
  });
  input.addEventListener("keyup", swallow);
  input.addEventListener("keypress", swallow);
  input.onblur = () => finish(true);
  input.onclick = (e) => e.stopPropagation();
  row.dataset.renaming = "1";
  label.replaceWith(input);
  input.focus();
  input.select();
}

/* Move the "open" marker without touching the rest of the list. Returns false
   if no row carries that id, which is the caller's cue to do a real reload. */
function markActiveSession(id) {
  const rows = $("#sessions").querySelectorAll(".session");
  let found = false;
  rows.forEach((row) => {
    const on = row.dataset.id === id;
    if (on) found = true;
    row.classList.toggle("on", on);
  });
  sessionsSig = "";        // the cached signature no longer matches the DOM
  return found;
}

/* What the rendered list is currently showing. A redraw that would produce the
   same rows is skipped entirely - loadSessions() is called after every turn,
   every rename and every new chat, and rebuilding an identical list just to
   replay its entry animation is the whole of the flicker. */
let sessionsSig = "";

async function loadSessions() {
  let sessions = [];
  const query = ($("#session-filter").value || "").trim().toLowerCase();
  try {
    // the server searches message text too, and caches the index by mtime
    const got = await api(`/ui/sessions${query ? `?q=${encodeURIComponent(query)}` : ""}`);
    sessions = got.sessions || [];
    state.live = got.live || {};
  } catch (e) { return; }
  const nav = $("#sessions");
  const sig = JSON.stringify([query, state.sessionId, sessions.map(
    (s) => [s.id, s.title, s.snippet, s.mode, s.pinned, s.running, s.updated])]);
  if (sig === sessionsSig && nav.firstChild) return;
  sessionsSig = sig;
  // rows already on screen are not re-animated: only ones that were not there
  // a moment ago get the entry fade
  const had = new Set([...nav.querySelectorAll(".session")].map((r) => r.dataset.id));
  nav.replaceChildren();
  if (!sessions.length) {
    nav.append(el("div", "empty", query
      ? "No conversation mentions that." : "No conversations yet."));
    return;
  }

  // A collapsed section must never hide a search hit, so while there is a
  // query every section is drawn open. The remembered state is untouched and
  // comes back the moment the search is cleared.
  const closed = query ? new Set() : closedGroups();
  let index = 0;

  SESSION_GROUPS.forEach((group) => {
    const rows = sessions.filter(group.pick);
    if (!rows.length) return;                 // an empty section is not a section

    const section = el("section", `sess-group${closed.has(group.key) ? " closed" : ""}`);
    section.dataset.group = group.key;

    const head = el("button", "group-head");
    head.type = "button";
    head.setAttribute("aria-expanded", String(!closed.has(group.key)));
    head.append(icon("down", "chev"), el("span", null, group.label),
                el("b", "count", String(rows.length)));
    head.title = `${group.label} - click to ${closed.has(group.key) ? "expand" : "collapse"}`;
    head.onclick = () => {
      const nowClosed = !section.classList.contains("closed");
      section.classList.toggle("closed", nowClosed);
      head.setAttribute("aria-expanded", String(!nowClosed));
      head.title = `${group.label} - click to ${nowClosed ? "expand" : "collapse"}`;
      if (!query) setGroupClosed(group.key, nowClosed);
    };
    section.append(head);

    // the inner wrapper is what the 1fr -> 0fr collapse animation clips
    const pane = el("div", "group-rows");
    const inner = el("div", "group-inner");
    pane.append(inner);
    section.append(pane);

    rows.forEach((s) => {
      const row = el("button", `session${s.id === state.sessionId ? " on" : ""}`
        + (s.running ? " live" : "") + (had.has(s.id) ? "" : " fresh"));
      if (!had.has(s.id)) {
        row.style.animationDelay = `${Math.min(index, 12) * 18}ms`;
      }
      index += 1;

      // One quiet marker per row. It was briefly a different colour per chat,
      // hashed from the id - twenty rows of unrelated hues turned out to read
      // as confetti, not as an index. It is a plain bullet now, and spends its
      // colour on the two states that mean something: the open row, and a
      // conversation that is still working.
      const dot = el("span", "dot");
      if (s.running) dot.title = "Still working - open it to watch";
      row.append(dot);

      const title = el("span", "t");
      title.append(el("span", null, s.title));
      if (s.snippet) {
        const snippet = el("span", "snippet");
        const at = s.snippet.toLowerCase().indexOf(query);
        if (query && at >= 0) {
          snippet.append(document.createTextNode(s.snippet.slice(0, at)),
                         el("b", null, s.snippet.slice(at, at + query.length)),
                         document.createTextNode(s.snippet.slice(at + query.length)));
        } else {
          snippet.textContent = s.snippet;
        }
        title.append(snippet);
      }
      row.append(title);
      row.dataset.id = s.id;
      if (s.pinned) {
        row.classList.add("pinned");
        // No pin glyph on the row any more: every row in this section is
        // pinned, so drawing it fifteen times says nothing the heading has not
        // already said. The class stays - it is what drag-to-reorder selects.
        // Reordering only makes sense against the real, unfiltered list: a
        // search result's position says nothing about where a chat actually
        // sits among the pins.
        if (!query) row.draggable = true;
      }
      // last, so it is the row's right-hand column whatever else the row carries
      if (!query) {
        const when = el("span", "when", rowTime(s.updated));
        when.title = new Date((s.updated || 0) * 1000).toLocaleString();
        row.append(when);
      }
      const more = el("button", "del");
      more.title = "More (or right-click)";
      more.append(icon("menu"));
      more.onclick = (e) => { e.stopPropagation(); sessionMenu(e, row, s); };
      row.append(more);
      row.oncontextmenu = (e) => sessionMenu(e, row, s);
      // a click that lands while the rename box is open is the button being
      // activated by a keystroke inside it, not someone picking the chat
      row.onclick = () => { if (!row.dataset.renaming) openSession(s.id); };
      inner.append(row);
    });

    nav.append(section);
  });
}

/* Drag-and-drop only reorders pinned chats - the rest of the list sorts
   itself by recency and dragging across day groups would just fight that.
   Delegated on #sessions once, rather than re-bound on every loadSessions()
   render (a rename, a pin toggle, the health poll - all of those redraw the
   list). The live DOM move while dragging is just visual feedback; drop is
   what actually persists it, and a failed save re-renders from the server so
   the list can never end up believing an order that was not saved. */
(() => {
  const nav = $("#sessions");
  let dragging = null;

  nav.addEventListener("dragstart", (e) => {
    const row = e.target.closest(".session.pinned[draggable]");
    if (!row) { e.preventDefault(); return; }
    dragging = row;
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", row.dataset.id);
    // the browser paints its drag image next frame, so the class does not
    // make the row itself vanish from that snapshot
    requestAnimationFrame(() => row.classList.add("dragging"));
  });

  nav.addEventListener("dragover", (e) => {
    if (!dragging) return;
    const over = e.target.closest(".session.pinned");
    if (!over || over === dragging) return;
    e.preventDefault();          // only pinned rows are valid drop targets
    const before = e.clientY < over.getBoundingClientRect().top
      + over.getBoundingClientRect().height / 2;
    over.insertAdjacentElement(before ? "beforebegin" : "afterend", dragging);
  });

  nav.addEventListener("dragend", async () => {
    if (!dragging) return;
    dragging.classList.remove("dragging");
    const ids = [...nav.querySelectorAll(".session.pinned")].map((r) => r.dataset.id);
    dragging = null;
    try {
      await api("/ui/sessions/reorder", { method: "POST", body: JSON.stringify({ ids }) });
    } catch (e) {
      toast("Could not save the new order", true);
    }
    loadSessions();               // confirms it from the server either way
  });
})();

/* A user message may carry a pasted file, which is often longer than the
   screen - so a tall bubble gets a fade and a "show more". */
function renderUserBody(body, content) {
  if (typeof content === "string") {
    body.textContent = content;
  } else {
    (content || []).forEach((part) => {
      if (part.type === "text") body.append(document.createTextNode(part.text));
      if (part.type === "image_url") {
        const img = el("img", "shot");
        img.src = part.image_url.url;
        body.append(img);
      }
    });
  }
  requestAnimationFrame(() => {
    if (body.scrollHeight > 420 && !body.parentElement.querySelector(".more")) {
      body.classList.add("clipped");
      const more = el("button", "more", "Show the whole message");
      more.onclick = () => { body.classList.remove("clipped"); more.remove(); };
      body.after(more);
    }
  });
}

function renderStoredMessages(messages) {
  const thread = $("#thread");
  thread.replaceChildren();
  const cards = new Map();
  restoring = true;
  messages.forEach((m) => {
    if (m.role === "user") {
      renderUserBody(newMessage("user"), m.content);
    } else if (m.role === "assistant") {
      const body = newMessage("assistant");
      if (m.reasoning_content) {
        thinkBlock(body).textContent = m.reasoning_content;
        // a reloaded conversation follows the same preference as a live one
        const node = body.querySelector(".think");
        node.querySelector(".label").textContent = "Thinking";
        node.dataset.settling = "1";
        node.open = thinkOpenPref();
        delete node.dataset.settling;
      }
      if (m.content) {
        const node = contentBlock(body);
        node.dataset.raw = m.content;
        node.innerHTML = markdown(m.content);
        node.classList.remove("live");
        node.dataset.closed = "1";
        messageActions(body);
      }
      (m.tool_calls || []).forEach((c) => {
        let args = {};
        try { args = JSON.parse(c.function.arguments || "{}"); } catch (e) { args = {}; }
        cards.set(c.id, toolCard(body, {
          id: c.id, name: c.function.name, args,
          label: Object.values(args)[0] ? String(Object.values(args)[0]).slice(0, 120) : "",
        }));
      });
    } else if (m.role === "tool") {
      const ok = !String(m.content || "").startsWith("error:");
      finishToolCard(cards.get(m.tool_call_id), ok, m.content || "", 0);
    }
  });
  restoring = false;
  refreshEditAction();
  scrollDown(true, true);          // put it at the bottom, do not travel there
}

async function openSession(id) {
  if (state.streaming) detach();
  let session;
  try {
    session = await api(`/ui/sessions/${id}`);
  } catch (e) {
    toast(e.message, true);
    return;
  }
  state.sessionId = id;
  state.warnedContext = false;
  showContext(null);               // the meter belongs to the conversation
  setEndpoint(session.provider || "local", session.model || "");
  setMode(session.mode === "agent" ? "agent" : "chat");
  if (session.workspace) setWorkspace(session.workspace);
  renderPlan(session.plan);
  renderStoredMessages(session.messages || []);
  // Opening a chat does not rename, reorder or re-time anything: the only
  // change to the list is which row is selected. Reloading it re-fetched,
  // re-created and re-animated every row, which is the flicker you saw on
  // every click. Move the marker instead, and only fall back to a real
  // reload if this conversation is not in the list at all (opening one the
  // sidebar has not caught up with, straight after a restart).
  if (!markActiveSession(id)) loadSessions();
  closeSidebar();
  if (session.running || state.live?.[id]?.running) await attachTurn(id);
}

function renderChips() {
  const box = $("#welcome .chips");
  if (!box) return;
  box.replaceChildren();
  SUGGESTIONS[state.mode].forEach(([ico, text], i) => {
    const chip = el("button", "chip");
    chip.style.animationDelay = `${140 + i * 70}ms`;
    chip.append(icon(ico), el("span", null, text));
    chip.onclick = () => send(text);
    box.append(chip);
  });
}

function newChat() {
  if (state.streaming) detach();   // the turn carries on in its own conversation
  state.sessionId = null;
  state.warnedContext = false;
  showContext(null);
  renderPlan([]);
  // start where the user said new chats should start; without a default set,
  // this keeps whichever endpoint was last chosen
  applyDefaultEndpoint();
  const thread = $("#thread");
  thread.replaceChildren();

  const welcome = el("div");
  welcome.id = "welcome";
  const mark = el("div", "hero-mark");
  mark.append(el("i"));
  welcome.append(mark,
    el("h1", null, state.config?.title || "Local chat"),
    el("p", "sub", "Runs on this machine, against your own model."),
    el("div", "chips"));

  // The two explainer cards said what the Chat / Agent toggle at the top says,
  // in more words. One statement of a thing is enough.
  thread.append(welcome);
  renderChips();
  loadSessions();
  $("#to-bottom").hidden = true;
}

/* -------------------------------------------------------------- modals -- */

let modalReturn = null;

function openModal(title, build, opts) {
  $("#modal-title").textContent = title;
  // Settings needs a two-column body; every other dialog stays a single
  // padded column, so the class is set per-open rather than left on.
  $(".modal-box").classList.toggle("wide", !!(opts && opts.wide));
  const box = $("#modal-body");
  box.replaceChildren();
  build(box);
  $("#modal").hidden = false;
  modalReturn = document.activeElement;
  setTimeout(() => {
    // tabindex -1 means "reachable, but not the way in" - the settings rail
    // parks its inactive sections there, and focusing one of those drew a ring
    // around Appearance while the Advanced pane was on screen
    const first = $("#modal-body").querySelector(
      'input, button:not([tabindex="-1"]), textarea, [tabindex]:not([tabindex="-1"])')
      || $("#modal-close");
    first?.focus();
  }, 30);
}

function closeModal() {
  $("#modal").hidden = true;
  if (modalReturn?.focus) modalReturn.focus();
  modalReturn = null;
}

/* The thread itself must not be a live region - it is rewritten on every
   token - so turn boundaries are announced here instead. */
function announce(text) {
  const node = $("#sr-status");
  if (node) node.textContent = text;
}

/* The system dialog is what people expect on Windows; the kit's own browser
   stays as the fallback (and is the only sane option from another device,
   where a dialog would open on the server's screen). */
async function pickWorkspace() {
  if (!state.config?.native_picker) { browseModal(); return; }
  const chip = $("#ws-change");
  chip.disabled = true;
  const label = $("#ws-path").textContent;
  $("#ws-path").textContent = "waiting for the folder dialog...";
  try {
    const got = await api("/ui/pick_folder", {
      method: "POST", headers: UI_HEADERS,
      body: JSON.stringify({ start: state.workspace }),
    });
    if (got.path) {
      setWorkspace(got.path);
      toast("Workspace updated");
    } else {
      $("#ws-path").textContent = label;
    }
  } catch (e) {
    $("#ws-path").textContent = label;
    toast(`Folder dialog unavailable: ${e.message}`, true);
    browseModal();
  } finally {
    chip.disabled = false;
  }
}

function browseModal() {
  openModal("Agent workspace", (box) => {
    box.append(el("div", "muted-note",
      state.config?.native_picker
        ? "The system dialog could not open, so here is the built-in browser."
        : "The agent may read and change files here, and nowhere else."));
    const input = el("input", "path-input");
    input.style.margin = "12px 0";
    input.value = state.workspace;
    const use = el("button", "btn primary", "Use this folder");
    use.onclick = () => {
      setWorkspace(input.value.trim());
      closeModal();
      toast("Workspace updated");
    };
    const list = el("ul", "dirlist");
    box.append(input, use, list);

    const show = async (path) => {
      let data;
      try {
        data = await api(`/ui/browse?path=${encodeURIComponent(path || "")}`);
      } catch (e) {
        list.replaceChildren(el("li", null, String(e.message)));
        return;
      }
      input.value = data.path;
      list.replaceChildren();
      const row = (label, target, ico) => {
        const li = el("li");
        const b = el("button");
        b.append(icon(ico || "folder"), el("span", null, label));
        b.onclick = () => show(target);
        li.append(b);
        list.append(li);
      };
      if (data.parent) row("..", data.parent, "down");
      data.dirs.forEach((d) => row(d.name, d.path));
      if (!data.dirs.length && !data.parent) list.append(el("li", null, "(no subfolders)"));
    };
    show(state.workspace);
  });
}

/* Switching model means loading different weights into the same VRAM, so the
   server restarts into them. The UI waits for it to come back. */
async function modelModal() {
  let data;
  try {
    data = await api("/ui/models");
  } catch (e) { toast(e.message, true); return; }

  openModal("Model", (box) => {
    if (data.note) box.append(el("div", "muted-note", data.note));
    if (!data.models.length) {
      box.append(el("div", "muted-note",
        "No models found under models/. Run start.bat and pick a profile to "
        + "download one."));
      return;
    }
    if (data.remote?.length) {
      box.append(el("div", "group-label", "This computer"));
    }
    data.models.forEach((model) => {
      const row = el("button", `model-row${model.current ? " current" : ""}`);
      const who = el("div", "who");
      who.append(el("div", "name", model.name));
      const bits = [];
      if (model.bpw) bits.push(`${model.bpw} bpw`);
      if (model.quality) bits.push(model.quality);
      if (model.size_gb) bits.push(`${model.size_gb} GB`);
      if (model.context) bits.push(`${Math.round(model.context / 1024)}k context`);
      who.append(el("div", "meta", model.fits === false
        ? model.why : bits.join("  ·  ")));
      row.append(who);
      if (model.current) row.append(el("span", "badge", "loaded"));
      else if (model.fits === false) {
        row.append(el("span", "badge no",
                      model.complete === false ? "incomplete" : "too big"));
      }
      row.disabled = model.current || !data.can_switch || model.fits === false;
      row.onclick = () => switchModel(model);
      box.append(row);
    });
    if (data.can_switch) {
      box.append(el("div", "muted-note",
        "Switching restarts the server and reloads the weights - about a "
        + "minute. Open conversations are kept."));
    }

    if (data.remote?.length) {
      box.append(el("div", "group-label", "Providers"));
      data.remote.forEach((entry) => {
        const row = el("button", `model-row${
          state.provider === entry.provider && state.model === entry.model
            ? " current" : ""}`);
        const who = el("div", "who");
        who.append(el("div", "name", entry.model),
                   el("div", "meta", `${entry.provider_name}  ·  ${entry.base_url}`));
        row.append(who);
        if (state.provider === entry.provider && state.model === entry.model) {
          row.append(el("span", "badge", "in use"));
        }
        row.onclick = () => {
          setEndpoint(entry.provider, entry.model, entry.provider_name);
          rememberEndpoint(entry.provider, entry.model, entry.provider_name);
          closeModal();
          toast(`This chat now uses ${entry.model}`);
        };
        box.append(row);
      });
      box.append(el("div", "muted-note",
        "A provider answers instantly - nothing is loaded into VRAM. Tools "
        + "still run on this computer, and approvals still apply."));
    }

    const manage = el("button", "btn block");
    manage.style.marginTop = "14px";
    manage.append(icon("plus"), el("span", null,
      data.remote?.length ? "Manage providers" : "Add an OpenAI-compatible provider"));
    manage.onclick = () => providersModal();
    box.append(manage);
  });
}

/* Any OpenAI-compatible endpoint: another runtime on this machine, a box on
   the network, or a hosted API. */
async function providersModal() {
  let data;
  try {
    data = await api("/ui/providers");
  } catch (e) { toast(e.message, true); return; }

  openModal("Providers", (box) => {
    box.append(el("div", "muted-note",
      "Anything that speaks the OpenAI API: llama.cpp, Ollama, LM Studio, vLLM, "
      + "a machine on your network, or a hosted service. Keys are kept in "
      + "providers.json on this computer and never leave it."));

    const local = el("div", "model-row current");
    const who = el("div", "who");
    who.append(el("div", "name", data.local.name),
               el("div", "meta", `${data.local.model}  ·  ${data.local.base_url}`));
    local.append(who, el("span", "badge", "built in"));
    box.append(local);

    data.providers.forEach((provider) => {
      const row = el("div", "model-row");
      const info = el("div", "who");
      const bits = [provider.base_url];
      if (provider.models.length) bits.push(`${provider.models.length} models`);
      if (provider.has_key) bits.push(`key ${provider.key_hint}`);
      info.append(el("div", "name", provider.name),
                  el("div", "meta", bits.join("  ·  ")));
      const edit = el("button", "act");
      edit.append(icon("pencil"), el("span", null, "Edit"));
      edit.onclick = () => providerForm(provider);
      const del = el("button", "act");
      del.append(icon("trash"), el("span", null, "Remove"));
      del.onclick = async () => {
        try {
          await api(`/ui/providers/${provider.id}`, { method: "DELETE" });
          if (state.provider === provider.id) setEndpoint("local", "");
          toast(`Removed ${provider.name}`);
          providersModal();
        } catch (e) { toast(e.message, true); }
      };
      row.append(info, edit, del);
      box.append(row);
    });

    const add = el("button", "btn primary block");
    add.style.marginTop = "14px";
    add.append(icon("plus"), el("span", null, "Add a provider"));
    add.onclick = () => providerForm(null);
    box.append(add);
  });
}

function providerForm(provider) {
  const values = {
    id: provider?.id || "",
    name: provider?.name || "",
    base_url: provider?.base_url || "",
    api_key: "",
    default_model: provider?.default_model || "",
    models: provider?.models || [],
    context_length: provider?.context_length || "",
    vision: provider?.vision ?? null,
  };

  openModal(provider ? `Edit ${provider.name}` : "Add a provider", (box) => {
    const field = (label, key, placeholder, type) => {
      const wrap = el("label", "field");
      const head = el("div", "head");
      head.append(el("b", null, label));
      const input = el("input", "path-input");
      input.type = type || "text";
      input.placeholder = placeholder;
      input.value = values[key];
      input.oninput = () => { values[key] = input.value; };
      wrap.append(head, input);
      box.append(wrap);
      return input;
    };
    field("Name", "name", "OpenRouter");
    field("Base URL", "base_url", "https://openrouter.ai/api/v1");
    const keyBox = field("API key", "api_key",
      provider?.has_key ? `saved (${provider.key_hint}) - leave blank to keep`
        : "leave blank if the endpoint needs none", "password");
    keyBox.autocomplete = "off";

    // The endpoint cannot tell us this - /models does not report a window - and
    // a remote model's context has nothing to do with this card's VRAM. Without
    // it the context meter has nothing to measure against and stays hidden.
    const ctxBox = field("Context", "context_length",
                         "tokens, e.g. 128000 - leave blank if you do not know");
    ctxBox.type = "number";
    ctxBox.min = "512";
    ctxBox.oninput = () => {
      values.context_length = ctxBox.value === "" ? "" : Number(ctxBox.value);
    };

    // Images. Detection is offered as the default, but it has to be
    // overridable: many OpenAI-compatible servers publish no modality
    // information at all, and without a way to say "yes it does" a working
    // vision model would be unusable here forever.
    const visField = el("label", "field");
    const visHead = el("div", "head");
    const detected = provider?.vision_detected;
    visHead.append(el("b", null, "Images"),
                   el("small", null, detected === true
                     ? "the endpoint says this model takes images"
                     : detected === false
                       ? "the endpoint says this model is text-only"
                       : "the endpoint did not say - press Test, or set it here"));
    const visRow = el("div", "textsize-row");
    [[null, "Auto"], [true, "Yes"], [false, "No"]].forEach(([value, label]) => {
      const on = (values.vision ?? null) === value;
      const btn = el("button", `btn small${on ? " primary" : ""}`, label);
      btn.type = "button";
      btn.onclick = () => {
        values.vision = value;
        visRow.querySelectorAll("button").forEach((x) => x.classList.remove("primary"));
        btn.classList.add("primary");
      };
      visRow.append(btn);
    });
    visField.append(visHead, visRow);
    box.append(visField);

    const modelWrap = el("label", "field");
    const modelHead = el("div", "head");
    modelHead.append(el("b", null, "Model"));
    const test = el("button", "btn small");
    test.textContent = "Test and list models";
    modelHead.append(test);
    const model = el("input", "path-input");
    model.placeholder = "qwen/qwen3-max";
    model.value = values.default_model;
    model.setAttribute("list", "provider-models");
    model.oninput = () => { values.default_model = model.value; };
    const options = el("datalist");
    options.id = "provider-models";
    const status = el("div", "muted-note");
    modelWrap.append(modelHead, model, options, status);
    box.append(modelWrap);

    const fillModels = (list) => {
      values.models = list;
      options.replaceChildren();
      list.forEach((name) => {
        const option = el("option");
        option.value = name;
        options.append(option);
      });
      if (!model.value && list.length) {
        model.value = list[0];
        values.default_model = list[0];
      }
      status.textContent = `${list.length} models offered - pick one above.`;
    };
    if (values.models.length) fillModels(values.models);

    test.onclick = async () => {
      test.disabled = true;
      status.textContent = "Asking the endpoint what it serves...";
      try {
        const got = await api("/ui/providers/test", {
          method: "POST",
          body: JSON.stringify({ id: values.id, base_url: values.base_url,
                                 api_key: values.api_key }),
        });
        fillModels(got.models);
        if (got.vision !== undefined && got.vision !== null) {
          status.textContent += got.vision
            ? "  This model takes images."
            : "  This model is text-only.";
        }
      } catch (e) {
        status.textContent = e.message;
      } finally { test.disabled = false; }
    };

    const save = el("button", "btn primary block");
    save.style.marginTop = "6px";
    save.textContent = provider ? "Save changes" : "Add provider";
    save.onclick = async () => {
      try {
        const got = await api("/ui/providers", {
          method: "POST", body: JSON.stringify(values),
        });
        toast(`Saved ${got.provider.name}`);
        await reloadConfig();          // the attach button follows the new answer
        providersModal();
      } catch (e) { toast(e.message, true); }
    };
    box.append(save);
  });
}

async function switchModel(model) {
  openModal("Loading model", (box) => {
    const busy = el("div", "busy-box");
    busy.append(el("div", "spinner"),
                el("div", null, `Restarting into ${model.name}. This takes about `
                  + "a minute while the weights load into VRAM."));
    box.append(busy);
  });
  try {
    await api("/ui/switch_model", {
      method: "POST", headers: UI_HEADERS,
      body: JSON.stringify({ dir: model.dir }),
    });
  } catch (e) {
    closeModal();
    toast(e.message, true);
    return;
  }
  // the server answers /ui/config again only once the weights are loaded
  const deadline = Date.now() + 15 * 60 * 1000;
  const poll = async () => {
    if (Date.now() > deadline) {
      closeModal();
      toast("The server has not come back - check the console window", true);
      return;
    }
    try {
      const config = await api("/ui/config");
      if (config.model && config.model !== state.config.model) {
        state.config = config;
        // the name node, not the button: the chip also holds an icon and a
        // chevron, and textContent on the button would swallow both
        $("#stat-model-name").textContent = config.model;
        $("#stat-model").title = config.model;
        $("#stat-context").textContent = config.context_length
          ? `${Math.round(config.context_length / 1024)}k tokens` : "-";
        refreshAttachButton();
        showSpeed(null);
        closeModal();
        toast(`Loaded ${config.model}`);
        return;
      }
    } catch (e) { /* still down, keep waiting */ }
    setTimeout(poll, 2000);
  };
  setTimeout(poll, 3000);
}

/* ----------------------------------------------------------- settings -- */

/* Settings used to be one scroll: font size two rows above top-k, the system
   prompt below the fold whatever you came in for, and providers hidden behind
   a button at the very top. Six named sections instead, one on screen at a
   time, so "where is the theme" has an answer you can point at.

   The builders below are shared by the panes. Each returns the field and
   appends nothing, so a pane decides its own order. */

function fieldSlider(key, label, min, max, step, note) {
  const field = el("label", "field");
  const head = el("div", "head");
  const value = el("span", "val", String(state.settings[key]));
  head.append(el("b", null, label), value);
  const slider = el("input");
  slider.type = "range";
  slider.min = min; slider.max = max; slider.step = step;
  slider.value = state.settings[key];
  slider.oninput = () => {
    state.settings[key] = Number(slider.value);
    value.textContent = slider.value;
    saveSettings();
  };
  field.append(head, slider);
  if (note) field.append(el("small", "field-note", note));
  return field;
}

/* A row of mutually exclusive buttons. `options` is [value, label][]; `pick`
   is called with the chosen value and decides what that means. */
function fieldChoices(label, sub, options, current, pick, note) {
  const field = el("div", "field");
  const head = el("div", "head");
  head.append(el("b", null, label));
  if (sub) head.append(el("small", null, sub));
  const row = el("div", "textsize-row");
  options.forEach(([value, text]) => {
    const b = el("button", `btn small${value === current ? " primary" : ""}`, text);
    b.type = "button";
    b.onclick = () => {
      row.querySelectorAll("button").forEach((x) => x.classList.remove("primary"));
      b.classList.add("primary");
      pick(value);
    };
    row.append(b);
  });
  field.append(head, row);
  if (note) field.append(el("small", "field-note", note));
  return field;
}

/* A real switch rather than a checkbox wedged into a heading: the label and
   its explanation are the row, the control is the right-hand end of it. */
function fieldSwitch(label, note, checked, onChange) {
  const field = el("label", "field row");
  const who = el("div");
  who.append(el("b", null, label));
  if (note) who.append(el("span", null, note));
  const box = el("span", "switch");
  const input = el("input");
  input.type = "checkbox";
  input.checked = !!checked;
  input.onchange = () => onChange(input, box);
  box.append(input, el("i"));
  field.append(who, box);
  return field;
}

function paneHead(pane, title, lede) {
  pane.append(el("h3", null, title));
  if (lede) pane.append(el("p", "lede", lede));
}

/* ---- the panes ---------------------------------------------------------- */

function paneAppearance(pane) {
  paneHead(pane, "Appearance",
    "How the app looks on this device. Kept in this browser and never sent to "
    + "the model.");

  pane.append(fieldChoices("Theme", "the topbar button flips light and dark",
    [["", "System"], ["light", "Light"], ["dark", "Dark"]], currentTheme(),
    (v) => setTheme(v),
    "System follows whatever this computer is set to, and changes with it."));

  pane.append(fieldChoices("Font size", null,
    [["0.9", "Small"], ["1", "Default"], ["1.15", "Large"], ["1.3", "Extra large"]],
    currentTextScale(), (v) => setTextScale(v),
    "Scales the whole interface, not just the conversation."));

  pane.append(fieldChoices("Sidebar width", "or drag its right edge",
    [["240", "Narrow"], ["292", "Default"], ["360", "Wide"], ["440", "Widest"]],
    String(Math.round(parseFloat(
      getComputedStyle(document.documentElement).getPropertyValue("--side-w"))) || 292),
    (v) => setSidebarWidth(Number(v), true),
    `Anything between ${SIDE_W.min} and ${SIDE_W.max} pixels. Double-click the `
    + "edge to come back to the default."));
}

function paneModels(pane) {
  paneHead(pane, "Models",
    "Which model answers, and where it runs. The model in use is also the chip "
    + "under the message box.");

  const now = el("button", "btn block");
  now.append(icon("cpu"), el("span", null, "Change the model for this chat"));
  now.onclick = () => modelModal();
  pane.append(now);

  const providers = el("button", "btn block");
  providers.style.margin = "9px 0 20px";
  providers.append(icon("globe"), el("span", null, "Providers and endpoints"));
  providers.onclick = () => providersModal();
  pane.append(providers);

  // Which endpoint a new conversation starts on.
  const defField = el("label", "field");
  const defHead = el("div", "head");
  defHead.append(el("b", null, "Default model for new chats"),
                 el("small", null, "existing conversations keep their own"));
  const defSel = el("select", "path-input");
  const saved = state.settings.defaultEndpoint;
  const none = el("option", null, "Whatever I used last");
  none.value = "";
  defSel.append(none);
  endpointChoices().forEach((c) => {
    const o = el("option", null, c.label);
    o.value = `${c.provider}|${c.model}`;
    if (saved && saved.provider === c.provider
        && (saved.model || "") === (c.model || "")) o.selected = true;
    defSel.append(o);
  });
  if (!saved) none.selected = true;
  defSel.onchange = () => {
    if (!defSel.value) delete state.settings.defaultEndpoint;
    else {
      const [provider, ...rest] = defSel.value.split("|");
      state.settings.defaultEndpoint = { provider, model: rest.join("|") };
    }
    saveSettings();
    toast(defSel.value
      ? `New chats will start on ${defSel.selectedOptions[0].textContent}`
      : "New chats will keep whichever endpoint you used last");
  };
  defField.append(defHead, defSel);
  pane.append(defField);

  // How much the model thinks before it answers. "Off" is the only exact
  // setting: the chat template emits an empty <think></think> pair and there
  // is nowhere to reason. The rest are guidance the model can decline - the
  // engine cannot cut thinking short once it has started - so the help text
  // says so rather than implying a hard budget. Built from what this endpoint
  // actually accepts, not from a fixed menu.
  const levels = availableEfforts();
  pane.append(fieldChoices("Thinking",
    "or type /effort in the chat",
    [["default", "Model default"], ["off", "Off"],
     ...levels.map((l) => [l, EFFORT_LABELS[l] || l])],
    state.settings.thinking || "default",
    (v) => {
      if (v === "default") delete state.settings.thinking;
      else state.settings.thinking = v;
      saveSettings();
    },
    levels.length
      ? "Off is exact - the chat template leaves the model nowhere to reason. "
        + "Anything else is guidance, not a budget: a level is a request the "
        + "model can decline, and thinking cannot be cut short once it starts."
      : "This endpoint did not report which effort levels it takes, so only "
        + "Off and Model default are offered. Press Test on the provider to "
        + "ask again."));
}

function paneGeneration(pane) {
  paneHead(pane, "Generation",
    "Sampling, and the instruction every turn starts from. These do go to the "
    + "model.");

  pane.append(fieldSlider("temperature", "Temperature", 0, 2, 0.05,
    "Higher wanders further from the likeliest words. 0.6 suits this model."));
  pane.append(fieldSlider("top_p", "Top-p", 0.1, 1, 0.01,
    "Considers only the likeliest words that add up to this much probability."));
  pane.append(fieldSlider("top_k", "Top-k", 0, 100, 1,
    "A hard cap on how many candidates are in play. 0 turns it off."));
  pane.append(fieldSlider("max_tokens", "Max new tokens", 256, 16384, 256,
    "The ceiling for one reply. It stops there whether or not it was finished."));

  pane.append(el("hr", "set-sep"));

  const field = el("label", "field");
  const head = el("div", "head");
  head.append(el("b", null, "System prompt"),
              el("small", null, "prepended to every turn"));
  const area = el("textarea");
  area.placeholder = "Empty = the built-in prompt for the current mode.";
  area.value = state.settings.system || "";
  area.oninput = () => { state.settings.system = area.value; saveSettings(); };
  field.append(head, area);
  field.append(el("small", "field-note",
    "Applies to both Chat and Agent. Clear it to go back to the built-in one."));
  pane.append(field);
}

function paneAgent(pane) {
  paneHead(pane, "Agent",
    "Agent mode reads and changes files in one folder, and asks before it "
    + "writes or runs anything.");

  const field = el("div", "field");
  const head = el("div", "head");
  head.append(el("b", null, "Workspace folder"));
  field.append(head);
  const path = el("input", "path-input");
  path.value = state.workspace || "";
  path.readOnly = true;
  path.title = state.workspace || "";
  field.append(path);
  const change = el("button", "btn block");
  change.style.marginTop = "9px";
  change.append(icon("folder"), el("span", null, "Change folder"));
  change.onclick = async () => {
    await pickWorkspace();
    path.value = state.workspace || "";
    path.title = state.workspace || "";
  };
  field.append(change);
  field.append(el("small", "field-note",
    "The agent may read and change files here, and nowhere else."));
  pane.append(field);

  const tools = (state.config?.tools?.agent || []);
  if (tools.length) {
    const list = el("div", "field");
    const lhead = el("div", "head");
    lhead.append(el("b", null, "Tools available in Agent mode"),
                 el("small", null, `${tools.length} total`));
    list.append(lhead);
    const ul = el("ul", "set-info");
    tools.forEach((t) => {
      const li = el("li");
      li.append(el("span", null, t.name),
                el("b", `risk-${t.risk || "safe"}`, t.risk || "safe"));
      ul.append(li);
    });
    list.append(ul);
    list.append(el("small", "field-note",
      "Anything not marked safe stops for your approval the first time it is "
      + "used in a conversation."));
    pane.append(list);
  }

  if (state.config?.max_steps) {
    pane.append(el("small", "field-note",
      `A single agent turn takes at most ${state.config.max_steps} steps before `
      + "it hands back to you."));
  }
}

function paneNotifications(pane) {
  paneHead(pane, "Notifications",
    "What happens when a reply lands while you are looking at something else.");

  pane.append(fieldSwitch("Notify when a reply lands",
    "A desktop notification when the answer finishes and this tab is in the "
    + "background. The tab title always flags it, notification or not.",
    state.settings.notify, async (input) => {
      if (input.checked && window.Notification
          && Notification.permission !== "granted") {
        const granted = await Notification.requestPermission();
        if (granted !== "granted") {
          input.checked = false;
          toast("The browser refused notifications", true);
        }
      }
      state.settings.notify = input.checked;
      saveSettings();
    }));

  if (window.Notification && Notification.permission === "denied") {
    pane.append(el("small", "field-note",
      "This browser is currently blocking notifications for this site. Turn "
      + "them back on in the site settings beside the address bar."));
  }
}

function paneAdvanced(pane) {
  paneHead(pane, "Advanced",
    "What this page is talking to, and how to put it back the way it came.");

  const cfg = state.config || {};
  const rows = [
    ["Server", cfg.title || "-"],
    ["Local model", cfg.model || "-"],
    ["Context window", cfg.context_length
      ? `${Math.round(cfg.context_length / 1024)}k tokens` : "-"],
    ["Answering this chat", state.provider === "local"
      ? "this computer" : (state.model || state.provider)],
    ["Providers configured", String((cfg.providers || []).length)],
    ["Model switching", cfg.can_switch_model ? "available" : "not supervised"],
    ["Images accepted", activeVision() ? "yes" : "no"],
  ];
  const info = el("div", "field");
  const ihead = el("div", "head");
  ihead.append(el("b", null, "This session"));
  info.append(ihead);
  const ul = el("ul", "set-info");
  rows.forEach(([k, v]) => {
    const li = el("li");
    li.append(el("span", null, k), el("b", null, v));
    li.title = `${k}: ${v}`;
    ul.append(li);
  });
  info.append(ul);
  pane.append(info);

  const reset = el("div", "field");
  const rhead = el("div", "head");
  rhead.append(el("b", null, "Reset preferences"));
  reset.append(rhead);
  const btn = el("button", "btn block quiet-danger");
  btn.append(icon("reset"), el("span", null, "Reset everything to defaults"));
  btn.onclick = () => {
    // The conversations are files on this computer and are not touched - only
    // the preferences this browser is holding.
    state.settings = { system: "", ...(state.config?.defaults || {}) };
    saveSettings();
    setTheme("");
    setTextScale("1");
    setSidebarWidth(SIDE_W.def, true);
    try { localStorage.removeItem("chatui.lastEndpoint"); } catch (e) { /* ignore */ }
    applyDefaultEndpoint();
    toast("Settings reset - your conversations were not touched");
    settingsModal("advanced");
  };
  reset.append(btn);
  reset.append(el("small", "field-note",
    "Theme, font size, sidebar width, sampling, the system prompt and the "
    + "default model. Your conversations, providers and API keys are kept - "
    + "those live on the computer, not in this browser."));
  pane.append(reset);
}

const SETTINGS_PANES = [
  { id: "appearance", label: "Appearance", icon: "palette", build: paneAppearance },
  { id: "models", label: "Models", icon: "cpu", build: paneModels },
  { id: "generation", label: "Generation", icon: "sliders", build: paneGeneration },
  { id: "agent", label: "Agent", icon: "bolt", build: paneAgent },
  { id: "notifications", label: "Notifications", icon: "bell", build: paneNotifications },
  { id: "advanced", label: "Advanced", icon: "terminal", build: paneAdvanced },
];

function lastSettingsTab() {
  try { return localStorage.getItem("chatui.settingsTab") || ""; }
  catch (e) { return ""; }
}

function settingsModal(startAt) {
  openModal("Settings", (box) => {
    const wrap = el("div", "settings");
    const nav = el("nav", "set-nav");
    nav.setAttribute("aria-label", "Settings sections");
    nav.setAttribute("role", "tablist");
    const view = el("div", "set-pane");
    view.setAttribute("role", "tabpanel");

    const wanted = startAt || lastSettingsTab();
    let active = SETTINGS_PANES.find((p) => p.id === wanted) || SETTINGS_PANES[0];

    const show = (paneDef) => {
      active = paneDef;
      try { localStorage.setItem("chatui.settingsTab", paneDef.id); }
      catch (e) { /* ignore */ }
      nav.querySelectorAll(".set-tab").forEach((b) => {
        const on = b.dataset.pane === paneDef.id;
        b.classList.toggle("on", on);
        b.setAttribute("aria-selected", String(on));
        b.tabIndex = on ? 0 : -1;
      });
      view.replaceChildren();
      paneDef.build(view);
      view.scrollTop = 0;
      // narrow screens turn the rail into a horizontal strip: the open section
      // has to be visible in it, or reopening on a remembered tab looks like
      // the wrong one is selected
      nav.querySelector(".set-tab.on")?.scrollIntoView(
        { block: "nearest", inline: "nearest" });
    };

    SETTINGS_PANES.forEach((paneDef) => {
      const b = el("button", "set-tab");
      b.type = "button";
      b.dataset.pane = paneDef.id;
      b.setAttribute("role", "tab");
      b.append(icon(paneDef.icon), el("span", null, paneDef.label));
      b.onclick = () => show(paneDef);
      nav.append(b);
    });

    // up/down walks the rail, the way a real tab list does
    nav.addEventListener("keydown", (e) => {
      const keys = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 };
      if (!(e.key in keys)) return;
      e.preventDefault();
      const at = SETTINGS_PANES.indexOf(active);
      const next = SETTINGS_PANES[
        (at + keys[e.key] + SETTINGS_PANES.length) % SETTINGS_PANES.length];
      show(next);
      nav.querySelector(`[data-pane="${next.id}"]`)?.focus();
    });

    wrap.append(nav, view);
    box.append(wrap);
    show(active);
  }, { wide: true });
}

function saveSettings() {
  try { localStorage.setItem("chatui.settings", JSON.stringify(state.settings)); }
  catch (e) { /* private mode: settings just do not persist */ }
}

function savedSettings() {
  try {
    return JSON.parse(localStorage.getItem("chatui.settings") || "{}");
  } catch (e) { return {}; }
}

function loadSettings() {
  try {
    const theme = localStorage.getItem("chatui.theme");
    if (theme) document.documentElement.dataset.theme = theme;
  } catch (e) { /* ignore */ }
  loadSidebarWidth();
  try {
    const scale = localStorage.getItem("chatui.textScale");
    if (scale) document.documentElement.style.setProperty("--text-scale", scale);
  } catch (e) { /* ignore */ }
}

/* "" means follow the OS - the attribute comes off entirely, which is what
   the stylesheet's :root:not([data-theme="dark"]) media block is written
   against. The topbar button still flips light/dark directly; Settings is
   where the third state lives, because that is a decision you make once. */
function setTheme(mode) {
  const root = document.documentElement;
  if (mode) root.dataset.theme = mode;
  else delete root.dataset.theme;
  try {
    if (mode) localStorage.setItem("chatui.theme", mode);
    else localStorage.removeItem("chatui.theme");
  } catch (e) { /* ignore */ }
}

function currentTheme() {
  try { return localStorage.getItem("chatui.theme") || ""; }
  catch (e) { return ""; }
}

function toggleTheme() {
  const root = document.documentElement;
  const dark = root.dataset.theme
    ? root.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  setTheme(dark ? "light" : "dark");
}

/* A display preference, not a generation parameter - kept out of
   state.settings so it is never sent to the model on every turn. */
function setTextScale(scale) {
  document.documentElement.style.setProperty("--text-scale", scale);
  try { localStorage.setItem("chatui.textScale", scale); } catch (e) { /* ignore */ }
}

function currentTextScale() {
  try { return localStorage.getItem("chatui.textScale") || "1"; }
  catch (e) { return "1"; }
}

/* ---------------------------------------------------------- attachments -- */

function attachmentChip(item) {
  const fig = el("figure");
  if (item.kind === "image") {
    const img = el("img");
    img.src = item.url;
    fig.append(img);
  } else {
    const doc = el("div", "doc");
    const who = el("div");
    who.append(el("div", "name", item.name),
               el("span", "size", `${Math.round(item.text.length / 1024)} KB of text`));
    doc.append(icon("file"), who);
    fig.append(doc);
  }
  const rm = el("button", null, "×");
  rm.title = "Remove";
  rm.onclick = () => {
    state.attachments = state.attachments.filter((a) => a !== item);
    fig.remove();
  };
  fig.append(rm);
  $("#attachments").append(fig);
}

function addAttachment(item) {
  state.attachments.push(item);
  attachmentChip(item);
}

/* Images go to the vision tower; text files are pasted into the prompt, which
   is the only thing a text-only model can do with them. */
function readFiles(files) {
  [...files].forEach((file) => {
    if (file.type.startsWith("image/")) {
      if (!activeVision()) {
        toast(state.provider && state.provider !== "local"
          ? "This provider is not marked as taking images - turn on Images in "
            + "its settings if the model supports them"
          : "This model is running text-only - images are ignored", true);
        return;
      }
      const reader = new FileReader();
      reader.onload = () => addAttachment({ kind: "image", url: reader.result });
      reader.readAsDataURL(file);
    } else if (isTextFile(file)) {
      const reader = new FileReader();
      reader.onload = () => {
        let text = String(reader.result);
        if (text.length > MAX_DOC_CHARS) {
          text = text.slice(0, MAX_DOC_CHARS) + "\n[... truncated ...]";
          toast(`${file.name} was truncated to fit the context`);
        }
        addAttachment({ kind: "text", name: file.name, text });
      };
      reader.readAsText(file);
    } else {
      toast(`${file.name} is not text or an image`, true);
    }
  });
}

/* ------------------------------------------------------------ sidebar -- */

/* Resizable width.

   292px is a guess about somebody else's conversation titles: long paths and
   auto-generated names run past it, and on a narrow screen it is a third of
   the window spent on an index. So it is a real column with a drag handle, and
   the width is remembered per browser like the theme and the text scale.

   The clamp is not decoration. Below the minimum the rows stop being readable
   (title, pin, time and menu all want the same 40px); past the maximum it is
   the thread that gets squeezed, which is the thing you are actually here for.
   Double-click the handle - or Home while it has focus - to go back to the
   default. */
const SIDE_W = { min: 200, max: 520, def: 292 };

function setSidebarWidth(px, persist) {
  const w = Math.round(Math.min(SIDE_W.max, Math.max(SIDE_W.min, px || 0)));
  document.documentElement.style.setProperty("--side-w", `${w}px`);
  const handle = $("#side-resize");
  if (handle) handle.setAttribute("aria-valuenow", String(w));
  if (persist) {
    try { localStorage.setItem("chatui.sidebarWidth", String(w)); }
    catch (e) { /* private mode: the width just does not persist */ }
  }
  return w;
}

function loadSidebarWidth() {
  let saved = 0;
  try { saved = Number(localStorage.getItem("chatui.sidebarWidth")) || 0; }
  catch (e) { /* ignore */ }
  // applied before the first paint, so a remembered width does not flash the
  // default first - loadSettings() runs ahead of the config fetch
  if (saved) setSidebarWidth(saved, false);
}

(() => {
  const handle = $("#side-resize");
  if (!handle) return;
  handle.setAttribute("aria-valuemin", String(SIDE_W.min));
  handle.setAttribute("aria-valuemax", String(SIDE_W.max));

  let startX = 0;
  let startW = 0;
  let live = SIDE_W.def;

  const move = (e) => { live = setSidebarWidth(startW + (e.clientX - startX), false); };
  const end = () => {
    document.body.classList.remove("resizing");
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", end);
    window.removeEventListener("pointercancel", end);
    setSidebarWidth(live, true);          // persist where the pointer left it
  };

  handle.addEventListener("pointerdown", (e) => {
    // the drawer on a narrow screen has no column to resize
    if (e.button !== 0 || compact()) return;
    e.preventDefault();
    startX = e.clientX;
    startW = live = $("#sidebar").getBoundingClientRect().width;
    document.body.classList.add("resizing");
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", end);
    window.addEventListener("pointercancel", end);
  });

  handle.addEventListener("dblclick", () => {
    setSidebarWidth(SIDE_W.def, true);
    toast("Sidebar width reset");
  });

  // it is a focusable separator, so the arrow keys have to move it
  handle.addEventListener("keydown", (e) => {
    const now = $("#sidebar").getBoundingClientRect().width;
    const step = e.shiftKey ? 32 : 8;
    if (e.key === "ArrowLeft") { e.preventDefault(); live = setSidebarWidth(now - step, true); }
    else if (e.key === "ArrowRight") { e.preventDefault(); live = setSidebarWidth(now + step, true); }
    else if (e.key === "Home") { e.preventDefault(); live = setSidebarWidth(SIDE_W.def, true); }
  });
})();

function openSidebar() {
  $("#sidebar").classList.add("open");
  $("#scrim").classList.add("on");
}

function closeSidebar() {
  $("#sidebar").classList.remove("open");
  $("#scrim").classList.remove("on");
}

/* ---------------------------------------------------------------- boot -- */

/* How much of the context window this conversation is using. The number comes
   from the last turn's prompt tokens, which is what the model actually read. */
function showContext(promptTokens) {
  // remembered so the meter can be redrawn when the window it measures against
  // changes - switching endpoint, or editing the provider's context length
  state.lastPromptTokens = promptTokens;
  const meter = $("#context-meter");
  const total = activeContextLength();
  if (!promptTokens || !total) { meter.hidden = true; return; }
  const pct = Math.min(100, Math.round((promptTokens / total) * 100));
  meter.hidden = false;
  meter.style.setProperty("--fill", `${pct}%`);
  meter.querySelector("i").style.setProperty("--fill", `${pct}%`);
  meter.className = `meter${pct >= 90 ? " hot" : pct >= 75 ? " warn" : ""}`;
  $("#context-used").textContent = `${(promptTokens / 1000).toFixed(1)}k used  ${pct}%`;
  if (pct >= 90 && !state.warnedContext) {
    state.warnedContext = true;
    toast("This conversation nearly fills the context - start a new one soon", true);
  }
}

/* A local model can take minutes; the tab is usually in the background by the
   time it finishes. */
function notifyIfAway() {
  if (!document.hidden) return;
  const original = document.title;
  document.title = `✓ Reply ready - ${original}`;
  const restore = () => {
    document.title = original;
    document.removeEventListener("visibilitychange", restore);
  };
  document.addEventListener("visibilitychange", restore);
  if (state.settings.notify && window.Notification?.permission === "granted") {
    try {
      new Notification(state.config?.title || "Local chat",
                       { body: "Your reply is ready", tag: "simplex-reply" });
    } catch (e) { /* some browsers refuse without a service worker */ }
  }
}

/* Live tokens/second: /health carries the server's cumulative counters, so the
   rate is the delta between two polls. The exact per-turn figure arrives with
   the `done` event and replaces this estimate when the turn ends.

   This used to also drive a status pill in the sidebar footer. The pill was
   removed: /health only ever described *this machine's* server, so it sat
   there in red saying "unreachable" whenever the UI was started on its own
   (webui.bat, which has no model behind it by design) or whenever the chat was
   running perfectly well through a remote provider. A permanent alarm for a
   condition that is usually normal is worse than no indicator at all.

   With the pill gone the poll has one job left - the live rate - so it now
   runs only while a turn is streaming instead of every five seconds forever. */
let sample = null;
let speedTimer = null;

function scheduleSpeedSample(ms) {
  clearTimeout(speedTimer);
  speedTimer = setTimeout(sampleSpeed, ms);
}

function showSpeed(value, live) {
  const pill = $("#speed");
  const text = value == null ? "-"
    : `${value >= 100 ? Math.round(value) : value.toFixed(1)}`;
  const speed = $("#stat-speed");
  speed.textContent = value == null ? "" : `${text} tok/s`;
  speed.hidden = value == null;        // a dash is not information
  if (value == null) { pill.hidden = true; return; }
  pill.hidden = false;
  pill.classList.toggle("live", !!live);
  pill.querySelector("b").textContent = text;
}

async function sampleSpeed() {
  if (!state.streaming) return;          // nothing to measure between turns
  try {
    const h = await api("/health");
    const now = performance.now();
    const total = h.completion_tokens_total;
    if (typeof total === "number") {
      if (sample && h.busy) {
        const seconds = (now - sample.t) / 1000;
        const tokens = total - sample.c;
        if (seconds > 0.4 && tokens > 0) showSpeed(tokens / seconds, true);
      }
      sample = { t: now, c: total };
    }
  } catch (e) {
    // No local server to ask - the UI on its own, or a chat running through a
    // provider. Not an error: the turn's real rate still arrives with `done`.
  }
  if (state.streaming) scheduleSpeedSample(1000);
}

async function boot() {
  loadSettings();
  try {
    state.config = await api("/ui/config");
  } catch (e) {
    $("#thread").append(el("div", "err-box", `Cannot reach the server: ${e.message}`));
    return;
  }
  document.title = state.config.title;
  $("#brand-name").textContent = state.config.title;
  $("#stat-model").disabled = false;
  setEndpoint("local", state.config.model);
  $("#stat-context").textContent = state.config.context_length
    ? `${Math.round(state.config.context_length / 1024)}k tokens` : "-";
  // server defaults first, then anything this browser has chosen before
  state.settings = { system: "", ...state.config.defaults, ...savedSettings() };
  setWorkspace(state.config.default_workspace);
  if (compact()) $("#input").placeholder = "Ask anything";
  newChat();
  setMode("chat");
  refreshAttachButton();
  // a turn that was running when the window closed is still running now
  try {
    const { live } = await api("/ui/sessions");
    const busy = Object.entries(live || {}).find(([, t]) => t.running);
    if (busy) {
      toast("Picking up a reply that kept running");
      await openSession(busy[0]);
    }
  } catch (e) { /* nothing to pick up */ }
}

/* Switching mode mid-conversation would leave a transcript full of tool calls
   the other mode does not have, so it starts a fresh chat instead. */
function requestMode(mode) {
  if (mode === state.mode) return;
  const hasHistory = state.sessionId && document.querySelector(".msg");
  if (hasHistory) newChat();
  setMode(mode);
  if (hasHistory) toast(`New ${mode} chat started`);
}

document.querySelectorAll(".mode").forEach((b) => {
  b.onclick = () => requestMode(b.dataset.mode);
});
$("#new-chat").onclick = () => { newChat(); closeSidebar(); $("#input").focus(); };
$("#send").onclick = () => send();
$("#stop").onclick = stop;
$("#ws-change").onclick = pickWorkspace;
// wrapped: the click event must not land in settingsModal's startAt argument
$("#settings-btn").onclick = () => settingsModal();
$("#stat-model").onclick = modelModal;
let filterTimer = null;
$("#session-filter").addEventListener("input", () => {
  clearTimeout(filterTimer);
  filterTimer = setTimeout(loadSessions, 200);   // not once per keystroke
});
$("#theme-btn").onclick = toggleTheme;
$("#modal-close").onclick = closeModal;
$("#modal").onclick = (e) => { if (e.target.id === "modal") closeModal(); };
$("#menu").onclick = openSidebar;
$("#scrim").onclick = closeSidebar;
$("#attach").onclick = () => $("#file-input").click();
$("#file-input").onchange = (e) => readFiles(e.target.files);
$("#to-bottom").onclick = () => scrollDown(true);

$("#thread").addEventListener("scroll", () => {
  $("#to-bottom").hidden = nearBottom() || !$("#thread").querySelector(".msg");
});

$("#input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
$("#input").addEventListener("input", (e) => {
  e.target.style.height = "auto";
  e.target.style.height = `${Math.min(e.target.scrollHeight, 230)}px`;
});
document.addEventListener("click", (e) => {
  if (!e.target.closest("#menu-pop")) closeMenu();
});
window.addEventListener("blur", closeMenu);
$("#sessions").addEventListener("scroll", closeMenu);

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { closeMenu(); closeModal(); closeSidebar(); }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    newChat();
    $("#input").focus();
  }
});

/* copy buttons on code blocks, wherever they appear */
document.addEventListener("click", async (e) => {
  const btn = e.target.closest(".copy-btn");
  if (!btn) return;
  await copyText(btn.closest(".code-wrap")?.querySelector("pre")?.textContent || "");
  btn.classList.add("done");
  btn.replaceChildren(icon("check"), document.createTextNode("copied"));
  setTimeout(() => {
    btn.classList.remove("done");
    btn.replaceChildren(icon("copy"), document.createTextNode("copy"));
  }, 1600);
});

document.addEventListener("paste", (e) => {
  if (e.clipboardData?.files?.length) readFiles(e.clipboardData.files);
});
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer?.files?.length) readFiles(e.dataTransfer.files);
});

boot();

/* ---------------------------------------------------------------------------
   Installable app.

   The service worker only caches the shell (see sw.js); everything the chat
   actually does still goes to the server. Registration needs a secure context,
   which localhost is and a plain http:// LAN or Tailscale address is not - so
   over the network the UI simply stays a normal page.
--------------------------------------------------------------------------- */
if ("serviceWorker" in navigator && window.isSecureContext) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => { /* not fatal */ });
  });
}

let installPrompt = null;
const installBtn = $("#install-btn");

window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  installPrompt = e;
  if (installBtn) installBtn.hidden = false;
});

window.addEventListener("appinstalled", () => {
  installPrompt = null;
  if (installBtn) installBtn.hidden = true;
  toast("Simplex is installed. It opens in its own window from now on.");
});

if (installBtn) {
  installBtn.onclick = async () => {
    if (!installPrompt) { installBtn.hidden = true; return; }
    installBtn.disabled = true;
    try {
      installPrompt.prompt();
      await installPrompt.userChoice;
    } finally {
      installPrompt = null;
      installBtn.hidden = true;
      installBtn.disabled = false;
    }
  };
}
