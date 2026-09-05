/* Simplex first-run setup.
 *
 * The whole page is a function of one JSON state object, pushed from the
 * server over SSE. There is no client-side model of what "should" happen next:
 * the phase in the state decides which screen is on. That way a reload, a
 * second tab, or a laptop lid closed for an hour all land on the truth.
 */
"use strict";

const $ = (s) => document.querySelector(s);
const UI_HEADERS = { "Content-Type": "application/json", "X-Simplex-UI": "1" };

const ui = {
  phase: "",
  chosen: "",
  state: null,
  logLines: [],
  events: 0,
};

/* ---------------------------------------------------------------- theme -- */
try {
  const t = localStorage.getItem("chatui.theme");
  if (t) document.documentElement.dataset.theme = t;
} catch (e) { /* private mode: the OS preference is a fine default */ }

$("#theme-btn").onclick = () => {
  const root = document.documentElement;
  const dark = root.dataset.theme
    ? root.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("chatui.theme", root.dataset.theme); } catch (e) { /* ignore */ }
};

/* ----------------------------------------------------------- formatting -- */
function bytes(n) {
  if (!n && n !== 0) return "-";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return (i === 0 ? Math.round(n) : n.toFixed(1)) + " " + u[i];
}

function clock(seconds) {
  if (seconds == null || seconds < 0 || !isFinite(seconds)) return "";
  const s = Math.round(seconds);
  if (s < 60) return s + "s";
  const m = Math.floor(s / 60);
  if (m < 60) return m + "m " + String(s % 60).padStart(2, "0") + "s";
  return Math.floor(m / 60) + "h " + String(m % 60).padStart(2, "0") + "m";
}

function ctxLabel(n) {
  if (n >= 1000) return Math.round(n / 1024) + "k";
  return String(n);
}

/* -------------------------------------------------------------- screens -- */
const SCREENS = ["choose", "install", "done", "error"];

function show(name) {
  SCREENS.forEach((s) => $("#screen-" + s).classList.toggle("hidden", s !== name));
}

/* --------------------------------------------------------------- choose -- */
function renderFacts(p) {
  const g = p.gpu || {};
  const cells = [];
  const vram = g.vram_gib ? g.vram_gib.toFixed(1) + " GB" : "not detected";
  cells.push(["Graphics card", g.name || "not detected", g.name ? "" : "bad"]);
  cells.push(["Video memory", vram, g.vram_gib ? "" : "bad"]);
  if (g.arch && g.arch !== "unknown") cells.push(["Generation", g.arch, g.support === "slow" ? "warn" : ""]);
  if (g.driver) cells.push(["Driver", g.driver, ""]);
  cells.push(["Free disk space", (p.disk_free_gb || 0).toFixed(0) + " GB",
    (p.disk_free_gb || 0) < 25 ? "warn" : ""]);
  cells.push(["Python", (p.python && p.python.version) || "-", ""]);

  $("#facts").innerHTML = cells.map(([k, v, cls]) =>
    `<div class="fact ${cls}"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join("");
}

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderBlockers(p) {
  const box = $("#blockers");
  const bits = [];
  const g = p.gpu || {};
  if (g.support === "unsupported") {
    bits.push(`<div class="alert bad"><b>This graphics card cannot run Simplex</b>
      <p>${esc((g.notes || []).join(" "))}</p></div>`);
  } else if (g.support === "unknown") {
    bits.push(`<div class="alert bad"><b>No NVIDIA graphics card was found</b>
      <p>Simplex needs an NVIDIA GPU with a current driver. If the card is there,
      install or repair the NVIDIA driver and reload this page.</p></div>`);
  } else if (g.support === "slow") {
    bits.push(`<div class="alert warn"><b>This card will work, but slowly</b>
      <p>${esc((g.notes || []).join(" "))}</p></div>`);
  }
  const blockers = p.source_build_blockers || [];
  if (blockers.length && !p.engine_wheel) {
    bits.push(`<div class="alert warn"><b>The engine will have to be compiled here</b>
      <p>No prebuilt engine matches this Python, and these are missing:
      ${esc(blockers.join(", "))}. Setup will still try, and will tell you exactly
      what to do if the build fails.</p></div>`);
  }
  box.innerHTML = bits.join("");
}

async function setVision(want) {
  // Its own endpoint: /setup/refresh re-probes the machine (nvidia-smi, wheel
  // tags, disk) which took seconds per click, and there null had to mean "keep
  // the current answer" so "decide for me" could never clear a yes or no.
  const btns = document.querySelectorAll("[data-vision]");
  btns.forEach((b) => { b.disabled = true; });
  try {
    apply(await post("/setup/vision", { vision: want }));
  } catch (e) {
    toast(e.message);
    btns.forEach((b) => { b.disabled = false; });
  }
}

function renderVisionChoice(menu) {
  let box = $("#vision-choice");
  if (!box) {
    box = document.createElement("div");
    box.id = "vision-choice";
    box.className = "note";
    $("#opts").parentNode.insertBefore(box, $("#opts"));
  }
  const on = menu.vision === true;
  const off = menu.vision === false;
  const hidden = menu.hidden_by_vision || [];
  box.innerHTML = `
    <b>Images</b> - the model can read pictures you paste into the chat.
    The vision tower costs ${esc(menu.vision_cost || "VRAM")} that would
    otherwise be context.
    <span class="s-seg">
      <button type="button" class="btn${on ? " primary" : ""}" data-vision="yes">Yes</button>
      <button type="button" class="btn${off ? " primary" : ""}" data-vision="no">No</button>
      <button type="button" class="btn${menu.vision === null || menu.vision === undefined ? " primary" : ""}"
        data-vision="auto">Decide for me</button>
    </span>
    ${hidden.length ? `<small class="s-caveat">Hidden because they cannot fit images
      with a useful context on this card: ${esc(hidden.join(", "))} bpw - higher
      quality than the options below.</small>` : ""}`;
  box.querySelectorAll("[data-vision]").forEach((b) => {
    b.onclick = () => {
      const v = b.dataset.vision;
      ui.chosen = null;                 // the menu changes, so re-pick from it
      setVision(v === "auto" ? null : v === "yes");
    };
  });
}

// --- simulation mode: the same planner, run against a card you do not have ---
// Read-only end to end: /setup/simulate never touches cfg or .env, so this can
// be opened mid-install without changing anything.
const sim = { cards: [], ready: false, busy: false };

function simInputs() {
  return {
    card: $("#sim-card"), vram: $("#sim-vram"), cc: $("#sim-cc"), out: $("#sim-out"),
  };
}

async function fillCards(cards) {
  const el = $("#sim-card");
  if (!el || sim.ready) return;
  if (!cards || !cards.length) {
    // The server may predate this panel: static files are served from disk, so
    // the browser can be running new JS against an old process whose /setup/state
    // carries no card list. Ask for it directly before giving up.
    try {
      const r = await fetch("/setup/cards", { cache: "no-store" });
      if (r.ok) cards = (await r.json()).cards;
    } catch (e) { /* fall through to the message below */ }
  }
  if (!cards || !cards.length) {
    const out = $("#sim-out");
    if (out) {
      out.innerHTML = `<p class="note">Simulation needs a newer setup server than the
        one running. Close this window, start Simplex again, and reopen this page.</p>`;
    }
    return;
  }
  sim.cards = cards;
  let gen = "";
  let html = "";
  cards.forEach((c, i) => {
    if (c.gen !== gen) {
      if (gen) html += "</optgroup>";
      html += `<optgroup label="${esc(c.gen)}">`;
      gen = c.gen;
    }
    html += `<option value="${i}">${esc(c.name)} - ${esc(c.vram_gib)} GB</option>`;
  });
  html += `</optgroup><optgroup label="Other"><option value="custom">Custom...</option></optgroup>`;
  el.innerHTML = html;
  sim.ready = true;

  const pickDefault = cards.findIndex((c) => c.vram_gib === 16);
  el.value = String(pickDefault >= 0 ? pickDefault : 0);
  el.onchange = () => {
    const custom = el.value === "custom";
    document.querySelectorAll(".sim-custom").forEach((n) => { n.hidden = !custom; });
    if (!custom) {
      const c = sim.cards[Number(el.value)];
      $("#sim-vram").value = c.vram_gib;
      $("#sim-cc").value = c.cc;
    }
    runSim();
  };
  $("#sim-vram").oninput = runSim;
  $("#sim-cc").oninput = runSim;
  document.querySelectorAll(".sim-custom").forEach((n) => { n.hidden = true; });
  const first = cards[Number(el.value)];
  $("#sim-vram").value = first.vram_gib;
  $("#sim-cc").value = first.cc;

  const box = $("#sim");
  if (box) box.ontoggle = () => { if (box.open && !$("#sim-out").innerHTML) runSim(); };
}

let simTimer = null;
function runSim() {
  clearTimeout(simTimer);
  simTimer = setTimeout(doSim, 150);         // typing in the number boxes
}

async function doSim() {
  const el = $("#sim-card");
  if (!el || sim.busy) return;
  const custom = el.value === "custom";
  const chosen = custom ? null : sim.cards[Number(el.value)];
  const payload = {
    vram: Number($("#sim-vram").value) || 0,
    cc: Number($("#sim-cc").value) || 0,
    name: chosen ? chosen.name : "",
    driver: "581.29",                        // a current driver; support notes are about the card
  };
  if (!payload.vram) return;
  sim.busy = true;
  try {
    renderSim(await post("/setup/simulate", payload));
  } catch (e) {
    $("#sim-out").innerHTML = `<p class="note">Could not simulate: ${esc(e.message)}</p>`;
  } finally {
    sim.busy = false;
  }
}

function renderSim(d) {
  const g = d.gpu;
  // "runs this kit" next to "nothing fits" reads as a contradiction, so the
  // memory verdict and the hardware verdict are combined into one sentence.
  let verdict = {
    ok: "runs this kit", slow: "runs, but at about half the speed",
    unsupported: "cannot run this kit at all", unknown: "unknown",
  }[d.support] || d.support;
  if (d.support !== "unsupported" && !d.fits) {
    verdict = d.support === "slow"
      ? "supported, but at about half the speed - and this model does not fit anyway"
      : "supported, but this model does not fit in it";
  }
  const notes = (d.notes || []).map((n) => `<div>${esc(n)}</div>`).join("");
  const budgetLine = d.support === "unsupported" ? ""
    : ` About <b>${esc(d.budget)} GB</b> usable for the model.`;

  let rows = "";
  if (d.options.length) {
    rows = `<table class="sim-tab"><thead><tr>
        <th>Quant</th><th>Quality</th><th>Context</th><th>Images</th>
        <th>VRAM</th><th>Download</th></tr></thead><tbody>` +
      d.options.map((o) => `<tr class="${o.quant === d.recommended ? "rec" : ""}">
        <td>${esc(o.quant)} bpw</td>
        <td>${esc(o.note)}${o.quant === d.recommended ? " - recommended" : ""}
            ${o.measured ? "" : " <em>(estimated)</em>"}</td>
        <td>${esc(ctxLabel(o.ctx))}${o.verified_ctx ? "" : "*"}</td>
        <td>${o.vision ? "yes" : "no"}</td>
        <td>${esc(o.need)} GB</td>
        <td>${esc(o.disk_gb)} GB</td></tr>`).join("") + "</tbody></table>";
  } else if (d.support === "unsupported") {
    rows = "";                               // the reason above is the whole story
  } else {
    rows = `<p class="note">Nothing fits. The smallest profile
      (${esc(d.smallest.quant)} bpw at ${esc(ctxLabel(d.smallest.ctx))} context) needs
      roughly a <b>${esc(d.min_card_gib)} GB</b> card.</p>`;
  }

  // the part people get backwards: int4 is the one with no hardware requirement
  const kv = (d.kv || []).map((k) => k.available
    ? `${esc(k.label)}`
    : `<span class="no">${esc(k.label)} - ${esc(k.why)}</span>`).join(" &middot; ");

  $("#sim-out").innerHTML = `
    <div class="sim-verdict ${esc(d.support)}">
      <b>${esc(g.name)}</b> - ${esc(g.vram_gib)} GB, ${esc(g.arch)}${g.cc ? ", compute " + esc(g.cc) : ""}
      &rarr; ${esc(verdict)}.${budgetLine}
      ${notes}
    </div>
    ${rows}
    <div class="sim-kv"><b>KV cache formats:</b> ${kv}</div>
    ${d.options.some((o) => !o.verified_ctx) ? `<p class="note" style="margin-top:8px">*
      context is the planner's ceiling; no prefill has proven it on this card.</p>` : ""}`;
}

function renderOptions(menu) {
  const wrap = $("#opts");
  const opts = menu.options || [];
  if (!opts.length) {
    wrap.innerHTML = `<p class="note">No model fits this graphics card with a useful
      context length.</p>`;
    $("#go").disabled = true;
    return;
  }
  if (!ui.chosen) {
    const rec = opts.find((o) => o.current) || opts.find((o) => o.downloaded)
      || opts.find((o) => o.recommended) || opts[0];
    ui.chosen = rec.id;
  }
  $("#menu-note").textContent =
    `${menu.gpu || "This GPU"} - about ${menu.budget} GB usable for the model.`
    + " Higher is smarter; lower leaves more room for a long conversation.";

  // The images question the console asks, asked here too - it changes which
  // quants fit and how much context each one gets, so it belongs before the
  // menu rather than being decided for the user.
  renderVisionChoice(menu);

  wrap.innerHTML = opts.map((o) => {
    const tags = (o.recommended ? '<span class="s-badge">Recommended</span>' : "")
      + (o.downloaded ? '<span class="s-badge have">Already downloaded</span>' : "");
    // Two different things can be unverified, and they mean different things:
    // an estimated VRAM figure, and a context no prefill has ever survived.
    const marks = [];
    if (!o.measured) marks.push("memory use estimated");
    if (!o.verified_ctx) marks.push("context not verified on hardware");
    const caveat = marks.length
      ? `<small class="s-caveat">${esc(marks.join(" - "))}</small>` : "";
    return `<button class="s-opt" type="button" data-id="${esc(o.id)}"
        aria-pressed="${o.id === ui.chosen}">
      <span class="dot" aria-hidden="true"></span>
      <span class="who">
        <b>${esc(o.quant)} bits per weight - ${esc(o.note)}${tags}</b>
        <small>${esc(ctxLabel(o.ctx))} context${o.vision ? " - reads images" : ""}
          - about ${esc(o.need)} GB of video memory</small>
        ${caveat}
      </span>
      <span class="num">${o.downloaded ? "<b>on disk</b>" : `<b>${esc(o.disk_gb)} GB</b>download`}</span>
    </button>`;
  }).join("");

  wrap.querySelectorAll(".s-opt").forEach((b) => {
    b.onclick = () => {
      ui.chosen = b.dataset.id;
      wrap.querySelectorAll(".s-opt").forEach((x) =>
        x.setAttribute("aria-pressed", String(x.dataset.id === ui.chosen)));
      updateGoNote(menu);
    };
  });
  updateGoNote(menu);
}

function updateGoNote(menu) {
  const o = (menu.options || []).find((x) => x.id === ui.chosen);
  if (!o) { $("#go-note").textContent = ""; return; }
  const free = (ui.state && ui.state.probe && ui.state.probe.disk_free_gb) || 0;
  // PyTorch and the engine need a few GB of their own on top of the weights
  const need = o.downloaded ? 6 : o.disk_gb + 6;
  if (free && free < need) {
    $("#go-note").innerHTML = `<b style="color:var(--warn)">Only ${free.toFixed(0)} GB free -
      this needs about ${Math.ceil(need)} GB.</b>`;
    return;
  }
  $("#go-note").textContent = o.downloaded
    ? "The weights are already here, so this is just the install."
    : `About ${o.disk_gb} GB to download.`;
}

/* -------------------------------------------------------------- install -- */
function renderDownload(d) {
  const bar = $("#dl-bar");
  const pct = d.percent || 0;
  const listing = d.state === "listing" || (d.state === "downloading" && !d.total_bytes);
  bar.classList.toggle("indet", listing);
  bar.querySelector("i").style.width = listing ? "" : pct + "%";

  if (d.state === "done") {
    $("#dl-left").textContent = d.message || "Downloaded";
    $("#dl-right").textContent = d.total_bytes ? bytes(d.total_bytes) : "";
    $("#dl-file").textContent = "";
    bar.querySelector("i").style.width = "100%";
    return;
  }
  if (d.state === "error") {
    $("#dl-left").textContent = "Download failed";
    $("#dl-right").textContent = "";
    $("#dl-file").textContent = d.message || "";
    return;
  }
  if (d.state === "listing" || d.state === "idle") {
    $("#dl-left").textContent = d.message || "Preparing";
    $("#dl-right").textContent = "";
    return;
  }
  $("#dl-left").textContent =
    `${pct.toFixed(1)}%  -  ${bytes(d.done_bytes)} of ${bytes(d.total_bytes)}`;
  const eta = clock(d.eta_seconds);
  $("#dl-right").textContent =
    (d.speed_bps ? bytes(d.speed_bps) + "/s" : "") + (eta ? "  -  " + eta + " left" : "");
  $("#dl-file").textContent = d.state === "verifying" ? "checking " + (d.current || "") : (d.current || "");
}

const MARKS = { pending: "·", running: "", ok: "✓", skipped: "✓", failed: "×" };

function renderSteps(steps, target) {
  target.innerHTML = (steps || []).map((s) => {
    const mark = s.state === "running" ? '<span class="spin"></span>' : MARKS[s.state] || "";
    const detail = s.state === "failed" ? s.error : s.detail;
    const secs = s.state === "ok" && s.seconds >= 1 ? clock(s.seconds) : "";
    return `<li data-state="${esc(s.state)}">
      <span class="mark">${mark}</span>
      <span><b>${esc(s.title)}</b>${detail ? `<small>${esc(detail)}</small>` : ""}</span>
      <span class="secs">${esc(secs)}</span>
    </li>`;
  }).join("");
}

function renderLine(entry, target) {
  if (!target) return;
  const stick = target.scrollTop + target.clientHeight >= target.scrollHeight - 24;
  const cls = entry.kind === "cmd" || entry.kind === "error" || entry.kind === "hint" ? entry.kind : "";
  const line = document.createElement("span");
  line.className = cls;
  line.textContent = entry.line + "\n";
  target.appendChild(line);
  while (target.childNodes.length > 1200) target.removeChild(target.firstChild);
  if (stick) target.scrollTop = target.scrollHeight;
}

function pushLog(entry, target) {
  ui.logLines.push(entry);
  if (ui.logLines.length > 1200) ui.logLines.splice(0, ui.logLines.length - 1200);
  renderLine(entry, target);
}

/* Repaint a log pane from the buffer - used when the error screen takes over
   from the install screen, so the lines that explain the failure are there. */
function redrawLog(target) {
  if (!target) return;
  target.textContent = "";
  ui.logLines.forEach((e) => renderLine(e, target));
  target.scrollTop = target.scrollHeight;
}

/* ---------------------------------------------------------------- apply -- */
function apply(state) {
  if (!state) return;
  ui.state = state;
  const phase = state.phase;

  if (phase === "choose" || phase === "probe") {
    show("choose");
    // #go disables itself on click. Coming back here - from Stop, or from
    // "Choose a different model" on the error screen - has to undo that, or
    // the only button on the page is permanently dead.
    $("#go").disabled = false;
    renderFacts(state.probe || {});
    renderBlockers(state.probe || {});
    renderOptions(state.menu || {});
    fillCards(state.cards);
  } else if (phase === "install") {
    show("install");
    $("#dl-model").textContent = state.model_id ? "- " + state.model_id : "";
    renderDownload(state.download || {});
    renderSteps(state.steps, $("#steps"));
  } else if (phase === "done") {
    show("done");
    startHandoff();
  } else if (phase === "cancelled") {
    show("choose");
    toast("Setup stopped. Nothing was lost - starting again resumes the download.");
  } else if (phase === "error") {
    show("error");
    const e = state.error || {};
    $("#err-msg").textContent = e.message || "Something went wrong.";
    $("#err-hint").textContent = e.hint || "";
    renderSteps(state.steps, $("#err-steps"));
    redrawLog($("#err-out"));
  }
  ui.phase = phase;
}

/* ---------------------------------------------------------------- wire --- */
async function post(path, body) {
  const r = await fetch(path, {
    method: "POST", headers: UI_HEADERS, body: JSON.stringify(body || {}),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.hint ? data.error + " " + data.hint : (data.error || "request failed"));
  return data;
}

$("#go").onclick = async () => {
  $("#go").disabled = true;
  try {
    const token = $("#hf-token").value.trim();
    if (token) await post("/setup/token", { token });
    apply(await post("/setup/start", { quant: ui.chosen }));
  } catch (e) {
    toast(e.message);
    $("#go").disabled = false;
  }
};

$("#cancel").onclick = async () => {
  $("#cancel").disabled = true;
  try { await post("/setup/cancel", {}); } catch (e) { /* the page will catch up */ }
  setTimeout(() => { $("#cancel").disabled = false; }, 1500);
};

$("#retry").onclick = async () => {
  $("#retry").disabled = true;
  try {
    apply(await post("/setup/start", {}));
  } catch (e) {
    toast(e.message);
  } finally {
    $("#retry").disabled = false;
  }
};

$("#back").onclick = async () => {
  try { apply(await post("/setup/refresh", {})); } catch (e) { toast(e.message); }
};

function toast(text) {
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = text;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), 6000);
}

/* ----------------------------------------------------------- streaming -- */
function listen() {
  const src = new EventSource("/setup/events?from=" + ui.events);
  src.onmessage = (m) => {
    let e;
    try { e = JSON.parse(m.data); } catch (err) { return; }
    ui.events = Math.max(ui.events, (e.i || 0) + 1);
    if (e.type === "log") {
      pushLog(e, ui.phase === "error" ? $("#err-out") : $("#out"));
    } else if (e.type === "steps") {
      renderSteps(e.steps, ui.phase === "error" ? $("#err-steps") : $("#steps"));
    } else if (e.type === "download") {
      if (ui.state) ui.state.download = e.download;
      renderDownload(e.download || {});
    } else if (e.type === "phase" || e.type === "chose") {
      apply(e.state);
    }
  };
  src.onerror = () => {
    src.close();
    // the setup server closes its socket when it hands over to the model
    // server; in every other case this is a hiccup worth retrying
    setTimeout(() => { if (ui.phase !== "done") listen(); }, 1200);
  };
}

/* ------------------------------------------------------------- handoff -- */
let handoffTimer = null;
let handoffTries = 0;

function startHandoff() {
  if (handoffTimer) return;
  const tick = async () => {
    handoffTries++;
    try {
      const r = await fetch("/health", { cache: "no-store" });
      if (r.ok) {
        const data = await r.json().catch(() => ({}));
        if (data.status !== "setup") { location.href = "/"; return; }
      }
    } catch (e) { /* nothing listening yet: that is the normal case */ }
    const mins = Math.floor(handoffTries * 2 / 60);
    $("#done-state").textContent = handoffTries < 8
      ? "Starting the model server"
      : "Loading the model into the graphics card";
    $("#done-note").textContent = mins >= 1
      ? `Still working - ${mins} minute${mins > 1 ? "s" : ""} so far. The first load is the
         slow one because the GPU kernels are compiled and cached.`
      : "";
    handoffTimer = setTimeout(tick, 2000);
  };
  tick();
}

/* ---------------------------------------------------------------- boot --- */
(async function boot() {
  try {
    const r = await fetch("/setup/state", { cache: "no-store" });
    if (!r.ok) {
      // Something is answering this port, but it is not setup: the model server
      // has taken it over (it 404s /setup/state). This page is stale, so go to
      // the app that is actually there rather than reporting a failure.
      location.href = "/";
      return;
    }
    const state = await r.json();
    ui.events = state.events || 0;
    const log = await (await fetch("/setup/log", { cache: "no-store" })).json();
    (log.log || []).forEach((e) => ui.logLines.push(e));
    apply(state);
    redrawLog(state.phase === "error" ? $("#err-out") : $("#out"));
    listen();
  } catch (e) {
    // Nothing is answering on this port. The overwhelmingly likely reason is not
    // that setup died: setup and the model server share this port, and between
    // the two there is a gap of several minutes while the weights load and the
    // GPU kernels compile. Reloading during that window used to land here and be
    // told to start Simplex again - while Simplex was, in fact, starting.
    // So wait for it, and only suggest a restart once it has really been too long.
    waitForServer();
  }
})();

function waitForServer() {
  const started = Date.now();
  document.body.innerHTML =
    '<div id="setup"><h1>Simplex is starting</h1>'
    + '<p class="lede" id="wait-state">Loading the model into the graphics card. '
    + 'The first start is the slow one - the GPU kernels are compiled and cached.</p>'
    + '<p class="note" id="wait-note"></p></div>';

  const tick = async () => {
    try {
      const r = await fetch("/health", { cache: "no-store" });
      if (r.ok || r.status === 503) {
        const data = await r.json().catch(() => ({}));
        // setup came back (a restart), or the model server took the port: either
        // way this page is stale and a reload gets the right one
        location.reload();
        return;
      }
    } catch (err) { /* still nothing listening */ }
    const mins = Math.round((Date.now() - started) / 60000);
    const note = $("#wait-note");
    if (note) {
      note.textContent = mins >= 10
        ? "This is longer than a first load usually takes. If the Simplex window has "
          + "closed or shows an error, close this page and start Simplex again."
        : (mins >= 1 ? `Still working - about ${mins} minute${mins > 1 ? "s" : ""} so far.` : "");
    }
    setTimeout(tick, 2000);
  };
  tick();
}
