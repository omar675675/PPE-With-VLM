"""
Dashboard for PPE violations found by process.py.

Run:
    .venv/bin/python process.py      # once, or whenever videos change
    .venv/bin/python dashboard.py [config.yaml]
Then open http://localhost:<dashboard.port from config.yaml>
"""

import json
import sys
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

CONFIG_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "config.yaml")
CFG = yaml.safe_load(CONFIG_PATH.read_text())

OUT_DIR = Path(CFG["output_dir"])
if not OUT_DIR.is_absolute():
    OUT_DIR = CONFIG_PATH.parent / OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.mount("/output", StaticFiles(directory=OUT_DIR), name="output")


@app.get("/api/data")
def data():
    f = OUT_DIR / "violations.json"
    if not f.exists():
        return JSONResponse({"videos": [], "violations": [], "types": list(CFG["ppe"]["checks"]),
                             "missing": True})
    return json.loads(f.read_text())


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PPE Violations</title>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #6f6e69;
    --line: #e1e0d9; --axis: #c3c2b7; --accent: #2a78d6;
    --critical: #d03b3b; --serious: #ec835a; --warning: #fab219;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #9a998f;
      --line: #2c2c2a; --axis: #383835; --accent: #3987e5;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #9a998f;
    --line: #2c2c2a; --axis: #383835; --accent: #3987e5;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--page); color: var(--ink);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  .wrap { max-width: 1400px; margin: 0 auto; padding: 20px 24px 40px; }
  header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 16px; }
  h1 { font-size: 20px; margin: 0; }
  header .sub { color: var(--muted); }
  h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--ink-2);
       margin: 28px 0 10px; font-weight: 600; }

  /* filter row */
  .filters { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: var(--surface); }
  .seg button { font: inherit; color: var(--ink-2); background: none; border: 0; padding: 6px 12px; cursor: pointer; }
  .seg button + button { border-left: 1px solid var(--line); }
  .seg button[aria-pressed="true"] { background: var(--accent); color: #fff; }

  /* stat tiles */
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 16px; }
  .tile { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
  .tile .label { color: var(--ink-2); font-size: 13px; display: flex; align-items: center; gap: 6px; }
  .tile .value { font-size: 32px; font-weight: 650; margin-top: 4px; font-variant-numeric: tabular-nums; }
  .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; background: var(--warning); }
  .dot.no_helmet { background: var(--critical); }
  .dot.no_vest { background: var(--serious); }

  /* per-video table */
  table { width: 100%; border-collapse: collapse; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--line); }
  th { color: var(--ink-2); font-weight: 600; font-size: 13px; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  tr:last-child td { border-bottom: 0; }
  tbody tr { cursor: pointer; }
  tbody tr:hover td { background: color-mix(in srgb, var(--accent) 8%, transparent); }
  .table-scroll { overflow-x: auto; }

  /* gallery */
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 12px; }
  .card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; overflow: hidden;
          cursor: pointer; display: flex; flex-direction: column; text-align: left; font: inherit; color: inherit; padding: 0; }
  .card:hover, .card:focus-visible { border-color: var(--accent); outline: none; }
  .card .img { height: 240px; background: var(--page); display: flex; align-items: center; justify-content: center; }
  .card img { max-width: 100%; max-height: 100%; display: block; }
  .card .meta { padding: 8px 10px 10px; display: grid; gap: 2px; }
  .badges { display: flex; flex-wrap: wrap; gap: 4px 10px; margin-bottom: 2px; }
  .badge { display: inline-flex; align-items: center; gap: 6px; font-weight: 600; font-size: 13px; }
  .badge .ico { width: 18px; height: 18px; border-radius: 50%; color: #0b0b0b; background: var(--warning);
                display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; }
  .badge.no_helmet .ico { background: var(--critical); color: #fff; }
  .badge.no_vest .ico { background: var(--serious); }
  .meta .line { color: var(--ink-2); font-size: 12.5px; }
  .meta .line.muted { color: var(--muted); }
  .empty { color: var(--muted); padding: 24px; background: var(--surface); border: 1px dashed var(--line); border-radius: 10px; }

  /* lightbox */
  .lb { position: fixed; inset: 0; background: rgba(0,0,0,.82); display: none; align-items: center;
        justify-content: center; flex-direction: column; gap: 10px; padding: 16px; z-index: 10; }
  .lb.open { display: flex; }
  .lb img { max-width: 100%; max-height: calc(100vh - 110px); border-radius: 6px; }
  .lb .cap { color: #f0efec; font-size: 14px; text-align: center; }
  .lb .nav { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }
  .lb button { font: inherit; background: #2c2c2a; color: #f0efec; border: 1px solid #383835; border-radius: 6px; padding: 6px 12px; cursor: pointer; }

  @media (max-width: 600px) { .wrap { padding: 16px; } .card .img { height: 200px; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>PPE Violations</h1>
    <span class="sub" id="sub"></span>
  </header>

  <div class="filters">
    <div class="seg" id="videoFilter" role="group" aria-label="Video"></div>
    <div class="seg" id="typeFilter" role="group" aria-label="Violation type"></div>
  </div>

  <div class="tiles" id="tiles"></div>

  <h2>By video</h2>
  <div class="table-scroll">
    <table>
      <thead id="thead"></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>

  <h2 id="galleryTitle">Flagged people</h2>
  <div class="grid" id="grid"></div>
</div>

<div class="lb" id="lb" role="dialog" aria-modal="true">
  <img id="lbImg" alt="">
  <div class="cap" id="lbCap"></div>
  <div class="nav">
    <button id="lbPrev">← Prev</button>
    <button id="lbToggle">Show crop</button>
    <button id="lbNext">Next →</button>
    <button id="lbClose">Close (Esc)</button>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let DATA = { videos: [], violations: [], types: [] };
let state = { video: "all", type: "all" };
let shown = [], lbIndex = 0, lbCrop = false;

const label = t => t.replace(/_/g, " ").replace(/^./, c => c.toUpperCase());   // no_helmet -> No helmet
const icon  = t => (t.replace(/^no_/, "")[0] || "?").toUpperCase();          // no_helmet -> H
const fmtTime = s => { const m = Math.floor(s / 60); return `${m}:${(s % 60).toFixed(1).padStart(4, "0")}`; };
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const count = (list, t) => list.filter(v => v.types.includes(t)).length;

function setPressed(groupEl, value) {
  groupEl.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", b.dataset.v === value));
}

function render() {
  const inVideo = DATA.violations.filter(v => state.video === "all" || v.video === state.video);
  shown = inVideo.filter(v => state.type === "all" || v.types.includes(state.type));

  // tiles follow the video filter (type filter only narrows the gallery)
  $("tiles").innerHTML =
    `<div class="tile"><div class="label">People flagged</div><div class="value">${inVideo.length}</div></div>` +
    DATA.types.map(t => `<div class="tile"><div class="label"><span class="dot ${esc(t)}"></span>${esc(label(t))}</div>
      <div class="value">${count(inVideo, t)}</div></div>`).join("");

  $("tbody").innerHTML = DATA.videos.map(s => {
    const vs = DATA.violations.filter(v => v.video === s.video);
    return `<tr data-video="${esc(s.video)}"><td>${esc(s.video)}</td><td class="num">${fmtTime(s.duration_s)}</td>
      <td class="num">${s.people}</td>${DATA.types.map(t => `<td class="num">${count(vs, t)}</td>`).join("")}
      <td class="num">${vs.length}</td></tr>`;
  }).join("");
  $("tbody").querySelectorAll("tr").forEach(tr => tr.onclick = () => {
    state.video = tr.dataset.video; setPressed($("videoFilter"), state.video); render();
  });

  $("galleryTitle").textContent = `Flagged people (${shown.length})`;
  if (!shown.length) {
    const running = DATA.status && DATA.status.state !== "done";
    $("grid").innerHTML = `<div class="empty">${DATA.missing
      ? "No results yet. Run <code>.venv/bin/python process.py</code> first."
      : running ? "Processing… flagged people will show up here as they are found."
      : "No violations for this filter."}</div>`;
    return;
  }
  $("grid").innerHTML = shown.map((v, i) => `
    <button class="card" data-i="${i}">
      <div class="img"><img src="/output/${esc(v.crop)}?v=${v.v}" alt="Person #${v.track_id}: ${esc(v.types.map(label).join(", "))}"></div>
      <div class="meta">
        <div class="badges">${v.types.map(t =>
          `<span class="badge ${esc(t)}"><span class="ico" aria-hidden="true">${esc(icon(t))}</span>${esc(label(t))}</span>`).join("")}</div>
        <span class="line">${esc(v.video)} · ${fmtTime(v.time_s)} · Person #${v.track_id}</span>
        ${v.types.map(t => { const d = v.details[t];
          return `<span class="line muted">${esc(label(t))}: conf ${d.conf.toFixed(2)} · ${d.hits}/${d.checks} frames</span>`; }).join("")}
      </div>
    </button>`).join("");
  $("grid").querySelectorAll(".card").forEach(c => c.onclick = () => openLb(+c.dataset.i));
}

function openLb(i) {
  lbIndex = (i + shown.length) % shown.length; lbCrop = false; drawLb();
  $("lb").classList.add("open");
}
function drawLb() {
  const v = shown[lbIndex];
  $("lbImg").src = "/output/" + (lbCrop ? v.crop : v.screenshot) + "?v=" + v.v;
  $("lbToggle").textContent = lbCrop ? "Show full frame" : "Show crop";
  $("lbCap").textContent = `Person #${v.track_id} — ${v.types.map(label).join(", ")} — ${v.video} at ${fmtTime(v.time_s)} ` +
    `(frame ${v.frame}) — ${lbIndex + 1}/${shown.length}`;
}
$("lbPrev").onclick = () => openLb(lbIndex - 1);
$("lbNext").onclick = () => openLb(lbIndex + 1);
$("lbToggle").onclick = () => { lbCrop = !lbCrop; drawLb(); };
$("lbClose").onclick = () => $("lb").classList.remove("open");
$("lb").onclick = e => { if (e.target === $("lb")) $("lb").classList.remove("open"); };
document.addEventListener("keydown", e => {
  if (!$("lb").classList.contains("open")) return;
  if (e.key === "Escape") $("lb").classList.remove("open");
  if (e.key === "ArrowLeft") openLb(lbIndex - 1);
  if (e.key === "ArrowRight") openLb(lbIndex + 1);
});

function filterGroup(el, items, allLabel, key) {
  // keep the current choice if it still exists, otherwise fall back to "all"
  if (state[key] !== "all" && !items.some(([v]) => v === state[key])) state[key] = "all";
  el.innerHTML = `<button data-v="all">${allLabel}</button>` +
    items.map(([v, text]) => `<button data-v="${esc(v)}">${esc(text)}</button>`).join("");
  setPressed(el, state[key]);
  el.onclick = e => {
    const b = e.target.closest("button"); if (!b) return;
    state[key] = b.dataset.v; setPressed(el, state[key]); render();
  };
}

// Poll for new results so the page updates live while processing runs.
const POLL_MS = 1500;
let lastText = "", lastFilters = "";

async function load() {
  try {
    const text = await (await fetch("/api/data", { cache: "no-store" })).text();
    if (text !== lastText) {
      lastText = text;
      const d = JSON.parse(text);
      // remember which person the lightbox is showing, so it stays on them
      const lbOpen = $("lb").classList.contains("open");
      const lbKey = lbOpen && shown[lbIndex] ? shown[lbIndex].video + "#" + shown[lbIndex].track_id : null;

      DATA = d;
      const st = d.status || {};
      const total = d.videos.reduce((a, s) => a + s.duration_s, 0);
      $("sub").textContent = st.state === "running"
        ? `Processing ${st.stream} · frame ${st.frame}${st.total > 0 ? " / " + st.total : ""}`
        : st.state === "starting" ? "Starting…"
        : d.videos.length ? `${d.videos.length} videos · ${fmtTime(total)} of footage` : "";

      const filters = JSON.stringify([d.types, d.videos.map(s => s.video)]);
      if (filters !== lastFilters) {
        lastFilters = filters;
        $("thead").innerHTML = `<tr><th>Video</th><th class="num">Length</th><th class="num">Person tracks</th>` +
          d.types.map(t => `<th class="num">${esc(label(t))}</th>`).join("") + `<th class="num">Flagged</th></tr>`;
        filterGroup($("videoFilter"), d.videos.map(s => [s.video, s.video]), "All videos", "video");
        filterGroup($("typeFilter"), d.types.map(t => [t, label(t)]), "All types", "type");
      }
      render();

      if (lbKey) {
        const i = shown.findIndex(v => v.video + "#" + v.track_id === lbKey);
        if (i >= 0) { lbIndex = i; drawLb(); } else $("lb").classList.remove("open");
      }
    }
  } catch (e) { /* server busy or restarting - try again next tick */ }
  setTimeout(load, POLL_MS);
}
load();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    dc = CFG.get("dashboard", {})
    uvicorn.run(app, host=dc.get("host", "0.0.0.0"), port=int(dc.get("port", 9000)))
