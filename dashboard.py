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
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from verify import Reviewer, key

CONFIG_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "config.yaml")
CFG = yaml.safe_load(CONFIG_PATH.read_text())

OUT_DIR = Path(CFG["output_dir"])
if not OUT_DIR.is_absolute():
    OUT_DIR = CONFIG_PATH.parent / OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

REVIEWER = Reviewer(CFG, OUT_DIR)

app = FastAPI()
app.mount("/output", StaticFiles(directory=OUT_DIR), name="output")


def read_results():
    f = OUT_DIR / "violations.json"
    return json.loads(f.read_text()) if f.exists() else None


@app.get("/api/data")
def data():
    vlm = {"model": CFG.get("vlm", {}).get("model"), "progress": REVIEWER.progress}
    d = read_results()
    if d is None:
        return JSONResponse({"videos": [], "violations": [], "types": list(CFG["ppe"]["checks"]),
                             "missing": True, "vlm": vlm})
    # attach the VLM / human review to each person. `cleared` lists the violations where a
    # human accepted the VLM's answer (the item is there); those no longer count.
    review = REVIEWER.load()
    for v in d["violations"]:
        rv = review.get(key(v), {})
        for r in rv.values():
            if r.get("state") == "deleted":  # old name
                r["state"] = "accepted"
        v["review"] = {t: rv[t] for t in v["types"] if t in rv}
        v["cleared"] = [t for t in v["types"] if v["review"].get(t, {}).get("state") == "accepted"]
    d["vlm"] = vlm
    return d


class Job(BaseModel):
    video: str
    track_id: int
    types: list[str]  # which violations to check, e.g. ["no_vest"]


class VerifyRequest(BaseModel):
    jobs: list[Job]


@app.post("/api/verify")
def verify(req: VerifyRequest):
    d = read_results()
    if not d or d.get("status", {}).get("state") != "done":
        raise HTTPException(409, "processing is not finished yet")
    records = {key(v): v for v in d["violations"]}
    jobs = [(records[key(j.model_dump())], j.types) for j in req.jobs if key(j.model_dump()) in records]
    return {"queued": REVIEWER.start(jobs)}


class Decision(BaseModel):
    video: str
    track_id: int
    type: str    # e.g. "no_vest"
    action: str  # "keep" (violation is real) or "accept" (the VLM is right)


@app.post("/api/review")
def review(dec: Decision):
    if dec.action not in ("keep", "accept"):
        raise HTTPException(400, "action must be keep or accept")
    if dec.type not in CFG["ppe"]["checks"]:
        raise HTTPException(400, "unknown violation type")
    REVIEWER.decide(key(dec.model_dump()), dec.type, dec.action)
    return {"ok": True}


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
    --page: #f3f4f6; --surface: #ffffff; --raised: #eceef1; --line: #dfe2e6; --line-2: #c9cdd3;
    --ink: #111317; --ink-2: #42464e; --muted: #737985;
    --brand: #f5b800; --brand-ink: #1a1400; --brand-text: #8a6500;
    --good: #12924a; --on-good: #ffffff; --bad: #d92d20; --helmet: #e5372b; --vest: #f07a14; --other: #c08a00; --info: #2f6fde;
    --img-bg: #e4e6ea; --tint: 11%;
    --mono: ui-monospace, "SF Mono", "JetBrains Mono", "Cascadia Mono", Menlo, Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --page: #0c0e11; --surface: #15181c; --raised: #1c2026; --line: #262b32; --line-2: #343a43;
      --ink: #f1f2f4; --ink-2: #b9bec7; --muted: #7f8692;
      --brand: #ffc61a; --brand-ink: #1a1400; --brand-text: #ffc61a;
      --good: #2fca6c; --on-good: #04140a; --bad: #ff5147; --helmet: #ff5147; --vest: #ff9636; --other: #f2c230; --info: #5b9bff;
      --img-bg: #0a0b0d; --tint: 14%;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page: #0c0e11; --surface: #15181c; --raised: #1c2026; --line: #262b32; --line-2: #343a43;
    --ink: #f1f2f4; --ink-2: #b9bec7; --muted: #7f8692;
    --brand: #ffc61a; --brand-ink: #1a1400; --brand-text: #ffc61a;
    --good: #2fca6c; --on-good: #04140a; --bad: #ff5147; --helmet: #ff5147; --vest: #ff9636; --other: #f2c230; --info: #5b9bff;
    --img-bg: #0a0b0d; --tint: 14%;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--page); color: var(--ink);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
         -webkit-font-smoothing: antialiased; }
  button { font: inherit; }
  td.num, .mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }

  /* top bar */
  .top { background: var(--surface); border-bottom: 1px solid var(--line); position: sticky; top: 0; z-index: 5; }
  .top-in { max-width: 1440px; margin: 0 auto; padding: 12px 28px; display: flex; align-items: center; gap: 12px; }
  .mark { width: 26px; height: 26px; border-radius: 6px; flex: none;
          background: repeating-linear-gradient(-45deg, var(--brand) 0 6px, #16181c 6px 11px); }
  h1 { font-size: 17px; font-weight: 700; margin: 0; letter-spacing: -.01em; }
  h1 small { font-weight: 450; color: var(--muted); margin-left: 10px; font-size: 13px; }
  .status { margin-left: auto; display: inline-flex; align-items: center; gap: 8px; color: var(--ink-2); font-size: 13px;
            background: var(--raised); border: 1px solid var(--line); border-radius: 999px; padding: 4px 12px 4px 10px; }
  .sdot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); flex: none; }
  .sdot.ok { background: var(--good); }
  .sdot.bad { background: var(--bad); }
  .sdot.run { background: var(--brand); animation: blink 1.1s ease-in-out infinite; }
  @keyframes blink { 50% { opacity: .3; } }
  .progress { height: 3px; }
  .progress i { display: block; height: 100%; width: 0; background: var(--brand); transition: width .4s; }

  .wrap { max-width: 1440px; margin: 0 auto; padding: 22px 28px 56px; }

  /* toolbar */
  .toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
  .seg { display: inline-flex; flex-wrap: wrap; background: var(--surface); border: 1px solid var(--line); border-radius: 9px; padding: 3px; gap: 2px; }
  .seg button { color: var(--ink-2); background: none; border: 0; border-radius: 6px; padding: 5px 13px; cursor: pointer; font-weight: 500; }
  .seg button:hover { color: var(--ink); background: var(--raised); }
  .seg button[aria-pressed="true"] { background: var(--ink); color: var(--surface); font-weight: 600; }
  .spacer { margin-left: auto; }
  .linkbtn { background: none; border: 0; color: var(--ink-2); cursor: pointer; padding: 6px 6px; font-size: 13px; font-weight: 500; }
  .linkbtn:hover { color: var(--ink); text-decoration: underline; text-underline-offset: 3px; }
  .verify { font-weight: 700; color: var(--brand-ink); background: var(--brand); border: 1px solid var(--brand);
            border-radius: 9px; padding: 7px 16px; cursor: pointer; }
  .verify:hover:not(:disabled) { filter: brightness(1.06); }
  .verify:disabled { background: var(--surface); border-color: var(--line); color: var(--muted); font-weight: 600; cursor: default; }
  .verify .sub { font-family: var(--mono); font-weight: 500; opacity: .75; margin-left: 8px; font-size: 12px; }

  /* stat tiles */
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 18px; }
  .tile { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px 14px;
          border-top: 3px solid var(--line-2); }
  .tile.t-no_helmet { border-top-color: var(--helmet); }
  .tile.t-no_vest { border-top-color: var(--vest); }
  .tile.t-review { border-top-color: var(--brand); }
  .tile .label { color: var(--ink-2); font-size: 13px; font-weight: 600; display: flex; align-items: center; gap: 7px; }
  .tile .value { font-family: var(--mono); font-size: 30px; font-weight: 700; margin-top: 6px; letter-spacing: -.02em; line-height: 1.1; }
  .tile.t-no_helmet .value { color: var(--helmet); }
  .tile.t-no_vest .value { color: var(--vest); }
  .tile .value small { font-size: 13px; color: var(--muted); font-weight: 500; margin-left: 8px; letter-spacing: 0; }
  .tile .share { height: 6px; background: var(--raised); border-radius: 3px; margin-top: 12px; overflow: hidden; display: flex; }
  .tile .share i { display: block; height: 100%; background: var(--ink-2); }
  .tile .share i.no_helmet { background: var(--helmet); }
  .tile .share i.no_vest { background: var(--vest); }
  .tile .share i.s-ok { background: var(--good); }
  .tile .share i.s-bad { background: var(--bad); }
  .tile .share i.s-fix { background: var(--info); }
  .legend { display: flex; flex-wrap: wrap; gap: 4px 12px; margin-top: 8px; font-size: 12px; color: var(--ink-2); }
  .legend span { display: inline-flex; align-items: center; gap: 5px; }
  .legend b { font-family: var(--mono); font-weight: 600; color: var(--ink); }
  .sw { width: 9px; height: 9px; border-radius: 2px; background: var(--other); flex: none; }
  .sw.no_helmet { background: var(--helmet); } .sw.no_vest { background: var(--vest); }
  .sw.s-ok { background: var(--good); } .sw.s-bad { background: var(--bad); } .sw.s-fix { background: var(--info); }
  .sw.s-none { background: var(--line-2); }

  /* section titles */
  h2 { font-size: 15px; color: var(--ink); margin: 32px 0 12px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
  h2 .count { font-family: var(--mono); font-size: 12px; font-weight: 600; color: var(--ink-2); background: var(--raised);
              border: 1px solid var(--line); border-radius: 999px; padding: 1px 8px; }

  /* per-video table */
  .table-box { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 10px 16px; border-bottom: 1px solid var(--line); white-space: nowrap; }
  th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
  td { font-weight: 500; }
  td.num, th.num { text-align: right; }
  td.num.hot-no_helmet { color: var(--helmet); font-weight: 700; }
  td.num.hot-no_vest { color: var(--vest); font-weight: 700; }
  td.num.hot { color: var(--ink); font-weight: 700; }
  td.num.zero { color: var(--muted); }
  tr:last-child td { border-bottom: 0; }
  tbody tr { cursor: pointer; }
  tbody tr:hover td { background: var(--raised); }

  /* gallery */
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 16px; }
  .card { background: var(--surface); border: 2px solid var(--line); border-radius: 12px; overflow: hidden;
          display: flex; flex-direction: column; position: relative; transition: border-color .15s; }
  .card:hover { border-color: var(--line-2); }
  .card.verified { border-color: var(--good); }
  .card.rejected { border-color: var(--bad); }
  .card.cleared { border-color: var(--info); }
  .card.picked { border-color: var(--brand); }
  .card .open { cursor: pointer; display: flex; flex-direction: column; text-align: left; color: inherit;
                background: none; border: 0; padding: 0; }
  .card .open:focus-visible { outline: 2px solid var(--brand); outline-offset: -2px; }
  .card .img { aspect-ratio: 4 / 5; background: var(--img-bg); display: flex; align-items: center; justify-content: center; }
  .card img { max-width: 100%; max-height: 100%; display: block; }
  .card.cleared .img img { opacity: .7; }

  .pick { position: absolute; top: 9px; left: 9px; z-index: 1; width: 28px; height: 28px; border-radius: 7px;
          background: rgba(10,11,13,.65); display: flex; align-items: center; justify-content: center; cursor: pointer; }
  .pick input { width: 16px; height: 16px; margin: 0; cursor: pointer; accent-color: var(--brand); }
  .chip { position: absolute; top: 9px; right: 9px; z-index: 1; font-size: 12px; font-weight: 700;
          padding: 4px 10px; border-radius: 999px; color: #fff; background: rgba(10,11,13,.75); display: inline-flex;
          align-items: center; gap: 6px; }
  .chip.ok { background: var(--good); color: var(--on-good); }
  .chip.bad { background: var(--bad); color: #fff; }
  .chip.fix { background: var(--info); color: #fff; }

  .body { padding: 12px 14px 14px; display: grid; gap: 10px; }
  .title { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
  .title b { font-weight: 700; font-size: 15px; }
  .title .when { font-family: var(--mono); font-size: 12px; color: var(--muted); }
  .tags { display: flex; flex-wrap: wrap; gap: 6px; }
  .tag { font-size: 12px; font-weight: 700; padding: 3px 9px; border-radius: 6px; display: inline-flex; align-items: center; gap: 5px;
         color: #fff; background: var(--other); }
  .tag.no_helmet { background: var(--helmet); }
  .tag.no_vest { background: var(--vest); color: #1d0e00; }
  .tag.fixed { background: color-mix(in srgb, var(--info) var(--tint), transparent); color: var(--info);
               box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--info) 45%, transparent); }
  .confs { display: grid; gap: 6px; }
  .conf { display: grid; grid-template-columns: 1fr auto; gap: 3px 8px; align-items: center; font-size: 12px; color: var(--ink-2); }
  .conf .bar { grid-column: 1 / -1; height: 5px; background: var(--raised); border-radius: 3px; overflow: hidden; }
  .conf .bar i { display: block; height: 100%; background: var(--other); border-radius: 3px; }
  .conf .bar i.no_helmet { background: var(--helmet); }
  .conf .bar i.no_vest { background: var(--vest); }
  .conf .v { font-family: var(--mono); color: var(--ink); }
  .conf.cleared { opacity: .45; }
  .conf.cleared > span:first-child { text-decoration: line-through; }

  .review { padding: 0 10px 10px; display: grid; gap: 8px; font-size: 12.5px; }
  .row { display: grid; gap: 5px; padding: 9px 11px; border-radius: 8px; border-left: 3px solid var(--line-2); background: var(--raised); }
  .row.ok { border-left-color: var(--good); background: color-mix(in srgb, var(--good) var(--tint), var(--surface)); }
  .row.bad { border-left-color: var(--bad); background: color-mix(in srgb, var(--bad) var(--tint), var(--surface)); }
  .row.fix { border-left-color: var(--info); background: color-mix(in srgb, var(--info) var(--tint), var(--surface)); }
  .row.wait { border-left-color: var(--brand); }
  .rhead { display: flex; align-items: center; gap: 7px; font-weight: 700; font-size: 13px; }
  .rhead .rtype { margin-left: auto; font-weight: 500; color: var(--muted); font-size: 12px; white-space: nowrap; }
  .row.ok .rhead { color: var(--good); } .row.bad .rhead { color: var(--bad); } .row.fix .rhead { color: var(--info); }
  .row.wait .rhead { color: var(--ink-2); }
  .why { margin: 0; color: var(--ink-2); line-height: 1.4; }
  .why .src { font-family: var(--mono); font-size: 10.5px; font-weight: 700; color: var(--muted); margin-right: 6px;
              border: 1px solid var(--line-2); border-radius: 4px; padding: 0 4px; }
  .act { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 3px; }
  .act button { padding: 6px 8px; border-radius: 7px; cursor: pointer; font-weight: 700; font-size: 12.5px;
                border: 1px solid var(--line-2); background: var(--surface); color: var(--ink); }
  .act button:hover { border-color: var(--ink-2); }
  .act button.acc { background: var(--good); border-color: var(--good); color: var(--on-good); }
  .act button.acc:hover { filter: brightness(1.07); }
  .act button:disabled { opacity: .5; cursor: default; }

  .empty { color: var(--muted); padding: 28px; background: var(--surface); border: 1px dashed var(--line-2); border-radius: 12px; }
  .empty code { font-family: var(--mono); color: var(--ink-2); }

  /* lightbox */
  .lb { position: fixed; inset: 0; background: rgba(6,7,9,.92); display: none; align-items: center;
        justify-content: center; flex-direction: column; gap: 12px; padding: 20px; z-index: 10; }
  .lb.open { display: flex; }
  .lb img { max-width: 100%; max-height: calc(100vh - 120px); border-radius: 8px; }
  .lb .cap { color: #dfe2e6; font-size: 13px; text-align: center; font-family: var(--mono); }
  .lb .nav { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }
  .lb button { background: #1c2026; color: #f1f2f4; border: 1px solid #343a43; border-radius: 7px; padding: 6px 12px; cursor: pointer; font-weight: 600; }
  .lb button:hover { border-color: #ffc61a; }

  @media (max-width: 640px) { .wrap, .top-in { padding-left: 16px; padding-right: 16px; } h1 small { display: none; } }
</style>
</head>
<body>
<div class="top">
  <div class="top-in">
    <span class="mark" aria-hidden="true"></span>
    <h1>PPE Violations<small>detection + VLM verification</small></h1>
    <div class="status" id="sub"></div>
  </div>
  <div class="progress"><i id="progress"></i></div>
</div>

<div class="wrap">
  <div class="toolbar">
    <div class="seg" id="videoFilter" role="group" aria-label="Video"></div>
    <div class="seg" id="typeFilter" role="group" aria-label="Violation type"></div>
    <span class="spacer"></span>
    <button class="linkbtn" id="pickAll" hidden>Select all</button>
    <button class="linkbtn" id="pickNone" hidden>Clear selection</button>
    <button class="verify" id="verifyBtn" disabled>Verify</button>
  </div>

  <div class="tiles" id="tiles"></div>

  <h2>By video</h2>
  <div class="table-box">
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
let picked = new Set();   // "video#track_id" of cards ticked for verification
let shown = [], lbIndex = 0, lbCrop = false;

const label = t => t.replace(/_/g, " ").replace(/^./, c => c.toUpperCase());   // no_helmet -> No helmet
const fmtTime = s => { const m = Math.floor(s / 60); return `${m}:${(s % 60).toFixed(1).padStart(4, "0")}`; };
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// violations still standing (not cleared by a reviewer who accepted the VLM's answer)
const active = v => v.types.filter(t => !v.cleared.includes(t));
const item = t => t.replace(/^no_/, "");                                       // no_helmet -> helmet
const count = (list, t) => list.filter(v => active(v).includes(t)).length;

function setPressed(groupEl, value) {
  groupEl.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", b.dataset.v === value));
}

function render() {
  const inVideo = DATA.violations.filter(v => state.video === "all" || v.video === state.video);
  shown = inVideo.filter(v => state.type === "all" || v.types.includes(state.type));
  const flagged = inVideo.filter(v => active(v).length).length;

  // tiles follow the video filter (type filter only narrows the gallery)
  const tracked = DATA.videos.filter(s => state.video === "all" || s.video === state.video).reduce((a, s) => a + s.people, 0);
  const pct = (n, d) => d ? Math.round(100 * n / d) : 0;
  // VLM review status of every violation in this video filter
  const rs = { ok: 0, bad: 0, fix: 0, none: 0 };
  inVideo.forEach(v => v.types.forEach(t => {
    const st = (v.review[t] || {}).state;
    rs[st === "verified" || st === "kept" ? "ok" : st === "rejected" ? "bad" : st === "accepted" ? "fix" : "none"]++;
  }));
  const all = rs.ok + rs.bad + rs.fix + rs.none;
  const reviewTile = `<div class="tile t-review"><div class="label">VLM review</div>
       <div class="value">${all - rs.none}<small>of ${all} checked</small></div>
       <div class="share">${["ok", "bad", "fix"].map(k => `<i class="s-${k}" style="width:${pct(rs[k], all)}%"></i>`).join("")}</div>
       <div class="legend"><span><span class="sw s-ok"></span><b>${rs.ok}</b> confirmed</span>
         <span><span class="sw s-bad"></span><b>${rs.bad}</b> needs review</span>
         <span><span class="sw s-fix"></span><b>${rs.fix}</b> corrected</span></div></div>`;
  $("tiles").innerHTML =
    `<div class="tile"><div class="label">People flagged</div>
       <div class="value">${flagged}<small>of ${tracked} tracked</small></div>
       <div class="share"><i style="width:${pct(flagged, tracked)}%"></i></div></div>` +
    DATA.types.map(t => { const n = count(inVideo, t);
      return `<div class="tile t-${esc(t)}"><div class="label"><span class="sw ${esc(t)}"></span>${esc(label(t))}</div>
       <div class="value">${n}<small>${pct(n, flagged)}%</small></div>
       <div class="share"><i class="${esc(t)}" style="width:${pct(n, flagged)}%"></i></div></div>`; }).join("") + reviewTile;

  $("tbody").innerHTML = DATA.videos.map(s => {
    const vs = DATA.violations.filter(v => v.video === s.video);
    return `<tr data-video="${esc(s.video)}"><td>${esc(s.video)}</td><td class="num">${fmtTime(s.duration_s)}</td>
      <td class="num">${s.people}</td>${DATA.types.map(t => { const n = count(vs, t);
        return `<td class="num ${n ? "hot-" + esc(t) : "zero"}">${n}</td>`; }).join("")}
      ${(n => `<td class="num ${n ? "hot" : "zero"}">${n}</td>`)(vs.filter(v => active(v).length).length)}</tr>`;
  }).join("");
  $("tbody").querySelectorAll("tr").forEach(tr => tr.onclick = () => {
    state.video = tr.dataset.video; setPressed($("videoFilter"), state.video); render();
  });

  renderVerify();
  $("galleryTitle").innerHTML = `Flagged people <span class="count">${shown.length}</span>`;
  if (!shown.length) {
    const running = DATA.status && DATA.status.state !== "done";
    $("grid").innerHTML = `<div class="empty">${DATA.missing
      ? "No results yet. Run <code>.venv/bin/python process.py</code> first."
      : running ? "Processing… flagged people will show up here as they are found."
      : "No violations for this filter."}</div>`;
    return;
  }
  $("grid").innerHTML = shown.map((v, i) => {
    const k = pk(v), canPick = verifiable(v).length > 0;
    return `
    <div class="card ${frame(v)} ${picked.has(k) ? "picked" : ""}" data-k="${esc(k)}" data-video="${esc(v.video)}" data-tid="${v.track_id}">
      ${canPick ? `<label class="pick" title="Select for verification"><input type="checkbox" ${picked.has(k) ? "checked" : ""}></label>` : ""}
      ${chipHtml(v)}
      <button class="open" data-i="${i}">
        <div class="img"><img src="/output/${esc(v.crop)}?v=${v.v}" alt="Person #${v.track_id}: ${esc(v.types.map(label).join(", "))}" loading="lazy"></div>
        <div class="body">
          <div class="title"><b>Person #${v.track_id}</b><span class="when">${esc(v.video)} · ${fmtTime(v.time_s)}</span></div>
          <div class="tags">${v.types.map(t => v.cleared.includes(t)
            ? `<span class="tag fixed" title="Corrected by reviewer (PPE model said: ${esc(label(t))})">✓ ${esc(label(item(t)))}</span>`
            : `<span class="tag ${esc(t)}">${esc(label(t))}</span>`).join("")}</div>
          <div class="confs">${v.types.map(t => { const d = v.details[t];
            return `<div class="conf ${v.cleared.includes(t) ? "cleared" : ""}" title="${esc(label(t))}: model confidence ${d.conf.toFixed(2)}, seen in ${d.hits} of ${d.checks} checked frames">
              <span>${esc(label(t))}</span><span class="v">${d.conf.toFixed(2)} · ${d.hits}/${d.checks}</span>
              <span class="bar"><i class="${esc(t)}" style="width:${Math.round(d.conf * 100)}%"></i></span></div>`; }).join("")}</div>
        </div>
      </button>
      ${reviewHtml(v)}
    </div>`; }).join("");
  $("grid").querySelectorAll(".open").forEach(b => b.onclick = () => openLb(+b.dataset.i));
  $("grid").querySelectorAll(".pick input").forEach(cb => cb.onchange = () => {
    const k = cb.closest(".card").dataset.k;
    cb.checked ? picked.add(k) : picked.delete(k);
    render();
  });
  $("grid").querySelectorAll(".act button").forEach(b => b.onclick = () => {
    const card = b.closest(".card");
    decide(card.dataset.video, +card.dataset.tid, b.dataset.type, b.dataset.act, b);
  });
}

const pk = v => v.video + "#" + v.track_id;
// Which violations the current page is about: on "No vest" only no_vest, on "All types" all of them.
const pageTypes = v => state.type === "all" ? v.types : v.types.filter(t => t === state.type);
// Of those, the ones the VLM has not answered yet (or failed on).
const verifiable = v => pageTypes(v).filter(t => !v.review[t] || v.review[t].state === "error");

// Card frame: red if any violation on this page needs a human, green if all are confirmed.
// "cleared" if a reviewer accepted the VLM on every violation on this page (nothing left).
function frame(v) {
  const st = pageTypes(v).map(t => (v.review[t] || {}).state);
  if (st.includes("rejected")) return "rejected";
  if (st.length && st.every(x => x === "accepted")) return "cleared";
  if (st.length && st.every(x => ["verified", "kept", "accepted"].includes(x))) return "verified";
  return "";
}

// Status chip on the image, matching the frame colour.
function chipHtml(v) {
  const f = frame(v), st = pageTypes(v).map(t => (v.review[t] || {}).state);
  if (f === "rejected") return `<span class="chip bad">! Needs review</span>`;
  if (f === "cleared") return `<span class="chip fix">✓ Cleared</span>`;
  if (f === "verified") return `<span class="chip ok">✓ ${st.every(x => x === "kept") ? "Kept" : "Verified"}</span>`;
  if (st.includes("pending")) return `<span class="chip"><span class="sdot run"></span>Checking</span>`;
  return "";
}

// The VLM / human review rows under each card, one per violation on this page.
function reviewHtml(v) {
  const many = pageTypes(v).length > 1;
  const rows = pageTypes(v).filter(t => v.review[t]).map(t => {
    const r = v.review[t], type = many ? `<span class="rtype">${esc(label(t))}</span>` : "";
    const why = r.reason ? `<p class="why"><span class="src">VLM</span>${esc(r.reason)}</p>` : "";
    const head = text => `<div class="rhead">${text}${type}</div>`;
    switch (r.state) {
      case "pending":  return `<div class="row wait">${head(`<span class="sdot run"></span>Waiting for VLM`)}</div>`;
      case "verified": return `<div class="row ok">${head("✓ Verified by VLM")}${why}</div>`;
      case "kept":     return `<div class="row ok">${head("✓ Violation kept by reviewer")}${why}</div>`;
      case "accepted": return `<div class="row fix">${head(`✓ Corrected: has a ${esc(item(t))}`)}${why}</div>`;
      case "error":    return `<div class="row">${head("VLM failed. Verify again to retry")}
                         <p class="why">${esc(r.reason || "")}</p></div>`;
      case "rejected": return `<div class="row bad">${head(`! VLM sees a ${esc(item(t))}`)}${why}
                         <div class="act"><button class="acc" data-act="accept" data-type="${esc(t)}" title="The VLM is right: change the tag to ${esc(label(item(t)))}">Accept VLM</button>
                         <button class="keep" data-act="keep" data-type="${esc(t)}" title="The VLM is wrong: keep ${esc(label(t))}">Keep violation</button></div></div>`;
    }
    return "";
  }).join("");
  return rows ? `<div class="review">${rows}</div>` : "";
}

// Verify button: ticked cards if any, otherwise everything on this page that is not verified yet.
function verifyTargets() {
  const open = shown.filter(v => verifiable(v).length);
  const sel = open.filter(v => picked.has(pk(v)));
  return sel.length ? sel : open;
}
function renderVerify() {
  // forget ticks on cards that are gone from this page or already answered
  const ok = new Set(shown.filter(v => verifiable(v).length).map(pk));
  picked.forEach(k => { if (!ok.has(k)) picked.delete(k); });

  const btn = $("verifyBtn"), p = DATA.vlm && DATA.vlm.progress, ready = DATA.status && DATA.status.state === "done";
  const n = verifyTargets().length, what = state.type === "all" ? "" : " for " + label(state.type).toLowerCase();
  $("pickAll").hidden = !ready || !ok.size || picked.size === ok.size;
  $("pickNone").hidden = !picked.size;
  btn.title = ready ? (DATA.vlm && DATA.vlm.model ? "Model: " + DATA.vlm.model : "") : "Available when processing is done";
  btn.disabled = !ready || !n;
  btn.textContent = !ready ? "Verify"
    : n ? `Verify ${n}${picked.size ? " selected" : ""}${what}`
    : shown.some(v => pageTypes(v).some(t => (v.review[t] || {}).state === "pending")) ? "Verifying…"
    : "All verified on this page";
  if (p) btn.insertAdjacentHTML("beforeend", `<span class="sub">${p.done}/${p.total}</span>`);
}
$("pickAll").onclick = () => { shown.filter(v => verifiable(v).length).forEach(v => picked.add(pk(v))); render(); };
$("pickNone").onclick = () => { picked.clear(); render(); };
$("verifyBtn").onclick = async () => {
  const jobs = verifyTargets().map(v => ({ video: v.video, track_id: v.track_id, types: verifiable(v) }));
  $("verifyBtn").disabled = true;
  picked.clear();
  try {
    await fetch("/api/verify", { method: "POST", headers: { "Content-Type": "application/json" },
                                 body: JSON.stringify({ jobs }) });
  } catch (e) {}
  refresh();
};

async function decide(video, track_id, type, action, btn) {
  btn.disabled = true;
  try {
    await fetch("/api/review", { method: "POST", headers: { "Content-Type": "application/json" },
                                 body: JSON.stringify({ video, track_id, type, action }) });
  } catch (e) {}
  refresh();
}

function openLb(i) {
  lbIndex = (i + shown.length) % shown.length; lbCrop = false; drawLb();
  $("lb").classList.add("open");
}
function drawLb() {
  const v = shown[lbIndex];
  $("lbImg").src = "/output/" + (lbCrop ? v.crop : v.screenshot) + "?v=" + v.v;
  $("lbToggle").textContent = lbCrop ? "Show full frame" : "Show crop";
  $("lbCap").textContent = `Person #${v.track_id} — ${active(v).map(label).join(", ") || "cleared"} — ${v.video} at ${fmtTime(v.time_s)} ` +
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

async function refresh() {
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
      const p = d.vlm && d.vlm.progress;
      const [dot, msg] = st.state === "running"
          ? ["run", `Processing ${esc(st.stream)} · <span class="mono">${st.frame}${st.total > 0 ? " / " + st.total : ""}</span>`]
        : st.state === "starting" ? ["run", "Starting…"]
        : p ? ["run", `VLM checking · <span class="mono">${p.done} / ${p.total}</span>`]
        : d.videos.length ? ["ok", `${d.videos.length} video${d.videos.length > 1 ? "s" : ""} · <span class="mono">${fmtTime(total)}</span> of footage`]
        : ["", "No results yet"];
      $("sub").innerHTML = `<span class="sdot ${dot}"></span>${msg}`;
      $("progress").style.width = st.state === "running" && st.total > 0 ? `${100 * st.frame / st.total}%`
        : p && p.total ? `${100 * p.done / p.total}%` : "0";

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
}
async function poll() { await refresh(); setTimeout(poll, POLL_MS); }
poll();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    dc = CFG.get("dashboard", {})
    uvicorn.run(app, host=dc.get("host", "0.0.0.0"), port=int(dc.get("port", 9000)))
