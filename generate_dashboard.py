#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "jinja2",
# ]
# ///
"""
Fast dashboard + slideshow generator for training runs.

Replaces the bash generate_html / generate_slideshow_html /
generate_jacobi_slideshow_html functions that hang on large runs.

Usage:
    python generate_dashboard.py runs/1/
    # or via uv:
    uv run generate_dashboard.py runs/1/

What it does (in order):
    1. Scan the run directory ONCE with os.scandir (not find)
    2. Classify files into buckets:
       - summary plots  (plots/*.png, training_plot.png, etc.)
       - training_plot history  (training_plot-*.png)
       - jacobi step images    (jacobi_images/jacobi_step*_layer*.png)
       - jacobi 4D slice images (jacobi_images/jacobi_step*_layer*_d*xd*.png)
       - epoch txts/csvs       (epoch_NNNN.txt/csv)
       - other csvs, txts, py files
    3. Generate index.html     — dashboard with only summary plots
    4. Generate slideshow.html — training plot history (subsampled)
    5. Generate jacobi.html    — jacobi field history (subsampled)
    6. Generate 4d_jacobi.html — 4D slice jacobi history (subsampled, grouped)
    All HTML is written via Jinja2 templates (embedded).
"""

from __future__ import annotations

import html as html_mod
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ── uv bootstrap (same pattern as train.py) ─────────────────────────────
def _ensure_uv_env():
    if not os.environ.get("UV_EXCLUDE_NEWER"):
        past = (datetime.now(timezone.utc) - timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
        os.environ["UV_EXCLUDE_NEWER"] = past
        try:
            os.execvpe("uv", ["uv", "run", "--quiet", sys.argv[0]] + sys.argv[1:], os.environ)
        except FileNotFoundError:
            pass  # no uv, hope jinja2 is installed

_ensure_uv_env()

from jinja2 import Environment, BaseLoader

# ═════════════════════════════════════════════════════════════════════════
# 1. SCAN — single pass over the directory tree
# ═════════════════════════════════════════════════════════════════════════

RE_TRAINING_PLOT_HIST = re.compile(r'^training_plot-\d+\.png$')
RE_JACOBI_IMG = re.compile(r'^jacobi_step(\d+)_layer(\d+)\.png$')
RE_JACOBI_4D_IMG = re.compile(r'^jacobi_step(\d+)_layer(\d+)_d(\d+)xd(\d+)\.png$')
RE_EPOCH_TXT = re.compile(r'^epoch_(\d+)\.txt$')
RE_EPOCH_CSV = re.compile(r'^epoch_(\d+)\.csv$')

class RunFiles:
    """All classified files from a single scan."""
    def __init__(self):
        self.summary_images: list[str] = []      # plots to show in dashboard
        self.training_hist: list[str] = []        # training_plot-*.png sorted
        self.jacobi_images: list[tuple[int, int, str]] = []  # (step, layer, relpath)
        self.jacobi_4d_images: list[tuple[int, int, int, int, str]] = []  # (step, layer, dim_a, dim_b, relpath)
        self.epoch_txts: list[str] = []
        self.epoch_csvs: list[str] = []
        self.other_csvs: list[str] = []
        self.other_txts: list[str] = []
        self.py_files: list[str] = []
        self.started_at: str = ""
        self.has_current_plot: bool = False

def scan_run_dir(run_dir: Path) -> RunFiles:
    """Walk the run directory once, classify everything."""
    rf = RunFiles()

    # Read started_at.txt
    sa_path = run_dir / "started_at.txt"
    if sa_path.is_file():
        rf.started_at = sa_path.read_text().strip().split("\n")[0].strip()

    if (run_dir / "training_plot.png").is_file():
        rf.has_current_plot = True

    for entry in _walk_fast(run_dir, max_depth=3):
        rel = entry.relative_to(run_dir).as_posix()
        name = entry.name
        suffix = entry.suffix.lower()

        # Skip npz, hidden files, the HTML outputs themselves
        if suffix == '.npz' or name.startswith('.') or suffix == '.html':
            continue

        # ── Training plot history ───────────────────────────────
        if RE_TRAINING_PLOT_HIST.match(name) and entry.parent == run_dir:
            rf.training_hist.append(rel)
            continue

        # ── Jacobi 4D slice images (must check BEFORE plain jacobi) ─
        m_jac4d = RE_JACOBI_4D_IMG.match(name)
        if m_jac4d and 'jacobi_images' in rel:
            step = int(m_jac4d.group(1))
            layer = int(m_jac4d.group(2))
            dim_a = int(m_jac4d.group(3))
            dim_b = int(m_jac4d.group(4))
            rf.jacobi_4d_images.append((step, layer, dim_a, dim_b, rel))
            continue

        # ── Jacobi images (plain, non-4D) ──────────────────────
        m_jac = RE_JACOBI_IMG.match(name)
        if m_jac and 'jacobi_images' in rel:
            step, layer = int(m_jac.group(1)), int(m_jac.group(2))
            rf.jacobi_images.append((step, layer, rel))
            continue

        # ── Epoch files ─────────────────────────────────────────
        if RE_EPOCH_TXT.match(name):
            rf.epoch_txts.append(rel)
            continue
        if RE_EPOCH_CSV.match(name):
            rf.epoch_csvs.append(rel)
            continue

        # ── Summary images (everything else that's an image) ────
        if suffix in ('.png', '.jpg', '.jpeg', '.svg'):
            # Exclude jacobi step images that aren't in jacobi_images/
            if name.startswith('jacobi_step'):
                continue
            rf.summary_images.append(rel)
            continue

        # ── Other files ─────────────────────────────────────────
        if name == 'started_at.txt':
            continue
        if suffix == '.csv':
            rf.other_csvs.append(rel)
        elif suffix == '.txt':
            rf.other_txts.append(rel)
        elif suffix == '.py':
            rf.py_files.append(rel)

    # Sort everything
    rf.summary_images.sort()
    rf.training_hist.sort()
    rf.jacobi_images.sort()
    rf.jacobi_4d_images.sort()
    rf.epoch_txts.sort()
    rf.epoch_csvs.sort()
    rf.other_csvs.sort()
    rf.other_txts.sort()
    rf.py_files.sort()

    return rf


def _walk_fast(root: Path, max_depth: int = 3) -> list[Path]:
    """os.scandir-based walk, much faster than find for large dirs."""
    results = []
    _walk_recurse(root, results, 0, max_depth)
    return results

def _walk_recurse(directory: Path, results: list, depth: int, max_depth: int):
    if depth > max_depth:
        return
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False):
                    results.append(Path(entry.path))
                elif entry.is_dir(follow_symlinks=False) and not entry.name.startswith('.'):
                    _walk_recurse(Path(entry.path), results, depth + 1, max_depth)
    except PermissionError:
        pass


# ═════════════════════════════════════════════════════════════════════════
# 2. SUBSAMPLE — pick evenly spaced frames
# ═════════════════════════════════════════════════════════════════════════

def subsample(items: list, max_count: int) -> list:
    """Return at most max_count evenly-spaced items, always including first+last."""
    n = len(items)
    if n <= max_count:
        return list(items)
    step = n / max_count
    indices = set()
    indices.add(0)
    indices.add(n - 1)
    for i in range(max_count):
        indices.add(min(int(i * step), n - 1))
    return [items[i] for i in sorted(indices)]


# ═════════════════════════════════════════════════════════════════════════
# 3. EPOCH PARSING — fast awk-like pass in Python
# ═════════════════════════════════════════════════════════════════════════

def parse_epoch_files(run_dir: Path, epoch_txts: list[str]) -> tuple[list[dict], dict[int, list[dict]]]:
    """
    Parse epoch_NNNN.txt files.
    Returns:
        overview: [{epoch, correct, total, pct}, ...]
        details:  {epoch_num: [{sample, params, expected, predicted, correct}, ...]}
    """
    overview = []
    details = {}

    for rel in epoch_txts:
        m = RE_EPOCH_TXT.match(Path(rel).name)
        if not m:
            continue
        epoch_num = int(m.group(1))
        fpath = run_dir / rel

        samples = []
        current = {}
        try:
            text = fpath.read_text(errors='replace')
        except Exception:
            continue

        for line in text.splitlines():
            line = line.strip()
            if line.startswith('params:'):
                current['params'] = line[7:].strip()
            elif line.startswith('expected:'):
                current['expected'] = line[9:].strip()
            elif line.startswith('predicted:'):
                current['predicted'] = line[10:].strip()
            elif line.startswith('correct:'):
                current['correct'] = line[8:].strip() == 'True'
                samples.append(current)
                current = {}

        if samples:
            correct = sum(1 for s in samples if s.get('correct'))
            total = len(samples)
            pct = int(100 * correct / total) if total > 0 else 0
            overview.append({'epoch': epoch_num, 'correct': correct, 'total': total, 'pct': pct})
            details[epoch_num] = samples

    overview.sort(key=lambda x: x['epoch'])
    return overview, details


# ═════════════════════════════════════════════════════════════════════════
# 4. ELAPSED TIME
# ═════════════════════════════════════════════════════════════════════════

def compute_elapsed(started_at: str) -> str:
    """Try to parse started_at and compute elapsed time string."""
    if not started_at:
        return ""
    # Try common formats
    for fmt in ("%Y-%m-%d %H:%M:%S %Z", "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S", "%a %b %d %H:%M:%S %Z %Y"):
        try:
            # Handle CEST/CET
            cleaned = started_at.replace(' CEST', '+0200').replace(' CET', '+0100')
            dt = datetime.strptime(cleaned, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            diff = datetime.now(timezone.utc) - dt
            secs = int(diff.total_seconds())
            if secs < 0:
                return ""
            days, rem = divmod(secs, 86400)
            hours, rem = divmod(rem, 3600)
            mins, _ = divmod(rem, 60)
            if days > 0:
                return f"{days}d {hours}h {mins}m"
            elif hours > 0:
                return f"{hours}h {mins}m"
            else:
                return f"{mins}m"
        except (ValueError, OverflowError):
            continue
    return ""


# ═════════════════════════════════════════════════════════════════════════
# 5. FILE PREVIEWS — read first N lines of text files for inline display
# ═════════════════════════════════════════════════════════════════════════

def build_previews(run_dir: Path, file_list: list[str], max_lines: int = 30) -> dict[str, str]:
    """Read first max_lines of each file, HTML-escaped, for inline preview."""
    previews = {}
    for rel in file_list:
        try:
            text = (run_dir / rel).read_text(errors='replace')
            lines = text.splitlines()[:max_lines]
            previews[rel] = html_mod.escape('\n'.join(lines))
        except Exception:
            previews[rel] = '(could not read)'
    return previews


# ═════════════════════════════════════════════════════════════════════════
# 6. HTML TEMPLATES (Jinja2, embedded)
# ═════════════════════════════════════════════════════════════════════════

JINJA_ENV = Environment(loader=BaseLoader(), autoescape=False)

# Register custom filters BEFORE compiling any templates
JINJA_ENV.filters['basename'] = lambda s: Path(s).name

# ── Shared slideshow CSS+JS (old version's polished UI) ─────────────────
# This is injected into each slideshow template. It provides:
# - Custom scrubber bar with gradient fill + draggable thumb
# - Double-buffered image transitions (for non-jacobi slideshows)
# - Full keyboard navigation (arrows, 0-9, Home/End, PageUp/Down, +/-, Space)
# - Scroll wheel + touch swipe support
# - Preload cache with sliding window

SLIDESHOW_SHARED_CSS = r'''
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #08090d; color: #e8eaf6;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    height: 100vh; display: flex; flex-direction: column;
    overflow: hidden; user-select: none;
  }
  .toolbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.5rem 1.5rem; background: #12152a;
    border-bottom: 1px solid #1e2340; flex-shrink: 0; z-index: 10;
  }
  .toolbar .title {
    font-size: 1rem; font-weight: 700;
    background: linear-gradient(135deg, #7c5cfc, #00d4aa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .toolbar .controls { display: flex; align-items: center; gap: 0.6rem; }
  .toolbar button {
    background: #1e2340; border: 1px solid #2a2f55; color: #e8eaf6;
    padding: 0.35rem 0.9rem; border-radius: 6px; cursor: pointer;
    font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;
    transition: background 0.15s, border-color 0.15s;
  }
  .toolbar button:hover { background: #2a2f55; border-color: #7c5cfc; }
  .toolbar button:active { background: #7c5cfc; color: #fff; }
  .toolbar button.playing { background: #00d4aa; color: #08090d; border-color: #00d4aa; }
  .counter {
    font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
    color: #6b70a0; min-width: 140px; text-align: center;
  }
  .counter .current { color: #7c5cfc; font-weight: 700; }
  .scrubber-container {
    flex-shrink: 0; height: 28px; background: #0d0f16;
    cursor: pointer; position: relative; display: flex;
    align-items: center; border-bottom: 1px solid #1e2340;
  }
  .scrubber-track {
    position: absolute; left: 12px; right: 12px; height: 6px;
    background: #1e2340; border-radius: 3px; overflow: hidden;
  }
  .scrubber-track:hover, .scrubber-container.dragging .scrubber-track { height: 10px; }
  .scrubber-fill {
    height: 100%; background: linear-gradient(90deg, #7c5cfc, #00d4aa);
    border-radius: 3px; pointer-events: none; will-change: width;
  }
  .scrubber-thumb {
    position: absolute; width: 16px; height: 16px;
    background: #e8eaf6; border: 2px solid #7c5cfc;
    border-radius: 50%; top: 50%; transform: translate(-50%, -50%);
    pointer-events: none; will-change: left;
    box-shadow: 0 0 8px rgba(124,92,252,0.4); transition: transform 0.1s;
  }
  .scrubber-container.dragging .scrubber-thumb {
    transform: translate(-50%, -50%) scale(1.3); background: #7c5cfc;
  }
  .slide-container {
    flex: 1; display: flex; align-items: center; justify-content: center;
    overflow: hidden; position: relative; background: #000;
  }
  .slide-container img {
    max-width: 100%; max-height: 100%; object-fit: contain;
    will-change: opacity; transition: opacity 0.08s ease;
  }
  .hint {
    position: absolute; bottom: 0.5rem; left: 50%; transform: translateX(-50%);
    font-size: 0.7rem; color: #4a4f78; pointer-events: none; opacity: 0.7; z-index: 5;
  }
  .speed-label { font-size: 0.7rem; color: #6b70a0; }
  a.back-link {
    position: absolute; top: 0.5rem; left: 0.8rem; z-index: 20;
    color: #7c5cfc; text-decoration: none; font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace; opacity: 0.7;
  }
  a.back-link:hover { opacity: 1; }
'''

SLIDESHOW_SHARED_JS = r'''
// === SCRUBBER ===
const scrubber = document.getElementById('scrubber');
const scrubberTrack = document.getElementById('scrubber-track');
const scrubberFill = document.getElementById('scrubber-fill');
const scrubberThumb = document.getElementById('scrubber-thumb');
const frameNum = document.getElementById('frame-num');
const frameTotal = document.getElementById('frame-total');
const btnPlay = document.getElementById('btn-play');
const speedDisplay = document.getElementById('speed-display');
const container = document.getElementById('slide-container');

function updateScrubber() {
  const t = getTotal();
  const pct = t > 1 ? idx / (t - 1) : 0;
  scrubberFill.style.width = (pct * 100) + '%';
  const trackRect = scrubberTrack.getBoundingClientRect();
  const left = trackRect.left - scrubber.getBoundingClientRect().left + pct * trackRect.width;
  scrubberThumb.style.left = left + 'px';
}

let dragging = false;
function scrubFromEvent(e) {
  const trackRect = scrubberTrack.getBoundingClientRect();
  let pct = (e.clientX - trackRect.left) / trackRect.width;
  pct = Math.max(0, Math.min(1, pct));
  show(Math.round(pct * (getTotal() - 1)));
}
scrubber.addEventListener('pointerdown', (e) => {
  dragging = true; scrubber.classList.add('dragging');
  scrubber.setPointerCapture(e.pointerId); scrubFromEvent(e);
});
scrubber.addEventListener('pointermove', (e) => { if (dragging) scrubFromEvent(e); });
scrubber.addEventListener('pointerup', () => { dragging = false; scrubber.classList.remove('dragging'); });
scrubber.addEventListener('pointercancel', () => { dragging = false; scrubber.classList.remove('dragging'); });

function togglePlay() {
  playing = !playing;
  btnPlay.textContent = playing ? '⏸' : '▶';
  btnPlay.classList.toggle('playing', playing);
  if (playing) {
    playInterval = setInterval(() => {
      if (idx >= getTotal() - 1) { togglePlay(); return; }
      show(idx + 1);
    }, speed);
  } else { clearInterval(playInterval); playInterval = null; }
}
function updateSpeed(s) {
  speed = Math.max(30, Math.min(10000, s));
  speedDisplay.textContent = speed + 'ms';
  if (playing) { clearInterval(playInterval); playInterval = setInterval(() => {
    if (idx >= getTotal() - 1) { togglePlay(); return; } show(idx + 1);
  }, speed); }
}
function next() { show(Math.min(idx + 1, getTotal() - 1)); }
function prev() { show(Math.max(idx - 1, 0)); }
function goStart() { show(0); }
function goEnd() { show(getTotal() - 1); }

document.getElementById('btn-next').addEventListener('click', next);
document.getElementById('btn-prev').addEventListener('click', prev);
document.getElementById('btn-start').addEventListener('click', goStart);
document.getElementById('btn-end').addEventListener('click', goEnd);
document.getElementById('btn-play').addEventListener('click', togglePlay);
document.getElementById('btn-slower').addEventListener('click', () => updateSpeed(speed + 100));
document.getElementById('btn-faster').addEventListener('click', () => updateSpeed(speed - 100));

document.addEventListener('keydown', (e) => {
  if (e.key >= '0' && e.key <= '9') {
    e.preventDefault();
    const digit = parseInt(e.key, 10);
    const t = getTotal();
    if (t === 0) return;
    if (digit === 0) { show(0); }
    else if (digit === 9) { show(t - 1); }
    else { show(Math.round((digit / 10) * (t - 1))); }
    return;
  }
  switch (e.key) {
    case 'ArrowLeft': case 'ArrowDown': e.preventDefault(); prev(); break;
    case 'ArrowRight': case 'ArrowUp': e.preventDefault(); next(); break;
    case 'Home': e.preventDefault(); goStart(); break;
    case 'End': e.preventDefault(); goEnd(); break;
    case ' ': e.preventDefault(); togglePlay(); break;
    case '+': e.preventDefault(); updateSpeed(speed - 100); break;
    case '-': e.preventDefault(); updateSpeed(speed + 100); break;
    case 'PageDown': e.preventDefault(); show(Math.min(idx + 50, getTotal() - 1)); break;
    case 'PageUp': e.preventDefault(); show(Math.max(idx - 50, 0)); break;
  }
});

container.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (e.deltaY > 0 || e.deltaX > 0) next(); else prev();
}, { passive: false });

let touchStartX = 0;
container.addEventListener('touchstart', (e) => { touchStartX = e.touches[0].clientX; }, { passive: true });
container.addEventListener('touchend', (e) => {
  const dx = e.changedTouches[0].clientX - touchStartX;
  if (Math.abs(dx) > 40) { if (dx < 0) next(); else prev(); }
}, { passive: true });

window.addEventListener('resize', () => updateScrubber());
'''

# ── Shared toolbar + scrubber HTML fragment ─────────────────────────────
SLIDESHOW_TOOLBAR_HTML = r'''
<a class="back-link" href="index.html">← Dashboard</a>
<div class="toolbar">
  <div class="title">{{ title }}</div>
  <div class="controls">
    <button id="btn-start" title="First (Home)">⏮</button>
    <button id="btn-prev" title="Previous">◀</button>
    <button id="btn-play" title="Play/Pause (Space)">▶</button>
    <button id="btn-next" title="Next">▶▶</button>
    <button id="btn-end" title="Last (End)">⏭</button>
    <span class="speed-label">Speed:</span>
    <button id="btn-slower">−</button>
    <span id="speed-display" class="counter" style="min-width:50px;">{{ default_speed }}ms</span>
    <button id="btn-faster">+</button>
    <div class="counter"><span class="current" id="frame-num">1</span> / <span id="frame-total">{{ frame_count }}</span></div>
  </div>
</div>
<div class="scrubber-container" id="scrubber">
  <div class="scrubber-track" id="scrubber-track">
    <div class="scrubber-fill" id="scrubber-fill" style="width:0%"></div>
  </div>
  <div class="scrubber-thumb" id="scrubber-thumb" style="left:12px"></div>
</div>
'''


# ── index.html template ────────────────────────────────────────────────
DASHBOARD_TEMPLATE = JINJA_ENV.from_string(r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Grokking Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
  :root {
    --bg:#08090d;--bg2:#0d0f16;--card:#12152a;--card2:#181c35;
    --border:#1e2340;--border-light:#2a2f55;
    --accent:#7c5cfc;--accent-dim:rgba(124,92,252,0.12);
    --accent2:#00d4aa;--accent2-dim:rgba(0,212,170,0.12);
    --accent3:#f472b6;
    --text:#e8eaf6;--muted:#6b70a0;--muted2:#4a4f78;
    --glow:rgba(124,92,252,0.08);
    --green:#00d4aa;--red:#ff5c72;--yellow:#f0c040;
    --radius:16px;--radius-sm:10px;
  }
  *{margin:0;padding:0;box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;-webkit-font-smoothing:antialiased}
  .hero{position:relative;background:linear-gradient(160deg,#130f30 0%,#08090d 40%,#061215 100%);border-bottom:1px solid var(--border);padding:3rem 2rem 2rem;text-align:center;overflow:hidden}
  .hero::before{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:radial-gradient(ellipse at 30% 20%,rgba(124,92,252,0.06) 0%,transparent 50%),radial-gradient(ellipse at 70% 80%,rgba(0,212,170,0.04) 0%,transparent 50%);pointer-events:none}
  .hero h1{font-size:2.8rem;font-weight:900;letter-spacing:-0.03em;background:linear-gradient(135deg,#7c5cfc 0%,#00d4aa 50%,#f472b6 100%);background-size:200% 200%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:gs 8s ease infinite;margin-bottom:0.4rem;position:relative}
  @keyframes gs{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
  .hero .subtitle{color:var(--muted);font-size:0.95rem;position:relative}
  .hero .meta-row{display:flex;justify-content:center;gap:1rem;flex-wrap:wrap;margin-top:1rem;position:relative}
  .meta-pill{display:inline-flex;align-items:center;gap:0.4rem;padding:0.35rem 1rem;background:var(--card);border:1px solid var(--border);border-radius:999px;font-size:0.78rem;color:var(--muted)}
  .meta-pill .val{color:var(--accent2);font-weight:600}
  .meta-pill .val-warm{color:var(--accent3);font-weight:600}
  .meta-pill .val-purple{color:var(--accent);font-weight:600}
  .stats-ribbon{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1px;background:var(--border);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;margin:2rem 0}
  .stat-card{background:var(--card);padding:1.2rem 1.5rem;text-align:center}
  .stat-card .stat-value{font-size:1.8rem;font-weight:800;font-family:'JetBrains Mono',monospace}
  .stat-card .stat-label{font-size:0.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-top:0.2rem}
  .stat-green{color:var(--green)}.stat-purple{color:var(--accent)}.stat-pink{color:var(--accent3)}.stat-yellow{color:var(--yellow)}
  .container{max-width:1440px;margin:0 auto;padding:0 1.5rem 3rem}
  .section-title{font-size:1.2rem;font-weight:700;margin:2.5rem 0 1rem;padding-bottom:0.6rem;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:0.6rem}
  .badge{background:var(--accent-dim);color:var(--accent);font-size:0.7rem;padding:0.2rem 0.6rem;border-radius:999px;font-weight:600;border:1px solid rgba(124,92,252,0.2)}
  .badge-green{background:var(--accent2-dim);color:var(--accent2);border-color:rgba(0,212,170,0.2)}
  .image-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:1.2rem}
  .image-card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;transition:transform 0.25s,box-shadow 0.25s}
  .image-card:hover{transform:translateY(-4px);box-shadow:0 12px 40px rgba(124,92,252,0.12)}
  .image-card img{width:100%;display:block;background:#000}
  .image-card .label{padding:0.65rem 1rem;font-size:0.78rem;color:var(--muted);font-family:'JetBrains Mono',monospace;border-top:1px solid var(--border)}
  .table-scroll{overflow-x:auto;border-radius:var(--radius);border:1px solid var(--border)}
  .epoch-overview{width:100%;border-collapse:collapse}
  .epoch-overview th{background:var(--card2);text-align:left;padding:0.7rem 1rem;font-size:0.75rem;color:var(--accent);border-bottom:2px solid var(--border);position:sticky;top:0;font-weight:600;text-transform:uppercase;letter-spacing:0.06em}
  .epoch-overview td{padding:0.5rem 1rem;font-size:0.82rem;border-bottom:1px solid var(--border);font-family:'JetBrains Mono',monospace;background:var(--card)}
  .epoch-overview tr:hover td{background:var(--card2)}
  .acc-good{color:var(--green);font-weight:700}.acc-mid{color:var(--yellow);font-weight:700}.acc-bad{color:var(--red);font-weight:700}
  .acc-bar-bg{width:130px;height:8px;background:rgba(255,255,255,0.05);border-radius:4px;overflow:hidden;display:inline-block;vertical-align:middle}
  .acc-bar-fg{height:100%;border-radius:4px;transition:width 0.3s}
  details{margin-bottom:0.4rem}
  details summary{cursor:pointer;padding:0.55rem 1rem;background:var(--card);border:1px solid var(--border);border-radius:var(--radius-sm);font-size:0.82rem;font-family:'JetBrains Mono',monospace;display:flex;align-items:center;gap:0.8rem;list-style:none}
  details summary::-webkit-details-marker{display:none}
  details summary::before{content:'▸';color:var(--muted);transition:transform 0.2s;font-size:0.9rem;flex-shrink:0}
  details[open] summary::before{transform:rotate(90deg)}
  details summary:hover{background:var(--card2)}
  details[open] summary{border-radius:var(--radius-sm) var(--radius-sm) 0 0;border-bottom-color:transparent}
  details .detail-body{background:var(--card);border:1px solid var(--border);border-top:none;border-radius:0 0 var(--radius-sm) var(--radius-sm);overflow-x:auto}
  .sample-table{width:100%;border-collapse:collapse}
  .sample-table th{background:var(--card2);text-align:left;padding:0.4rem 0.8rem;font-size:0.7rem;color:var(--muted);border-bottom:1px solid var(--border);text-transform:uppercase}
  .sample-table td{padding:0.35rem 0.8rem;font-size:0.78rem;border-bottom:1px solid rgba(30,35,64,0.6);font-family:'JetBrains Mono',monospace}
  .row-correct td{color:var(--green)}.row-wrong td{color:var(--red)}
  .file-link-grid{display:flex;flex-wrap:wrap;gap:0.5rem}
  .file-link{display:inline-flex;align-items:center;gap:0.3rem;padding:0.3rem 0.75rem;background:var(--card);border:1px solid var(--border);border-radius:999px;font-size:0.75rem;font-family:'JetBrains Mono',monospace;color:var(--accent);text-decoration:none;transition:background 0.15s}
  .file-link:hover{background:var(--card2);border-color:var(--accent)}
  .file-link-green{color:var(--accent2)}.file-link-green:hover{border-color:var(--accent2)}
  .file-link-pink{color:var(--accent3)}.file-link-pink:hover{border-color:var(--accent3)}
  .file-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(350px,1fr));gap:1rem}
  .file-card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden}
  .file-card .file-header{padding:0.65rem 1rem;font-size:0.82rem;font-weight:600;background:linear-gradient(90deg,var(--accent-dim),transparent);border-bottom:1px solid var(--border);font-family:'JetBrains Mono',monospace;display:flex;align-items:center;gap:0.4rem}
  .dot{width:8px;height:8px;border-radius:50%;display:inline-block}
  .dot-csv{background:var(--accent2)}.dot-txt{background:var(--accent)}.dot-py{background:var(--yellow)}
  .file-card pre{padding:0.8rem 1rem;font-size:0.72rem;line-height:1.55;overflow-x:auto;max-height:300px;overflow-y:auto;color:#aab;background:var(--bg2);font-family:'JetBrains Mono',monospace}
  .file-card a.file-open{display:block;padding:0.6rem 1rem;text-align:center;color:var(--accent);text-decoration:none;font-size:0.8rem;font-weight:600;border-top:1px solid var(--border)}
  .file-card a.file-open:hover{background:var(--accent-dim)}
  .empty-state{color:var(--muted);font-style:italic;padding:1rem 0;font-size:0.9rem}
  .footer{text-align:center;padding:2rem;color:var(--muted2);font-size:0.75rem;border-top:1px solid var(--border);margin-top:2rem}
  ::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
  @media(max-width:600px){.image-grid,.file-grid{grid-template-columns:1fr}.hero h1{font-size:1.8rem}.stats-ribbon{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<div class="hero">
  <h1>Grokking Dashboard</h1>
  <div class="subtitle">Live training monitor</div>
  <div class="meta-row">
    {% if started_at %}<div class="meta-pill">🏁 Started <span class="val-warm">{{ started_at }}</span></div>{% endif %}
    {% if elapsed %}<div class="meta-pill">⏱ Elapsed <span class="val-purple">{{ elapsed }}</span></div>{% endif %}
    <div class="meta-pill">📸 Snapshot <span class="val">{{ timestamp }}</span></div>
  </div>
</div>
<div class="container">

<div class="stats-ribbon">
  <div class="stat-card"><div class="stat-value stat-purple">{{ epoch_count }}</div><div class="stat-label">Epochs</div></div>
  <div class="stat-card"><div class="stat-value stat-green">{{ latest_acc }}</div><div class="stat-label">Latest Accuracy</div></div>
  <div class="stat-card"><div class="stat-value stat-pink">{{ n_plots }}</div><div class="stat-label">Plots</div></div>
  <div class="stat-card"><div class="stat-value stat-yellow">{{ total_files }}</div><div class="stat-label">Total Files</div></div>
</div>

<!-- Slideshow links -->
{% if has_slideshow or has_jacobi or has_jacobi_4d %}
<div class="section-title">🎬 Slideshows</div>
<div class="file-link-grid">
  {% if has_slideshow %}<a class="file-link" href="slideshow.html" target="_blank">📈 Training Plot History ({{ n_training_hist }} frames)</a>{% endif %}
  {% if has_jacobi %}<a class="file-link file-link-green" href="jacobi.html" target="_blank">🌊 Jacobi Field History ({{ n_jacobi_steps }} steps)</a>{% endif %}
  {% if has_jacobi_4d %}<a class="file-link file-link-pink" href="4d_jacobi.html" target="_blank">🔮 4D Jacobi Slices ({{ n_jacobi_4d_steps }} steps, {{ n_jacobi_4d_slices }} slices)</a>{% endif %}
</div>
{% endif %}

<!-- Plots -->
<div class="section-title">📊 Plots &amp; Visualizations <span class="badge">{{ n_plots }}</span></div>
{% if images %}
<div class="image-grid">
  {% for img in images %}
  <div class="image-card">
    <a href="{{ img }}?t={{ cache_bust }}" target="_blank"><img src="{{ img }}?t={{ cache_bust }}" alt="{{ img }}" loading="lazy"></a>
    <div class="label">{{ img }}</div>
  </div>
  {% endfor %}
</div>
{% else %}
<div class="empty-state">No images found yet.</div>
{% endif %}

<!-- Epoch accuracy -->
<div class="section-title">🧪 Epoch Accuracy <span class="badge">{{ epoch_count }} epochs</span></div>
{% if overview %}
<div class="table-scroll"><table class="epoch-overview">
<thead><tr><th>Epoch</th><th>Correct</th><th>Total</th><th>Accuracy</th><th style="min-width:150px">Progress</th></tr></thead>
<tbody>
{% for e in overview %}
{% set cls = 'acc-good' if e.pct >= 80 else ('acc-mid' if e.pct >= 50 else 'acc-bad') %}
{% set col = 'var(--green)' if e.pct >= 80 else ('#f0c040' if e.pct >= 50 else 'var(--red)') %}
<tr><td>{{ e.epoch }}</td><td>{{ e.correct }}</td><td>{{ e.total }}</td><td class="{{ cls }}">{{ e.pct }}%</td>
<td><div class="acc-bar-bg"><div class="acc-bar-fg" style="width:{{ e.pct }}%;background:{{ col }}"></div></div></td></tr>
{% endfor %}
</tbody>
</table></div>

<!-- Epoch details (collapsible) -->
{% if details %}
<div class="section-title">🔬 Epoch Details <span class="badge-green badge">expandable</span></div>
{% for e in overview %}
{% if e.epoch in details %}
<details>
<summary>
  <span>Epoch {{ e.epoch }}</span>
  {% set cls = 'acc-good' if e.pct >= 80 else ('acc-mid' if e.pct >= 50 else 'acc-bad') %}
  <span class="{{ cls }}">{{ e.pct }}% ({{ e.correct }}/{{ e.total }})</span>
</summary>
<div class="detail-body">
<table class="sample-table">
<thead><tr><th>#</th><th>Params</th><th>Expected</th><th>Predicted</th><th>✓</th></tr></thead>
<tbody>
{% for s in details[e.epoch] %}
<tr class="{{ 'row-correct' if s.correct else 'row-wrong' }}">
<td>{{ loop.index }}</td><td>{{ s.params }}</td><td>{{ s.expected }}</td><td>{{ s.predicted }}</td><td>{{ '✓' if s.correct else '✗' }}</td>
</tr>
{% endfor %}
</tbody>
</table>
</div>
</details>
{% endif %}
{% endfor %}
{% endif %}
{% else %}
<div class="empty-state">No epoch data found yet.</div>
{% endif %}

<!-- File previews -->
{% if other_csvs or other_txts or py_files %}
<div class="section-title">📁 Files <span class="badge">{{ (other_csvs|length) + (other_txts|length) + (py_files|length) }}</span></div>

{% if other_csvs %}
<div class="file-grid">
  {% for csv in other_csvs %}
  <div class="file-card">
    <div class="file-header"><span class="dot dot-csv"></span>{{ csv | basename }}</div>
    <pre>{{ previews.get(csv, '') }}</pre>
    <a class="file-open" href="{{ csv }}" target="_blank">Open full file →</a>
  </div>
  {% endfor %}
</div>
{% endif %}

{% if other_txts %}
<div class="file-grid" style="margin-top:1rem">
  {% for txt in other_txts %}
  <div class="file-card">
    <div class="file-header"><span class="dot dot-txt"></span>{{ txt | basename }}</div>
    <pre>{{ previews.get(txt, '') }}</pre>
    <a class="file-open" href="{{ txt }}" target="_blank">Open full file →</a>
  </div>
  {% endfor %}
</div>
{% endif %}

{% if py_files %}
<div class="file-grid" style="margin-top:1rem">
  {% for py in py_files %}
  <div class="file-card">
    <div class="file-header"><span class="dot dot-py"></span>{{ py | basename }}</div>
    <pre>{{ previews.get(py, '') }}</pre>
    <a class="file-open" href="{{ py }}" target="_blank">Open full file →</a>
  </div>
  {% endfor %}
</div>
{% endif %}

{% endif %}

</div>
<div class="footer">Generated by generate_dashboard.py · {{ timestamp }}</div>
</body>
</html>''')


# ── slideshow.html template (old UI: scrubber + preload + keyboard) ────
SLIDESHOW_TEMPLATE = JINJA_ENV.from_string(r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Training Plot Slideshow</title>
<style>
''' + SLIDESHOW_SHARED_CSS + r'''
</style>
</head>
<body>
''' + SLIDESHOW_TOOLBAR_HTML + r'''
<div class="slide-container" id="slide-container">
  <img id="slide-img-a" src="" alt="frame" style="position:absolute;max-width:100%;max-height:100%;object-fit:contain;">
  <img id="slide-img-b" src="" alt="frame" style="position:absolute;max-width:100%;max-height:100%;object-fit:contain;opacity:0;">
  <div class="hint">← → navigate · 0-9 jump · Scroll wheel · Space play/pause · Home/End · +/− speed</div>
</div>
<script>
const images = {{ frames_json }};
let idx = 0, playing = false, playInterval = null, speed = {{ default_speed }};
let activeSlide = 'a';

function getTotal() { return images.length; }
document.getElementById('frame-total').textContent = getTotal();

// === PRELOAD CACHE ===
const cache = new Map();
const PRELOAD_AHEAD = 15, PRELOAD_BEHIND = 5, MAX_CACHE = 60;
function preloadAround(center) {
  const t = getTotal();
  const lo = Math.max(0, center - PRELOAD_BEHIND);
  const hi = Math.min(t - 1, center + PRELOAD_AHEAD);
  for (let i = lo; i <= hi; i++) {
    if (!cache.has(images[i])) { const img = new Image(); img.src = images[i]; cache.set(images[i], img); }
  }
  if (cache.size > MAX_CACHE) {
    for (const [url] of cache) { if (cache.size <= MAX_CACHE) break; cache.delete(url); }
  }
}

function show(i) {
  const t = getTotal();
  if (t === 0) return;
  idx = Math.max(0, Math.min(t - 1, i));
  document.getElementById('frame-num').textContent = idx + 1;

  const src = images[idx];
  const front = document.getElementById('slide-img-' + activeSlide);
  const backId = activeSlide === 'a' ? 'b' : 'a';
  const back = document.getElementById('slide-img-' + backId);

  const cached = cache.get(src);
  if (cached && cached.complete && cached.naturalWidth > 0) {
    back.src = src;
    back.style.opacity = '1';
    front.style.opacity = '0';
    activeSlide = backId;
  } else {
    const loader = new Image();
    const targetIdx = idx;
    loader.onload = () => {
      if (idx !== targetIdx) return;
      back.src = src;
      if (back.decode) {
        back.decode().then(() => {
          if (idx !== targetIdx) return;
          back.style.opacity = '1'; front.style.opacity = '0'; activeSlide = backId;
        }).catch(() => { back.style.opacity = '1'; front.style.opacity = '0'; activeSlide = backId; });
      } else { back.style.opacity = '1'; front.style.opacity = '0'; activeSlide = backId; }
    };
    loader.src = src;
    cache.set(src, loader);
  }
  updateScrubber();
  preloadAround(idx);
}

''' + SLIDESHOW_SHARED_JS + r'''
preloadAround(0);
show(0);
</script>
</body>
</html>''')


# ── jacobi.html template (old UI: scrubber + preload + keyboard) ───────
JACOBI_TEMPLATE = JINJA_ENV.from_string(r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Jacobi Field Slideshow</title>
<style>
''' + SLIDESHOW_SHARED_CSS + r'''
  .slide-container {
    flex-wrap: wrap; align-content: center; gap: 6px; padding: 8px; overflow: auto;
  }
  .slide-container .layer-img {
    max-height: 48%; max-width: 48%; object-fit: contain;
    border: 1px solid #1e2340; border-radius: 4px; flex-shrink: 1;
  }
  .slide-container.single-layer .layer-img { max-height: 95%; max-width: 95%; }
  .step-label {
    position: absolute; top: 0.5rem; left: 50%; transform: translateX(-50%);
    font-size: 0.9rem; font-weight: 700; color: #7c5cfc;
    background: rgba(8,9,13,0.85); padding: 0.2rem 1rem;
    border-radius: 6px; border: 1px solid #2a2f55; z-index: 5;
    font-family: 'JetBrains Mono', monospace;
  }
</style>
</head>
<body>
''' + SLIDESHOW_TOOLBAR_HTML + r'''
<div class="slide-container" id="slide-container">
  <div class="step-label" id="step-label">Step ?</div>
  <div class="hint">← → navigate · 0-9 jump · Scroll wheel · Space play/pause · Home/End · +/− speed</div>
</div>
<script>
const data = {{ data_json }};
const steps = {{ steps_json }};
let idx = 0, playing = false, playInterval = null, speed = {{ default_speed }};

function getTotal() { return steps.length; }
document.getElementById('frame-total').textContent = getTotal();

// === PRELOAD CACHE ===
const cache = new Map();
const PRELOAD_AHEAD = 5, PRELOAD_BEHIND = 2, MAX_CACHE = 120;
function preloadAround(center) {
  const t = getTotal();
  const lo = Math.max(0, center - PRELOAD_BEHIND);
  const hi = Math.min(t - 1, center + PRELOAD_AHEAD);
  for (let i = lo; i <= hi; i++) {
    const layers = data[steps[i]];
    if (!layers) continue;
    layers.forEach(([layer, path]) => {
      if (!cache.has(path)) { const img = new Image(); img.src = path; cache.set(path, img); }
    });
  }
  if (cache.size > MAX_CACHE) {
    for (const [url] of cache) { if (cache.size <= MAX_CACHE) break; cache.delete(url); }
  }
}

function show(i) {
  const t = getTotal();
  if (t === 0) return;
  idx = Math.max(0, Math.min(t - 1, i));
  document.getElementById('frame-num').textContent = idx + 1;

  const step = steps[idx];
  const layers = data[step];
  const container = document.getElementById('slide-container');
  const label = document.getElementById('step-label');
  if (label) label.textContent = 'Step ' + step + ' — ' + layers.length + ' layer' + (layers.length !== 1 ? 's' : '');

  container.querySelectorAll('.layer-img').forEach(el => el.remove());
  container.classList.toggle('single-layer', layers.length === 1);
  layers.forEach(([layer, path]) => {
    const img = document.createElement('img');
    img.className = 'layer-img'; img.src = path;
    img.alt = 'Layer ' + layer; img.title = 'Step ' + step + ', Layer ' + layer;
    container.appendChild(img);
  });

  updateScrubber();
  preloadAround(idx);
}

''' + SLIDESHOW_SHARED_JS + r'''
preloadAround(0);
show(0);
</script>
</body>
</html>''')


# ── 4d_jacobi.html template (old UI: scrubber + preload + keyboard) ────
JACOBI_4D_TEMPLATE = JINJA_ENV.from_string(r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>4D Jacobi Slices Slideshow</title>
<style>
''' + SLIDESHOW_SHARED_CSS + r'''
  .slide-container {
    flex-wrap: wrap; align-content: flex-start; gap: 8px; padding: 12px; overflow: auto;
  }
  .layer-section { width: 100%; }
  .layer-section h3 {
    font-size: 0.85rem; font-weight: 600; color: #f472b6;
    margin: 0.5rem 0; font-family: 'JetBrains Mono', monospace;
  }
  .slice-grid { display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; }
  .slice-card { border: 1px solid #1e2340; border-radius: 8px; overflow: hidden; background: #000; }
  .slice-card img { max-width: 280px; max-height: 280px; display: block; }
  .slice-card .slice-label {
    padding: 0.3rem 0.6rem; font-size: 0.7rem; color: #6b70a0;
    font-family: 'JetBrains Mono', monospace; background: #12152a;
    border-top: 1px solid #1e2340; text-align: center;
  }
  .step-label {
    position: absolute; top: 0.5rem; left: 50%; transform: translateX(-50%);
    font-size: 0.9rem; font-weight: 700; color: #f472b6;
    background: rgba(8,9,13,0.85); padding: 0.2rem 1rem;
    border-radius: 6px; border: 1px solid #2a2f55; z-index: 5;
    font-family: 'JetBrains Mono', monospace;
  }
</style>
</head>
<body>
''' + SLIDESHOW_TOOLBAR_HTML + r'''
<div class="slide-container" id="slide-container">
  <div class="step-label" id="step-label">Step ?</div>
  <div class="hint">← → navigate · 0-9 jump · Scroll wheel · Space play/pause · Home/End · +/− speed</div>
</div>
<script>
const data = {{ data_json }};
const steps = {{ steps_json }};
let idx = 0, playing = false, playInterval = null, speed = {{ default_speed }};

function getTotal() { return steps.length; }
document.getElementById('frame-total').textContent = getTotal();

// === PRELOAD CACHE ===
const cache = new Map();
const PRELOAD_AHEAD = 3, PRELOAD_BEHIND = 1, MAX_CACHE = 200;
function preloadAround(center) {
  const t = getTotal();
  const lo = Math.max(0, center - PRELOAD_BEHIND);
  const hi = Math.min(t - 1, center + PRELOAD_AHEAD);
  for (let i = lo; i <= hi; i++) {
    const layers = data[steps[i]];
    if (!layers) continue;
    const layerNums = Object.keys(layers).map(Number).sort((a,b)=>a-b);
    for (const ln of layerNums) {
      layers[ln].forEach(s => {
        if (!cache.has(s.path)) { const img = new Image(); img.src = s.path; cache.set(s.path, img); }
      });
    }
  }
  if (cache.size > MAX_CACHE) {
    for (const [url] of cache) { if (cache.size <= MAX_CACHE) break; cache.delete(url); }
  }
}

function show(i) {
  const t = getTotal();
  if (t === 0) return;
  idx = Math.max(0, Math.min(t - 1, i));
  document.getElementById('frame-num').textContent = idx + 1;

  const step = steps[idx];
  const layers = data[step];
  const container = document.getElementById('slide-container');
  const label = document.getElementById('step-label');
  if (label) label.textContent = 'Step ' + step;

  // Remove old content (keep step-label and hint)
  container.querySelectorAll('.layer-section').forEach(el => el.remove());

  const layerNums = Object.keys(layers).map(Number).sort((a,b)=>a-b);
  for (const ln of layerNums) {
    const section = document.createElement('div');
    section.className = 'layer-section';
    section.innerHTML = `<h3>Layer ${ln}</h3>`;
    const grid = document.createElement('div');
    grid.className = 'slice-grid';
    layers[ln].forEach(s => {
      const card = document.createElement('div');
      card.className = 'slice-card';
      card.innerHTML = `<img src="${s.path}"><div class="slice-label">d${s.dim_a} × d${s.dim_b}</div>`;
      grid.appendChild(card);
    });
    section.appendChild(grid);
    container.appendChild(section);
  }

  updateScrubber();
  preloadAround(idx);
}

''' + SLIDESHOW_SHARED_JS + r'''
preloadAround(0);
show(0);
</script>
</body>
</html>''')


# ═════════════════════════════════════════════════════════════════════════
# 7. MAIN — generate all HTML files
# ═════════════════════════════════════════════════════════════════════════

import json

MAX_SLIDESHOW_FRAMES = 300
MAX_JACOBI_STEPS = 200


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_dashboard.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory")
        sys.exit(1)

    t0 = time.time()
    print(f"[dashboard] Scanning {run_dir} ...")
    rf = scan_run_dir(run_dir)

    # ── Parse epoch data ────────────────────────────────────────────
    overview, details = parse_epoch_files(run_dir, rf.epoch_txts)

    # ── Compute stats ───────────────────────────────────────────────
    latest_acc = "—"
    if overview:
        latest_acc = f"{overview[-1]['pct']}%"

    elapsed = compute_elapsed(rf.started_at)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cache_bust = int(time.time())

    total_files = (len(rf.summary_images) + len(rf.training_hist) +
                   len(rf.jacobi_images) + len(rf.jacobi_4d_images) +
                   len(rf.epoch_txts) + len(rf.epoch_csvs) +
                   len(rf.other_csvs) + len(rf.other_txts) + len(rf.py_files))

    # ── Prepare images list (include current training_plot) ─────────
    images = list(rf.summary_images)
    if rf.has_current_plot and "training_plot.png" not in images:
        images.insert(0, "training_plot.png")

    # ── Build file previews ─────────────────────────────────────────
    previews = build_previews(run_dir, rf.other_csvs + rf.other_txts + rf.py_files)

    # ── Generate index.html ─────────────────────────────────────────
    jacobi_4d_steps = sorted(set(s for s, _, _, _, _ in rf.jacobi_4d_images))
    jacobi_4d_slices = sorted(set((da, db) for _, _, da, db, _ in rf.jacobi_4d_images))

    index_html = DASHBOARD_TEMPLATE.render(
        images=images,
        overview=overview,
        details=details,
        started_at=rf.started_at,
        elapsed=elapsed,
        timestamp=timestamp,
        cache_bust=cache_bust,
        epoch_count=len(overview),
        latest_acc=latest_acc,
        n_plots=len(images),
        total_files=total_files,
        has_slideshow=len(rf.training_hist) > 1,
        has_jacobi=len(rf.jacobi_images) > 0,
        has_jacobi_4d=len(rf.jacobi_4d_images) > 0,
        n_training_hist=len(rf.training_hist),
        n_jacobi_steps=len(set(s for s, _, _ in rf.jacobi_images)),
        n_jacobi_4d_steps=len(jacobi_4d_steps),
        n_jacobi_4d_slices=len(jacobi_4d_slices),
        other_csvs=rf.other_csvs,
        other_txts=rf.other_txts,
        py_files=rf.py_files,
        previews=previews,
    )
    (run_dir / "index.html").write_text(index_html)
    print(f"  ✓ index.html ({len(images)} plots, {len(overview)} epochs)")

    # ── Generate slideshow.html ─────────────────────────────────────
    if rf.training_hist:
        all_frames = list(rf.training_hist)
        if rf.has_current_plot:
            all_frames.append('training_plot.png')
        frames = subsample(all_frames, MAX_SLIDESHOW_FRAMES)
        if len(frames) < len(all_frames):
            print(f"  Training slideshow: subsampled {len(all_frames)} → {len(frames)} frames")

        slideshow_html = SLIDESHOW_TEMPLATE.render(
            title="Training Plot History",
            default_speed=500,
            frame_count=len(frames),
            frames_json=json.dumps(frames),
        )
        (run_dir / "slideshow.html").write_text(slideshow_html)
        print(f"  ✓ slideshow.html ({len(frames)} frames)")

    # ── Generate jacobi.html (non-4D) ──────────────────────────────
    if rf.jacobi_images:
        # Group by step → [(layer, path), ...]
        step_map = defaultdict(list)
        for step, layer, path in rf.jacobi_images:
            step_map[step].append((layer, path))
        for step in step_map:
            step_map[step].sort()

        all_steps = sorted(step_map.keys())
        sampled_steps = subsample(all_steps, MAX_JACOBI_STEPS)
        if len(sampled_steps) < len(all_steps):
            print(f"  Jacobi slideshow: subsampled {len(all_steps)} → {len(sampled_steps)} steps")

        # Build data: {step: [(layer, path), ...]}
        data = {s: step_map[s] for s in sampled_steps}

        jacobi_html = JACOBI_TEMPLATE.render(
            title="Jacobi Field History",
            default_speed=300,
            frame_count=len(sampled_steps),
            steps_json=json.dumps(sampled_steps),
            data_json=json.dumps(data),
        )
        (run_dir / "jacobi.html").write_text(jacobi_html)
        print(f"  ✓ jacobi.html ({len(sampled_steps)} steps)")

    # ── Generate 4d_jacobi.html ─────────────────────────────────────
    if rf.jacobi_4d_images:
        # Group by step → layer → [{dim_a, dim_b, path}, ...]
        step_layer_map = defaultdict(lambda: defaultdict(list))
        for step, layer, dim_a, dim_b, path in rf.jacobi_4d_images:
            step_layer_map[step][layer].append({
                'dim_a': dim_a,
                'dim_b': dim_b,
                'path': path,
            })

        # Sort slices within each layer
        for step in step_layer_map:
            for layer in step_layer_map[step]:
                step_layer_map[step][layer].sort(key=lambda x: (x['dim_a'], x['dim_b']))

        all_steps_4d = sorted(step_layer_map.keys())
        sampled_steps_4d = subsample(all_steps_4d, MAX_JACOBI_STEPS)
        if len(sampled_steps_4d) < len(all_steps_4d):
            print(f"  4D Jacobi slideshow: subsampled {len(all_steps_4d)} → {len(sampled_steps_4d)} steps")

        # Build data: {step: {layer: [{dim_a, dim_b, path}, ...]}}
        data_4d = {}
        for s in sampled_steps_4d:
            data_4d[s] = {}
            for layer in sorted(step_layer_map[s].keys()):
                data_4d[s][layer] = step_layer_map[s][layer]

        jacobi_4d_html = JACOBI_4D_TEMPLATE.render(
            title="4D Jacobi Slices",
            default_speed=800,
            frame_count=len(sampled_steps_4d),
            steps_json=json.dumps(sampled_steps_4d),
            data_json=json.dumps(data_4d),
        )
        (run_dir / "4d_jacobi.html").write_text(jacobi_4d_html)
        print(f"  ✓ 4d_jacobi.html ({len(sampled_steps_4d)} steps, {len(jacobi_4d_slices)} slice pairs)")

    elapsed_gen = time.time() - t0
    print(f"[dashboard] Done in {elapsed_gen:.2f}s")


if __name__ == "__main__":
    main()
