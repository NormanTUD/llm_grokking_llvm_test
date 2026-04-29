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
# 5. HTML TEMPLATES (Jinja2, embedded)
# ═════════════════════════════════════════════════════════════════════════

JINJA_ENV = Environment(loader=BaseLoader(), autoescape=False)

# Register custom filters BEFORE compiling any templates
JINJA_ENV.filters['basename'] = lambda s: Path(s).name

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

<!-- Other files -->
{% if other_csvs or other_txts or py_files %}
<div class="section-title">📁 Files <span class="badge">{{ (other_csvs|length) + (other_txts|length) + (py_files|length) }}</span></div>
<div class="file-link-grid">
  {% for f in other_csvs %}<a class="file-link file-link-green" href="{{ f }}" target="_blank">📄 {{ f | basename }}</a>{% endfor %}
  {% for f in other_txts %}<a class="file-link" href="{{ f }}" target="_blank">📝 {{ f | basename }}</a>{% endfor %}
  {% for f in py_files %}<a class="file-link file-link-pink" href="{{ f }}" target="_blank">🐍 {{ f | basename }}</a>{% endfor %}
</div>
{% endif %}

</div>
<div class="footer">Generated by generate_dashboard.py · {{ timestamp }}</div>
</body>
</html>''')


# ── slideshow.html template ────────────────────────────────────────────
SLIDESHOW_TEMPLATE = JINJA_ENV.from_string(r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Training Plot Slideshow</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  :root{--bg:#08090d;--card:#12152a;--border:#1e2340;--accent:#7c5cfc;--text:#e8eaf6;--muted:#6b70a0}
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);display:flex;flex-direction:column;align-items:center;min-height:100vh;padding:1rem}
  h1{font-size:1.5rem;margin:1rem 0 0.5rem;font-weight:700}
  .controls{display:flex;align-items:center;gap:1rem;margin:1rem 0;flex-wrap:wrap;justify-content:center}
  button{background:var(--card);border:1px solid var(--border);color:var(--text);padding:0.5rem 1.2rem;border-radius:8px;cursor:pointer;font-size:0.85rem;font-weight:600;transition:background 0.15s}
  button:hover{background:var(--accent);border-color:var(--accent)}
  button.active{background:var(--accent);border-color:var(--accent)}
  .speed-group{display:flex;gap:0.3rem}
  .frame-info{font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:var(--muted);min-width:120px;text-align:center}
  .slider-row{width:100%;max-width:900px;margin:0.5rem 0}
  input[type=range]{width:100%;accent-color:var(--accent)}
  .img-container{margin:1rem 0;border:1px solid var(--border);border-radius:12px;overflow:hidden;background:#000}
  .img-container img{max-width:90vw;max-height:70vh;display:block}
  a.back{color:var(--accent);text-decoration:none;font-size:0.85rem;margin-top:1rem}
</style>
</head>
<body>
<h1>📈 Training Plot History</h1>
<div class="controls">
  <button onclick="prev()">⏮ Prev</button>
  <button id="playBtn" onclick="togglePlay()">▶ Play</button>
  <button onclick="next()">Next ⏭</button>
  <div class="speed-group">
    <button onclick="setSpeed(2000)">🐢</button>
    <button onclick="setSpeed(800)" class="active" id="speedMed">🐇</button>
    <button onclick="setSpeed(200)">⚡</button>
  </div>
</div>
<div class="slider-row"><input type="range" id="slider" min="0" max="{{ frames|length - 1 }}" value="0" oninput="goTo(+this.value)"></div>
<div class="frame-info" id="info">Frame 1 / {{ frames|length }}</div>
<div class="img-container"><img id="mainImg" src="{{ frames[0] }}"></div>
<a class="back" href="index.html">← Back to Dashboard</a>
<script>
const frames = {{ frames_json }};
let idx = 0, playing = false, timer = null, speed = 800;
function show(i){idx=i;document.getElementById('mainImg').src=frames[i];document.getElementById('slider').value=i;document.getElementById('info').textContent=`Frame ${i+1} / ${frames.length}`;}
function next(){show((idx+1)%frames.length)}
function prev(){show((idx-1+frames.length)%frames.length)}
function togglePlay(){playing=!playing;document.getElementById('playBtn').textContent=playing?'⏸ Pause':'▶ Play';if(playing){timer=setInterval(next,speed)}else{clearInterval(timer)}}
function setSpeed(s){speed=s;if(playing){clearInterval(timer);timer=setInterval(next,speed)}document.querySelectorAll('.speed-group button').forEach(b=>b.classList.remove('active'));event.target.classList.add('active')}
function goTo(i){show(i)}
</script>
</body>
</html>''')


# ── jacobi.html template ──────────────────────────────────────────────
JACOBI_TEMPLATE = JINJA_ENV.from_string(r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Jacobi Field Slideshow</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  :root{--bg:#08090d;--card:#12152a;--border:#1e2340;--accent:#00d4aa;--accent2:#7c5cfc;--text:#e8eaf6;--muted:#6b70a0}
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);display:flex;flex-direction:column;align-items:center;min-height:100vh;padding:1rem}
  h1{font-size:1.5rem;margin:1rem 0 0.5rem;font-weight:700}
  .controls{display:flex;align-items:center;gap:1rem;margin:1rem 0;flex-wrap:wrap;justify-content:center}
  button{background:var(--card);border:1px solid var(--border);color:var(--text);padding:0.5rem 1.2rem;border-radius:8px;cursor:pointer;font-size:0.85rem;font-weight:600;transition:background 0.15s}
  button:hover{background:var(--accent);border-color:var(--accent)}
  button.active{background:var(--accent2);border-color:var(--accent2)}
  .speed-group{display:flex;gap:0.3rem}
  .frame-info{font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:var(--muted);min-width:200px;text-align:center}
  .slider-row{width:100%;max-width:900px;margin:0.5rem 0}
  input[type=range]{width:100%;accent-color:var(--accent)}
  .layer-grid{display:flex;flex-wrap:wrap;gap:1rem;justify-content:center;margin:1rem 0}
  .layer-card{border:1px solid var(--border);border-radius:12px;overflow:hidden;background:#000}
  .layer-card img{max-width:420px;max-height:420px;display:block}
  .layer-card .layer-label{padding:0.4rem 0.8rem;font-size:0.75rem;color:var(--muted);font-family:'JetBrains Mono',monospace;background:var(--card);border-top:1px solid var(--border);text-align:center}
  a.back{color:var(--accent);text-decoration:none;font-size:0.85rem;margin-top:1rem}
</style>
</head>
<body>
<h1>🌊 Jacobi Field History</h1>
<div class="controls">
  <button onclick="prev()">⏮ Prev</button>
  <button id="playBtn" onclick="togglePlay()">▶ Play</button>
  <button onclick="next()">Next ⏭</button>
  <div class="speed-group">
    <button onclick="setSpeed(2000)">🐢</button>
    <button onclick="setSpeed(800)" class="active" id="speedMed">🐇</button>
    <button onclick="setSpeed(200)">⚡</button>
  </div>
</div>
<div class="slider-row"><input type="range" id="slider" min="0" max="{{ steps|length - 1 }}" value="0" oninput="goTo(+this.value)"></div>
<div class="frame-info" id="info">Step 1 / {{ steps|length }}</div>
<div class="layer-grid" id="layerGrid"></div>
<a class="back" href="index.html">← Back to Dashboard</a>
<script>
const data = {{ data_json }};
const steps = {{ steps_json }};
let idx = 0, playing = false, timer = null, speed = 800;
function show(i){
  idx=i;
  const step = steps[i];
  const layers = data[step];
  document.getElementById('slider').value=i;
  document.getElementById('info').textContent=`Step ${step} (${i+1} / ${steps.length})`;
  let html='';
  for(const [layer, path] of layers){
    html+=`<div class="layer-card"><img src="${path}"><div class="layer-label">Layer ${layer}</div></div>`;
  }
  document.getElementById('layerGrid').innerHTML=html;
}
function next(){show((idx+1)%steps.length)}
function prev(){show((idx-1+steps.length)%steps.length)}
function togglePlay(){playing=!playing;document.getElementById('playBtn').textContent=playing?'⏸ Pause':'▶ Play';if(playing){timer=setInterval(next,speed)}else{clearInterval(timer)}}
function setSpeed(s){speed=s;if(playing){clearInterval(timer);timer=setInterval(next,speed)}document.querySelectorAll('.speed-group button').forEach(b=>b.classList.remove('active'));event.target.classList.add('active')}
function goTo(i){show(i)}
show(0);
</script>
</body>
</html>''')


# ── 4d_jacobi.html template ───────────────────────────────────────────
JACOBI_4D_TEMPLATE = JINJA_ENV.from_string(r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>4D Jacobi Slices Slideshow</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  :root{--bg:#08090d;--card:#12152a;--border:#1e2340;--accent:#f472b6;--accent2:#7c5cfc;--text:#e8eaf6;--muted:#6b70a0}
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);display:flex;flex-direction:column;align-items:center;min-height:100vh;padding:1rem}
  h1{font-size:1.5rem;margin:1rem 0 0.5rem;font-weight:700}
  .subtitle{color:var(--muted);font-size:0.85rem;margin-bottom:1rem}
  .controls{display:flex;align-items:center;gap:1rem;margin:1rem 0;flex-wrap:wrap;justify-content:center}
  button{background:var(--card);border:1px solid var(--border);color:var(--text);padding:0.5rem 1.2rem;border-radius:8px;cursor:pointer;font-size:0.85rem;font-weight:600;transition:background 0.15s}
  button:hover{background:var(--accent);border-color:var(--accent)}
  button.active{background:var(--accent2);border-color:var(--accent2)}
  .speed-group{display:flex;gap:0.3rem}
  .frame-info{font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:var(--muted);min-width:200px;text-align:center}
  .slider-row{width:100%;max-width:900px;margin:0.5rem 0}
  input[type=range]{width:100%;accent-color:var(--accent)}
  .step-group{margin:1rem 0;width:100%;max-width:1400px}
  .step-group h2{font-size:1rem;font-weight:600;color:var(--accent);margin-bottom:0.8rem;font-family:'JetBrains Mono',monospace}
  .layer-section{margin-bottom:1.5rem}
  .layer-section h3{font-size:0.85rem;font-weight:600;color:var(--accent2);margin-bottom:0.5rem;font-family:'JetBrains Mono',monospace}
  .slice-grid{display:flex;flex-wrap:wrap;gap:0.8rem;justify-content:center}
  .slice-card{border:1px solid var(--border);border-radius:10px;overflow:hidden;background:#000;transition:transform 0.2s,box-shadow 0.2s}
  .slice-card:hover{transform:translateY(-3px);box-shadow:0 8px 24px rgba(244,114,182,0.15)}
  .slice-card img{max-width:320px;max-height:320px;display:block}
  .slice-card .slice-label{padding:0.35rem 0.7rem;font-size:0.72rem;color:var(--muted);font-family:'JetBrains Mono',monospace;background:var(--card);border-top:1px solid var(--border);text-align:center}
  a.back{color:var(--accent);text-decoration:none;font-size:0.85rem;margin-top:1rem}
</style>
</head>
<body>
<h1>🔮 4D Jacobi Slices</h1>
<div class="subtitle">Pairwise dimension slices of the 4D Jacobian, grouped by step and layer</div>
<div class="controls">
  <button onclick="prev()">⏮ Prev</button>
  <button id="playBtn" onclick="togglePlay()">▶ Play</button>
  <button onclick="next()">Next ⏭</button>
  <div class="speed-group">
    <button onclick="setSpeed(3000)">🐢</button>
    <button onclick="setSpeed(1200)" class="active" id="speedMed">🐇</button>
    <button onclick="setSpeed(400)">⚡</button>
  </div>
</div>
<div class="slider-row"><input type="range" id="slider" min="0" max="{{ steps|length - 1 }}" value="0" oninput="goTo(+this.value)"></div>
<div class="frame-info" id="info">Step 1 / {{ steps|length }}</div>
<div class="step-group" id="stepGroup"></div>
<a class="back" href="index.html">← Back to Dashboard</a>
<script>
const data = {{ data_json }};
const steps = {{ steps_json }};
let idx = 0, playing = false, timer = null, speed = 1200;
function show(i){
  idx=i;
  const step = steps[i];
  const layers = data[step];
  document.getElementById('slider').value=i;
  document.getElementById('info').textContent=`Step ${step} (${i+1} / ${steps.length})`;
  let html=`<h2>Step ${step}</h2>`;
  // layers is {layer_num: [{dim_a, dim_b, path}, ...]}
  const layerNums = Object.keys(layers).map(Number).sort((a,b)=>a-b);
  for(const ln of layerNums){
    html+=`<div class="layer-section"><h3>Layer ${ln}</h3><div class="slice-grid">`;
    const slices = layers[ln];
    for(const s of slices){
      html+=`<div class="slice-card"><img src="${s.path}"><div class="slice-label">d${s.dim_a} × d${s.dim_b}</div></div>`;
    }
    html+=`</div></div>`;
  }
  document.getElementById('stepGroup').innerHTML=html;
}
function next(){show((idx+1)%steps.length)}
function prev(){show((idx-1+steps.length)%steps.length)}
function togglePlay(){playing=!playing;document.getElementById('playBtn').textContent=playing?'⏸ Pause':'▶ Play';if(playing){timer=setInterval(next,speed)}else{clearInterval(timer)}}
function setSpeed(s){speed=s;if(playing){clearInterval(timer);timer=setInterval(next,speed)}document.querySelectorAll('.speed-group button').forEach(b=>b.classList.remove('active'));event.target.classList.add('active')}
function goTo(i){show(i)}
show(0);
</script>
</body>
</html>''')


# ═════════════════════════════════════════════════════════════════════════
# 6. MAIN — generate all HTML files
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

    # ── Generate index.html ─────────────────────────────────────────
    # Determine unique jacobi 4D steps and slices
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
    )
    (run_dir / "index.html").write_text(index_html)
    print(f"  ✓ index.html")

    # ── Generate slideshow.html ─────────────────────────────────────
    if rf.training_hist:
        frames = subsample(rf.training_hist, MAX_SLIDESHOW_FRAMES)
        slideshow_html = SLIDESHOW_TEMPLATE.render(
            frames=frames,
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

        # Build data: {step: [(layer, path), ...]}
        data = {s: step_map[s] for s in sampled_steps}

        jacobi_html = JACOBI_TEMPLATE.render(
            steps=sampled_steps,
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

        # Build data: {step: {layer: [{dim_a, dim_b, path}, ...]}}
        data_4d = {}
        for s in sampled_steps_4d:
            data_4d[s] = {}
            for layer in sorted(step_layer_map[s].keys()):
                data_4d[s][layer] = step_layer_map[s][layer]

        jacobi_4d_html = JACOBI_4D_TEMPLATE.render(
            steps=sampled_steps_4d,
            steps_json=json.dumps(sampled_steps_4d),
            data_json=json.dumps(data_4d),
        )
        (run_dir / "4d_jacobi.html").write_text(jacobi_4d_html)
        print(f"  ✓ 4d_jacobi.html ({len(sampled_steps_4d)} steps, {len(jacobi_4d_slices)} slice pairs)")

    elapsed_gen = time.time() - t0
    print(f"[dashboard] Done in {elapsed_gen:.2f}s")


if __name__ == "__main__":
    main()
