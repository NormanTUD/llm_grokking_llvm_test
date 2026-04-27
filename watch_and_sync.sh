#!/bin/bash

usage() {
    echo "Usage: $0 <run_folder> [--copy-to user@server:/remote/path] [--interval seconds]"
    echo ""
    echo "Beispiel:"
    echo "  $0 runs/0/"
    echo "  $0 runs/0/ --copy-to root@myserver.de:/var/www/grok_test/"
    echo "  $0 runs/0/ --copy-to root@myserver.de:/var/www/grok_test/ --interval 30"
    exit 1
}

if [ -z "$1" ]; then
    usage
fi

RUN_DIR="${1%/}/"
shift

COPY_TO=""
INTERVAL=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --copy-to)
            COPY_TO="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "Unbekanntes Argument: $1"
            usage
            ;;
    esac
done

# -------------------------------------------------------
# Locate the directory where this script lives, so we can
# find the viz_*.py files relative to it.
# -------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -------------------------------------------------------
# List of visualization scripts to run each cycle.
# Each produces a .png inside the run folder automatically.
# -------------------------------------------------------
VIZ_SCRIPTS=(
    "viz_fibre_bundle.py"
    "viz_persistent_homology.py"
    "viz_kelp_forest.py"
)

parse_remote() {
    REMOTE_USERHOST="${COPY_TO%%:*}"
    REMOTE_PATH="${COPY_TO#*:}"
    REMOTE_PATH="${REMOTE_PATH%/}/"
}

ensure_remote_dir() {
    if [ -n "$COPY_TO" ]; then
        parse_remote
        echo "Stelle sicher, dass Remote-Verzeichnis existiert: $REMOTE_PATH"
        ssh "$REMOTE_USERHOST" "mkdir -p '$REMOTE_PATH'"
    fi
}

# -------------------------------------------------------
# Run all visualization scripts against the run folder
# and copy the .py source files into the run folder so
# they are included in the rsync / HTML dashboard.
# -------------------------------------------------------
run_visualizations() {
    local DIR="$1"

    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  Running Python visualization scripts                   ║"
    echo "╚══════════════════════════════════════════════════════════╝"

    for viz in "${VIZ_SCRIPTS[@]}"; do
        local VIZ_PATH="${SCRIPT_DIR}/${viz}"

        if [ ! -f "$VIZ_PATH" ]; then
            echo "  ⚠ Visualization script not found: $VIZ_PATH — skipping."
            continue
        fi

        # --- Copy the .py source into the run folder so it gets synced ---
        local DEST_PY="${DIR}${viz}"
        if [ "$VIZ_PATH" != "$(realpath "$DEST_PY" 2>/dev/null)" ]; then
            cp -f "$VIZ_PATH" "$DEST_PY"
            echo "  📄 Copied ${viz} → ${DEST_PY}"
        fi

        # --- Execute the visualization script ---
        echo "  🔬 Running: python3 ${viz} ${DIR}"
        python3 "$VIZ_PATH" "${DIR}" 2>&1 | sed 's/^/     /'
        local EXIT_CODE=${PIPESTATUS[0]}

        if [ "$EXIT_CODE" -eq 0 ]; then
            echo "  ✅ ${viz} completed successfully."
        else
            echo "  ❌ ${viz} exited with code ${EXIT_CODE} — continuing."
        fi
        echo ""
    done
}

# -------------------------------------------------------
# Generate a standalone HTML slideshow for training_plot history
# -------------------------------------------------------
generate_slideshow_html() {
    local DIR="$1"
    local OUT="${DIR}slideshow.html"

    # Collect all training_plot history images in order
    local IMAGES=()
    while IFS= read -r f; do
        [ -n "$f" ] && IMAGES+=("${f#./}")
    done < <(cd "$DIR" && find . -type f -name 'training_plot-*.png' 2>/dev/null | sort)

    # Add the current latest as the final frame
    if [ -f "${DIR}training_plot.png" ]; then
        IMAGES+=("training_plot.png")
    fi

    if [ ${#IMAGES[@]} -eq 0 ]; then
        return
    fi

    local TOTAL=${#IMAGES[@]}
    local TIMESTAMP
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S %Z')

    cat > "$OUT" <<'SLIDESHOW_HEAD'
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Training Plot History</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #08090d;
    color: #e8eaf6;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    user-select: none;
  }
  .toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 1.5rem;
    background: #12152a;
    border-bottom: 1px solid #1e2340;
    flex-shrink: 0;
    z-index: 10;
  }
  .toolbar .title {
    font-size: 1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #7c5cfc, #00d4aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .toolbar .controls {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }
  .toolbar button {
    background: #1e2340;
    border: 1px solid #2a2f55;
    color: #e8eaf6;
    padding: 0.35rem 0.9rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    transition: background 0.15s, border-color 0.15s;
  }
  .toolbar button:hover {
    background: #2a2f55;
    border-color: #7c5cfc;
  }
  .toolbar button:active {
    background: #7c5cfc;
    color: #fff;
  }
  .toolbar button.playing {
    background: #00d4aa;
    color: #08090d;
    border-color: #00d4aa;
  }
  .counter {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #6b70a0;
    min-width: 120px;
    text-align: center;
  }
  .counter .current {
    color: #7c5cfc;
    font-weight: 700;
  }
  .progress-bar-container {
    flex-shrink: 0;
    height: 4px;
    background: #1e2340;
    cursor: pointer;
    position: relative;
  }
  .progress-bar-container:hover {
    height: 8px;
  }
  .progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #7c5cfc, #00d4aa);
    transition: width 0.1s ease;
    border-radius: 0 2px 2px 0;
  }
  .slide-container {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    position: relative;
    background: #000;
  }
  .slide-container img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
  }
  .hint {
    position: absolute;
    bottom: 1rem;
    left: 50%;
    transform: translateX(-50%);
    font-size: 0.75rem;
    color: #4a4f78;
    pointer-events: none;
    opacity: 0.7;
  }
  .speed-label {
    font-size: 0.7rem;
    color: #6b70a0;
  }
</style>
</head>
<body>
SLIDESHOW_HEAD

    # Build the JS image array
    cat >> "$OUT" <<EOF
<div class="toolbar">
  <div class="title">Training Plot History</div>
  <div class="controls">
    <button id="btn-start" title="Go to first (Home)">⏮</button>
    <button id="btn-prev" title="Previous (←)">◀</button>
    <button id="btn-play" title="Play/Pause (Space)">▶</button>
    <button id="btn-next" title="Next (→)">▶▶</button>
    <button id="btn-end" title="Go to last (End)">⏭</button>
    <span class="speed-label">Speed:</span>
    <button id="btn-slower" title="Slower">−</button>
    <span id="speed-display" class="counter" style="min-width:50px;">500ms</span>
    <button id="btn-faster" title="Faster">+</button>
    <div class="counter"><span class="current" id="frame-num">1</span> / ${TOTAL}</div>
  </div>
</div>
<div class="progress-bar-container" id="progress-bar">
  <div class="progress-bar-fill" id="progress-fill" style="width: 0%"></div>
</div>
<div class="slide-container">
  <img id="slide-img" src="" alt="Training plot">
  <div class="hint">← → arrow keys · Space to play/pause · Home/End to jump</div>
</div>
<script>
const images = [
EOF

    # Write each image path as a JS string
    for img in "${IMAGES[@]}"; do
        local cb="?t=$(date +%s)"
        echo "  \"${img}${cb}\"," >> "$OUT"
    done

    cat >> "$OUT" <<'SLIDESHOW_JS'
];
const total = images.length;
let idx = total - 1;  // start at the latest
let playing = false;
let playInterval = null;
let speed = 500;  // ms between frames

const imgEl = document.getElementById('slide-img');
const frameNum = document.getElementById('frame-num');
const progressFill = document.getElementById('progress-fill');
const btnPlay = document.getElementById('btn-play');
const speedDisplay = document.getElementById('speed-display');

function show(i) {
  idx = Math.max(0, Math.min(total - 1, i));
  imgEl.src = images[idx];
  frameNum.textContent = idx + 1;
  progressFill.style.width = ((idx + 1) / total * 100) + '%';
}

function next() { show(idx + 1); if (idx >= total - 1 && playing) togglePlay(); }
function prev() { show(idx - 1); }
function goStart() { show(0); }
function goEnd() { show(total - 1); }

function togglePlay() {
  playing = !playing;
  btnPlay.textContent = playing ? '⏸' : '▶';
  btnPlay.classList.toggle('playing', playing);
  if (playing) {
    playInterval = setInterval(next, speed);
  } else {
    clearInterval(playInterval);
    playInterval = null;
  }
}

function updateSpeed(newSpeed) {
  speed = Math.max(50, Math.min(5000, newSpeed));
  speedDisplay.textContent = speed + 'ms';
  if (playing) {
    clearInterval(playInterval);
    playInterval = setInterval(next, speed);
  }
}

document.getElementById('btn-next').addEventListener('click', next);
document.getElementById('btn-prev').addEventListener('click', prev);
document.getElementById('btn-start').addEventListener('click', goStart);
document.getElementById('btn-end').addEventListener('click', goEnd);
document.getElementById('btn-play').addEventListener('click', togglePlay);
document.getElementById('btn-slower').addEventListener('click', () => updateSpeed(speed + 100));
document.getElementById('btn-faster').addEventListener('click', () => updateSpeed(speed - 100));

document.getElementById('progress-bar').addEventListener('click', (e) => {
  const rect = e.currentTarget.getBoundingClientRect();
  const pct = (e.clientX - rect.left) / rect.width;
  show(Math.round(pct * (total - 1)));
});

document.addEventListener('keydown', (e) => {
  switch (e.key) {
    case 'ArrowRight': e.preventDefault(); next(); break;
    case 'ArrowLeft':  e.preventDefault(); prev(); break;
    case 'Home':       e.preventDefault(); goStart(); break;
    case 'End':        e.preventDefault(); goEnd(); break;
    case ' ':          e.preventDefault(); togglePlay(); break;
    case 'ArrowUp':    e.preventDefault(); updateSpeed(speed - 100); break;
    case 'ArrowDown':  e.preventDefault(); updateSpeed(speed + 100); break;
  }
});

// Preload adjacent images for smooth navigation
function preload(i) {
  if (i >= 0 && i < total) { const img = new Image(); img.src = images[i]; }
}
const observer = new MutationObserver(() => { preload(idx + 1); preload(idx - 1); });

show(idx);
</script>
</body>
</html>
SLIDESHOW_JS

    echo "Slideshow generated: $OUT (${TOTAL} frames)"
}

# -------------------------------------------------------
# Use awk to parse ALL epoch .txt files at once
# -------------------------------------------------------
generate_epoch_html() {
    local DIR="$1"
    local OVERVIEW_FILE="$2"
    local DETAILS_FILE="$3"

    local EPOCH_FILES
    EPOCH_FILES=$(cd "$DIR" && find . -type f -name 'epoch_*.txt' | sort)

    if [ -z "$EPOCH_FILES" ]; then
        echo '<div class="empty-state">No epoch sample files found yet.</div>' > "$OVERVIEW_FILE"
        echo "" > "$DETAILS_FILE"
        return
    fi

    local FULL_PATHS=""
    while IFS= read -r f; do
        FULL_PATHS="${FULL_PATHS} ${DIR}${f#./}"
    done <<< "$EPOCH_FILES"

    awk '
    BEGIN {
        epoch = 0; sample = 0; correct_count = 0; total = 0;
        params = ""; expected = ""; predicted = ""; is_correct = "";
        current_file = ""; epoch_idx = 0;
    }

    FNR == 1 {
        if (current_file != "" && total > 0) {
            pct = int((correct_count * 100) / total)
            overview[epoch_idx] = epoch "\t" correct_count "\t" total "\t" pct
            epoch_order[epoch_idx] = epoch
            epoch_idx++
        }
        current_file = FILENAME
        match(FILENAME, /epoch_0*([0-9]+)\.txt/, m)
        epoch = m[1] + 0
        sample = 0; correct_count = 0; total = 0
        detail_rows[epoch] = ""
    }

    {
        line = $0
        gsub(/^[[:space:]]+/, "", line)

        if (line ~ /^params:/) {
            params = line; sub(/^params:[[:space:]]*/, "", params)
        }
        if (line ~ /^expected:/) {
            expected = line; sub(/^expected:[[:space:]]*/, "", expected)
        }
        if (line ~ /^predicted:/) {
            predicted = line; sub(/^predicted:[[:space:]]*/, "", predicted)
        }
        if (line ~ /^correct:/) {
            is_correct = line; sub(/^correct:[[:space:]]*/, "", is_correct)
            total++
            sample++
            if (is_correct == "True") {
                correct_count++
                cls = "row-correct"
                icon = "✔"
            } else {
                cls = "row-wrong"
                icon = "✗"
            }
            detail_rows[epoch] = detail_rows[epoch] \
                "<tr class=\"" cls "\"><td>" sample "</td><td>" params "</td><td>" expected "</td><td>" predicted "</td><td>" icon "</td></tr>\n"
            params = ""; expected = ""; predicted = ""; is_correct = ""
        }
    }

    END {
        if (total > 0) {
            pct = int((correct_count * 100) / total)
            overview[epoch_idx] = epoch "\t" correct_count "\t" total "\t" pct
            epoch_order[epoch_idx] = epoch
            epoch_idx++
        }

        # Overview table
        print "<div class=\"table-scroll\"><table class=\"epoch-overview\">"
        print "<thead><tr><th>Epoch</th><th>Correct</th><th>Total</th><th>Accuracy</th><th style=\"min-width:150px\">Progress</th></tr></thead>"
        print "<tbody>"
        for (i = 0; i < epoch_idx; i++) {
            split(overview[i], f, "\t")
            en = f[1]; ec = f[2]; et = f[3]; pct = f[4]
            if (pct >= 80) { acc_cls = "acc-good"; bar_col = "var(--green)" }
            else if (pct >= 50) { acc_cls = "acc-mid"; bar_col = "#f0c040" }
            else { acc_cls = "acc-bad"; bar_col = "var(--red)" }
            print "<tr><td>" en "</td><td>" ec "</td><td>" et "</td><td class=\"" acc_cls "\">" pct "%</td>"
            print "<td><div class=\"acc-bar-bg\"><div class=\"acc-bar-fg\" style=\"width:" pct "%;background:" bar_col "\"></div></div></td></tr>"
        }
        print "</tbody></table></div>"
        print "===SPLIT==="

        # Collapsible details
        for (i = 0; i < epoch_idx; i++) {
            split(overview[i], f, "\t")
            en = f[1]; ec = f[2]; et = f[3]; pct = f[4]
            if (pct >= 80) acc_cls = "acc-good"
            else if (pct >= 50) acc_cls = "acc-mid"
            else acc_cls = "acc-bad"
            print "<details>"
            print "<summary><strong>Epoch " en "</strong> <span class=\"" acc_cls "\">" ec "/" et " correct (" pct "%)</span></summary>"
            print "<div class=\"detail-body\"><table class=\"sample-table\">"
            print "<thead><tr><th>#</th><th>Params</th><th>Expected</th><th>Predicted</th><th>✓</th></tr></thead><tbody>"
            printf "%s", detail_rows[en]
            print "</tbody></table></div></details>"
        }
    }
    ' $FULL_PATHS > /tmp/_epoch_combined.html

    awk '/===SPLIT===/{found=1; next} !found{print}' /tmp/_epoch_combined.html > "$OVERVIEW_FILE"
    awk '/===SPLIT===/{found=1; next} found{print}' /tmp/_epoch_combined.html > "$DETAILS_FILE"
}

# -------------------------------------------------------
# Generate HTML dashboard
# -------------------------------------------------------
generate_html() {
    local DIR="$1"
    local OUT="${DIR}index.html"
    local TIMESTAMP
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S %Z')

    echo "Generiere HTML Dashboard..."

    # --- Read started_at.txt ---
    local STARTED_AT=""
    local ELAPSED_STR=""
    if [ -f "${DIR}started_at.txt" ]; then
        STARTED_AT=$(cat "${DIR}started_at.txt" | head -1 | xargs)
        local started_clean
        started_clean=$(echo "$STARTED_AT" | sed 's/ CEST$/+0200/' | sed 's/ CET$/+0100/')
        local started_epoch
        started_epoch=$(date -d "$started_clean" +%s 2>/dev/null)
        if [ -n "$started_epoch" ]; then
            local now_epoch
            now_epoch=$(date +%s)
            local diff=$((now_epoch - started_epoch))
            if [ "$diff" -ge 0 ]; then
                local days=$((diff / 86400))
                local hours=$(( (diff % 86400) / 3600 ))
                local mins=$(( (diff % 3600) / 60 ))
                local secs=$((diff % 60))
                if [ "$days" -gt 0 ]; then
                    ELAPSED_STR="${days}d ${hours}h ${mins}m"
                elif [ "$hours" -gt 0 ]; then
                    ELAPSED_STR="${hours}h ${mins}m ${secs}s"
                else
                    ELAPSED_STR="${mins}m ${secs}s"
                fi
            fi
        fi
    fi

    # Collect files
    local IMAGES_REL=()
    local CSV_REL=()
    local EPOCH_TXT_REL=()
    local EPOCH_CSV_REL=()
    local OTHER_TXT_REL=()
    local OTHER_CSV_REL=()
    local PY_REL=()

    while IFS= read -r f; do
        [ -n "$f" ] && IMAGES_REL+=("$f")
    done < <(cd "$DIR" && find . -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.svg" \) | sort)

    while IFS= read -r f; do
        if [ -n "$f" ]; then
            local base
            base=$(basename "$f")
            if [[ "$base" =~ ^epoch_[0-9]+\.csv$ ]]; then
                EPOCH_CSV_REL+=("$f")
            else
                OTHER_CSV_REL+=("$f")
            fi
        fi
    done < <(cd "$DIR" && find . -type f -iname "*.csv" | sort)

    while IFS= read -r f; do
        if [ -n "$f" ]; then
            local base
            base=$(basename "$f")
            if [[ "$base" == "started_at.txt" ]]; then
                continue
            elif [[ "$base" =~ ^epoch_[0-9]+\.txt$ ]]; then
                EPOCH_TXT_REL+=("$f")
            else
                OTHER_TXT_REL+=("$f")
            fi
        fi
    done < <(cd "$DIR" && find . -type f -iname "*.txt" | sort)

    # Collect .py files (the viz scripts we copied in)
    while IFS= read -r f; do
        [ -n "$f" ] && PY_REL+=("$f")
    done < <(cd "$DIR" && find . -type f -iname "*.py" | sort)

    local EPOCH_COUNT=${#EPOCH_TXT_REL[@]}
    local TOTAL_FILES=$(( ${#IMAGES_REL[@]} + ${#EPOCH_TXT_REL[@]} + ${#EPOCH_CSV_REL[@]} + ${#OTHER_CSV_REL[@]} + ${#OTHER_TXT_REL[@]} + ${#PY_REL[@]} ))

    # Parse epochs with awk
    local OVERVIEW_TMP="/tmp/_epoch_overview.html"
    local DETAILS_TMP="/tmp/_epoch_details.html"
    generate_epoch_html "$DIR" "$OVERVIEW_TMP" "$DETAILS_TMP"

    local LATEST_ACC=""
    if [ -f "$OVERVIEW_TMP" ]; then
        LATEST_ACC=$(grep -oP '\d+%' "$OVERVIEW_TMP" | tail -1)
    fi

    # --- Write HTML ---
    cat > "$OUT" <<'HEADER'
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Grokking Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

  :root {
    --bg: #08090d; --bg2: #0d0f16; --card: #12152a; --card2: #181c35;
    --border: #1e2340; --border-light: #2a2f55;
    --accent: #7c5cfc; --accent-dim: rgba(124,92,252,0.12);
    --accent2: #00d4aa; --accent2-dim: rgba(0,212,170,0.12);
    --accent3: #f472b6;
    --text: #e8eaf6; --muted: #6b70a0; --muted2: #4a4f78;
    --glow: rgba(124, 92, 252, 0.08);
    --green: #00d4aa; --red: #ff5c72; --yellow: #f0c040;
    --radius: 16px; --radius-sm: 10px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }

  /* Hero */
  .hero {
    position: relative;
    background: linear-gradient(160deg, #130f30 0%, #08090d 40%, #061215 100%);
    border-bottom: 1px solid var(--border);
    padding: 3rem 2rem 2rem;
    text-align: center;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(ellipse at 30% 20%, rgba(124,92,252,0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 70% 80%, rgba(0,212,170,0.04) 0%, transparent 50%);
    pointer-events: none;
  }
  .hero h1 {
    font-size: 2.8rem; font-weight: 900; letter-spacing: -0.03em;
    background: linear-gradient(135deg, #7c5cfc 0%, #00d4aa 50%, #f472b6 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: gradient-shift 8s ease infinite;
    margin-bottom: 0.4rem;
    position: relative;
  }
  @keyframes gradient-shift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
  }
  .hero .subtitle { color: var(--muted); font-size: 0.95rem; font-weight: 400; position: relative; }
  .hero .meta-row {
    display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;
    margin-top: 1rem; position: relative;
  }
  .meta-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.35rem 1rem;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 999px; font-size: 0.78rem; color: var(--muted);
  }
  .meta-pill .val { color: var(--accent2); font-weight: 600; }
  .meta-pill .val-warm { color: var(--accent3); font-weight: 600; }
  .meta-pill .val-purple { color: var(--accent); font-weight: 600; }

  /* Stats ribbon */
  .stats-ribbon {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1px; background: var(--border);
    border: 1px solid var(--border); border-radius: var(--radius);
    overflow: hidden; margin: 2rem 0;
  }
  .stat-card {
    background: var(--card); padding: 1.2rem 1.5rem;
    text-align: center;
  }
  .stat-card .stat-value {
    font-size: 1.8rem; font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
  }
  .stat-card .stat-label {
    font-size: 0.75rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-top: 0.2rem;
  }
  .stat-green { color: var(--green); }
  .stat-purple { color: var(--accent); }
  .stat-pink { color: var(--accent3); }
  .stat-yellow { color: var(--yellow); }

  .container { max-width: 1440px; margin: 0 auto; padding: 0 1.5rem 3rem; }

  /* Section titles */
  .section-title {
    font-size: 1.2rem; font-weight: 700; letter-spacing: -0.01em;
    margin: 2.5rem 0 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 0.6rem;
  }
  .badge {
    background: var(--accent-dim); color: var(--accent);
    font-size: 0.7rem; padding: 0.2rem 0.6rem;
    border-radius: 999px; font-weight: 600;
    border: 1px solid rgba(124,92,252,0.2);
  }
  .badge-green {
    background: var(--accent2-dim); color: var(--accent2);
    border-color: rgba(0,212,170,0.2);
  }
  .badge-orange {
    background: rgba(240,192,64,0.12); color: var(--yellow);
    border-color: rgba(240,192,64,0.2);
  }

  /* Image grid */
  .image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(440px, 1fr));
    gap: 1.2rem;
  }
  .image-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
  }
  .image-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(124,92,252,0.12);
    border-color: var(--border-light);
  }
  .image-card img { width: 100%; display: block; background: #000; }
  .image-card .label {
    padding: 0.65rem 1rem; font-size: 0.78rem; color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    border-top: 1px solid var(--border);
    display: flex; align-items: center; gap: 0.4rem;
  }
  .image-card .label::before {
    content: '🖼'; font-size: 0.7rem;
  }

  /* Epoch overview table */
  .table-scroll {
    overflow-x: auto; border-radius: var(--radius);
    border: 1px solid var(--border);
  }
  .epoch-overview {
    width: 100%; border-collapse: collapse;
  }
  .epoch-overview th {
    background: var(--card2); text-align: left;
    padding: 0.7rem 1rem; font-size: 0.75rem;
    color: var(--accent); border-bottom: 2px solid var(--border);
    position: sticky; top: 0; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
  }
  .epoch-overview td {
    padding: 0.5rem 1rem; font-size: 0.82rem;
    border-bottom: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    background: var(--card);
  }
  .epoch-overview tr:hover td { background: var(--card2); }
  .acc-good { color: var(--green); font-weight: 700; }
  .acc-mid  { color: var(--yellow); font-weight: 700; }
  .acc-bad  { color: var(--red); font-weight: 700; }
  .acc-bar-bg {
    width: 130px; height: 8px; background: rgba(255,255,255,0.05);
    border-radius: 4px; overflow: hidden; display: inline-block;
    vertical-align: middle;
  }
  .acc-bar-fg { height: 100%; border-radius: 4px; transition: width 0.3s ease; }

  /* Collapsible epoch details */
  details { margin-bottom: 0.4rem; }
  details summary {
    cursor: pointer; padding: 0.55rem 1rem;
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius-sm); font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
    display: flex; align-items: center; gap: 0.8rem;
    transition: background 0.15s, border-color 0.15s;
    list-style: none;
  }
  details summary::-webkit-details-marker { display: none; }
  details summary::before {
    content: '▸'; color: var(--muted); transition: transform 0.2s;
    font-size: 0.9rem; flex-shrink: 0;
  }
  details[open] summary::before { transform: rotate(90deg); }
  details summary:hover { background: var(--card2); border-color: var(--border-light); }
  details[open] summary { border-radius: var(--radius-sm) var(--radius-sm) 0 0; border-bottom-color: transparent; }
  details .detail-body {
    background: var(--card); border: 1px solid var(--border);
    border-top: none; border-radius: 0 0 var(--radius-sm) var(--radius-sm);
    overflow-x: auto;
  }
  .sample-table { width: 100%; border-collapse: collapse; }
  .sample-table th {
    background: var(--card2); text-align: left;
    padding: 0.4rem 0.8rem; font-size: 0.7rem;
    color: var(--muted); border-bottom: 1px solid var(--border);
    text-transform: uppercase; letter-spacing: 0.05em;
  }
  .sample-table td {
    padding: 0.35rem 0.8rem; font-size: 0.78rem;
    border-bottom: 1px solid rgba(30,35,64,0.6);
    font-family: 'JetBrains Mono', monospace;
  }
  .row-correct td { color: var(--green); }
  .row-wrong td { color: var(--red); }

  /* File list */
  .file-link-grid {
    display: flex; flex-wrap: wrap; gap: 0.5rem;
  }
  .file-link {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.3rem 0.75rem;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 999px; font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent); text-decoration: none;
    transition: background 0.15s, border-color 0.15s;
  }
  .file-link:hover { background: var(--card2); border-color: var(--accent); }
  .file-link-green { color: var(--accent2); }
  .file-link-green:hover { border-color: var(--accent2); }

  /* File cards */
  .file-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 1rem;
  }
  .file-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .file-card:hover { transform: translateY(-2px); box-shadow: 0 6px 24px var(--glow); }
  .file-card .file-header {
    padding: 0.65rem 1rem; font-size: 0.82rem; font-weight: 600;
    background: linear-gradient(90deg, var(--accent-dim), transparent);
    border-bottom: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    display: flex; align-items: center; gap: 0.4rem;
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .dot-csv { background: var(--accent2); }
  .dot-txt { background: var(--accent); }
  .dot-py  { background: var(--yellow); }
  .file-card pre {
    padding: 0.8rem 1rem; font-size: 0.72rem; line-height: 1.55;
    overflow-x: auto; max-height: 300px; overflow-y: auto;
    color: #aab; background: var(--bg2);
    font-family: 'JetBrains Mono', monospace;
  }
  .file-card a.file-open {
    display: block; padding: 0.6rem 1rem; text-align: center;
    color: var(--accent); text-decoration: none;
    font-size: 0.8rem; font-weight: 600;
    border-top: 1px solid var(--border);
    transition: background 0.15s;
  }
  .file-card a.file-open:hover { background: var(--accent-dim); }

  .empty-state { color: var(--muted); font-style: italic; padding: 1rem 0; font-size: 0.9rem; }

  /* Footer */
  .footer {
    text-align: center; padding: 2rem; color: var(--muted2);
    font-size: 0.75rem; border-top: 1px solid var(--border);
    margin-top: 2rem;
  }
  .footer a { color: var(--muted); text-decoration: none; }
  .footer a:hover { color: var(--accent); }

  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--border-light); }

  @media (max-width: 600px) {
    .image-grid, .file-grid { grid-template-columns: 1fr; }
    .hero h1 { font-size: 1.8rem; }
    .stats-ribbon { grid-template-columns: repeat(2, 1fr); }
  }
</style>
</head>
<body>
HEADER

    # --- Hero ---
    cat >> "$OUT" <<EOF
<div class="hero">
  <h1>Grokking Dashboard</h1>
  <div class="subtitle">Live training monitor</div>
  <div class="meta-row">
EOF

    if [ -n "$STARTED_AT" ]; then
        echo "    <div class=\"meta-pill\">🕐 Started <span class=\"val-warm\">${STARTED_AT}</span></div>" >> "$OUT"
    fi
    if [ -n "$ELAPSED_STR" ]; then
        echo "    <div class=\"meta-pill\">⏱ Elapsed <span class=\"val-purple\">${ELAPSED_STR}</span></div>" >> "$OUT"
    fi
    echo "    <div class=\"meta-pill\">📸 Snapshot <span class=\"val\">${TIMESTAMP}</span></div>" >> "$OUT"

    cat >> "$OUT" <<EOF
  </div>
</div>
<div class="container">
EOF

    # --- Stats ribbon ---
    cat >> "$OUT" <<EOF
<div class="stats-ribbon">
  <div class="stat-card">
    <div class="stat-value stat-purple">${EPOCH_COUNT}</div>
    <div class="stat-label">Epochs</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-green">${LATEST_ACC:-—}</div>
    <div class="stat-label">Latest Accuracy</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-pink">${#IMAGES_REL[@]}</div>
    <div class="stat-label">Plots</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-yellow">${TOTAL_FILES}</div>
    <div class="stat-label">Total Files</div>
  </div>
</div>
EOF

    # --- PLOTS ---
    cat >> "$OUT" <<EOF
<div class="section-title">📊 Plots &amp; Visualizations <span class="badge">${#IMAGES_REL[@]}</span></div>
EOF
    if [ ${#IMAGES_REL[@]} -gt 0 ]; then
        echo '<div class="image-grid">' >> "$OUT"
        for img in "${IMAGES_REL[@]}"; do
            local clean="${img#./}"
            local cb="?t=$(date +%s)"
            cat >> "$OUT" <<EOF
  <div class="image-card">
    <a href="${clean}${cb}" target="_blank"><img src="${clean}${cb}" alt="${clean}" loading="lazy"></a>
    <div class="label">${clean}</div>
  </div>
EOF
        done
        echo '</div>' >> "$OUT"
    else
        echo '<div class="empty-state">No images found yet.</div>' >> "$OUT"
    fi

    # --- EPOCH ACCURACY TABLE ---
    cat >> "$OUT" <<EOF
<div class="section-title">🧪 Epoch Accuracy <span class="badge">${EPOCH_COUNT} epochs</span></div>
EOF
    cat "$OVERVIEW_TMP" >> "$OUT"

    # --- EPOCH DETAIL COLLAPSIBLES ---
    if [ -s "$DETAILS_TMP" ]; then
        echo '<h3 style="color:var(--muted);margin:1.5rem 0 0.8rem;font-size:0.9rem;font-weight:500;">Click an epoch to inspect individual predictions:</h3>' >> "$OUT"
        cat "$DETAILS_TMP" >> "$OUT"
    fi

    # --- EPOCH SAMPLE FILES (links only) ---
    if [ ${#EPOCH_TXT_REL[@]} -gt 0 ] || [ ${#EPOCH_CSV_REL[@]} -gt 0 ]; then
        local LINK_COUNT=$(( ${#EPOCH_TXT_REL[@]} + ${#EPOCH_CSV_REL[@]} ))
        cat >> "$OUT" <<EOF
<div class="section-title">📁 Sample Files <span class="badge-green badge">${LINK_COUNT} files</span></div>
<p style="color:var(--muted);font-size:0.85rem;margin-bottom:0.8rem;">Raw epoch data files — click to open:</p>
<div class="file-link-grid">
EOF
        for f in "${EPOCH_TXT_REL[@]}"; do
            local clean="${f#./}"
            echo "  <a class=\"file-link\" href=\"${clean}\" target=\"_blank\">📝 $(basename "$clean")</a>" >> "$OUT"
        done
        for f in "${EPOCH_CSV_REL[@]}"; do
            local clean="${f#./}"
            echo "  <a class=\"file-link file-link-green\" href=\"${clean}\" target=\"_blank\">📊 $(basename "$clean")</a>" >> "$OUT"
        done
        echo '</div>' >> "$OUT"
    fi

    # --- OTHER CSV FILES (with preview) ---
    if [ ${#OTHER_CSV_REL[@]} -gt 0 ]; then
        cat >> "$OUT" <<EOF
<div class="section-title">📈 CSV Data <span class="badge-green badge">${#OTHER_CSV_REL[@]}</span></div>
EOF
        echo '<div class="file-grid">' >> "$OUT"
        for csv in "${OTHER_CSV_REL[@]}"; do
            local clean="${csv#./}"
            local preview
            preview=$(head -20 "${DIR}${clean}" 2>/dev/null | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')
            cat >> "$OUT" <<EOF
  <div class="file-card">
    <div class="file-header"><span class="dot dot-csv"></span>${clean}</div>
    <pre>${preview}</pre>
    <a class="file-open" href="${clean}" target="_blank">Open full file →</a>
  </div>
EOF
        done
        echo '</div>' >> "$OUT"
    fi

    # --- OTHER TXT FILES (with preview) ---
    if [ ${#OTHER_TXT_REL[@]} -gt 0 ]; then
        cat >> "$OUT" <<EOF
<div class="section-title">📝 Text Logs <span class="badge">${#OTHER_TXT_REL[@]}</span></div>
EOF
        echo '<div class="file-grid">' >> "$OUT"
        for txt in "${OTHER_TXT_REL[@]}"; do
            local clean="${txt#./}"
            local preview
            preview=$(head -30 "${DIR}${clean}" 2>/dev/null | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')
            cat >> "$OUT" <<EOF
  <div class="file-card">
    <div class="file-header"><span class="dot dot-txt"></span>${clean}</div>
    <pre>${preview}</pre>
    <a class="file-open" href="${clean}" target="_blank">Open full file →</a>
  </div>
EOF
        done
        echo '</div>' >> "$OUT"
    fi

    # --- PYTHON VISUALIZATION SCRIPTS (with preview) ---
    if [ ${#PY_REL[@]} -gt 0 ]; then
        cat >> "$OUT" <<EOF
<div class="section-title">🐍 Visualization Scripts <span class="badge-orange badge">${#PY_REL[@]}</span></div>
EOF
        echo '<div class="file-grid">' >> "$OUT"
        for py in "${PY_REL[@]}"; do
            local clean="${py#./}"
            local preview
            preview=$(head -40 "${DIR}${clean}" 2>/dev/null | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')
            cat >> "$OUT" <<EOF
  <div class="file-card">
    <div class="file-header"><span class="dot dot-py"></span>${clean}</div>
    <pre>${preview}</pre>
    <a class="file-open" href="${clean}" target="_blank">Open full file →</a>
  </div>
EOF
        done
        echo '</div>' >> "$OUT"
    fi

    # --- Footer ---
    cat >> "$OUT" <<EOF
<div class="footer">
  ${TIMESTAMP}
</div>
</div>
</body>
</html>
EOF

    echo "HTML Dashboard generiert: $OUT"
}

# -------------------------------------------------------
# Run all visualization scripts against the run folder
# and copy the .py source files into the run folder so
# they are included in the rsync / HTML dashboard.
# -------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VIZ_SCRIPTS=(
    "viz_fibre_bundle.py"
    "viz_persistent_homology.py"
    "viz_kelp_forest.py"
)

run_visualizations() {
    local DIR="$1"

    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  Running Python visualization scripts                   ║"
    echo "╚══════════════════════════════════════════════════════════╝"

    for viz in "${VIZ_SCRIPTS[@]}"; do
        local VIZ_PATH="${SCRIPT_DIR}/${viz}"

        if [ ! -f "$VIZ_PATH" ]; then
            echo "  ⚠ Visualization script not found: $VIZ_PATH — skipping."
            continue
        fi

        # --- Copy the .py source into the run folder so it gets synced ---
        local DEST_PY="${DIR}${viz}"
        if [ "$VIZ_PATH" != "$(realpath "$DEST_PY" 2>/dev/null)" ]; then
            cp -f "$VIZ_PATH" "$DEST_PY"
            echo "  📄 Copied ${viz} → ${DEST_PY}"
        fi

        # --- Execute the visualization script ---
        echo "  🔬 Running: python3 ${viz} ${DIR}"
        python3 "$VIZ_PATH" "${DIR}" 2>&1 | sed 's/^/     /'
        local EXIT_CODE=${PIPESTATUS[0]}

        if [ "$EXIT_CODE" -eq 0 ]; then
            echo "  ✅ ${viz} completed successfully."
        else
            echo "  ❌ ${viz} exited with code ${EXIT_CODE} — continuing."
        fi
        echo ""
    done
}

# --- Main Loop ---
echo "Überwache: $RUN_DIR"
echo "Intervall: ${INTERVAL}s"
[ -n "$COPY_TO" ] && echo "Sync zu:   $COPY_TO"
echo "---"

if [ -n "$COPY_TO" ]; then
    ensure_remote_dir
fi

while true; do
    # 1) Generate error plots (existing)
    bash plot_errors.sh "$RUN_DIR"

    bash plot_csv_extras.sh "$RUN_DIR"

    # 2) Run Python visualization scripts & copy .py files into run folder
    run_visualizations "$RUN_DIR"

    # 3) Generate the HTML dashboard (now picks up .py files + new .png images)
    generate_html "$RUN_DIR"

    generate_slideshow_html "$RUN_DIR"

    # 4) Find all syncable files (now includes .py)
    FILES=$(find "$RUN_DIR" -type f \( \
        -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.svg" \
        -o -iname "*.csv" -o -iname "*.txt" -o -iname "*.html" -o -iname "*.py" \
    \))

    if [ -n "$FILES" ]; then
        COUNT=$(echo "$FILES" | wc -l)
        echo "Gefundene Dateien ($COUNT)"

        if [ -n "$COPY_TO" ]; then
            echo "Synce nach $COPY_TO ..."
            rsync -avz \
                --include='*/' \
                --include='*.png' --include='*.jpg' --include='*.jpeg' --include='*.svg' \
                --include='*.csv' --include='*.txt' --include='*.html' \
                --exclude='*' \
                "${RUN_DIR}" "$COPY_TO"
            echo "Sync abgeschlossen."
        fi
    else
        echo "Keine Dateien gefunden in $RUN_DIR"
    fi

    echo "--- Warte ${INTERVAL}s ---"
    sleep "$INTERVAL"
done
