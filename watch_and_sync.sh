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
# Use awk to parse ALL epoch .txt files at once and
# produce both the overview JSON and detail HTML
# -------------------------------------------------------
generate_epoch_html() {
    local DIR="$1"
    local OVERVIEW_FILE="$2"
    local DETAILS_FILE="$3"

    # Find all epoch txt files
    local EPOCH_FILES
    EPOCH_FILES=$(cd "$DIR" && find . -type f -name 'epoch_*.txt' | sort)

    if [ -z "$EPOCH_FILES" ]; then
        echo '<div class="empty-state">No epoch sample files found yet.</div>' > "$OVERVIEW_FILE"
        echo "" > "$DETAILS_FILE"
        return
    fi

    # Build full paths
    local FULL_PATHS=""
    while IFS= read -r f; do
        FULL_PATHS="${FULL_PATHS} ${DIR}${f#./}"
    done <<< "$EPOCH_FILES"

    # Single awk pass over all files to produce overview + details
    awk '
    BEGIN {
        epoch = 0; sample = 0; correct_count = 0; total = 0;
        params = ""; expected = ""; predicted = ""; is_correct = "";
        current_file = "";
    }

    FNR == 1 {
        # Finish previous epoch if any
        if (current_file != "" && total > 0) {
            pct = int((correct_count * 100) / total)
            overview[epoch_idx] = epoch "\t" correct_count "\t" total "\t" pct
            epoch_idx++
        }
        # Extract epoch number from filename
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
                icon = "✅"
            } else {
                cls = "row-wrong"
                icon = "❌"
            }
            detail_rows[epoch] = detail_rows[epoch] \
                "<tr class=\"" cls "\"><td>" sample "</td><td>" params "</td><td>" expected "</td><td>" predicted "</td><td>" icon "</td></tr>\n"
            params = ""; expected = ""; predicted = ""; is_correct = ""
        }
    }

    END {
        # Finish last epoch
        if (total > 0) {
            pct = int((correct_count * 100) / total)
            overview[epoch_idx] = epoch "\t" correct_count "\t" total "\t" pct
            epoch_idx++
        }

        # Print overview table
        print "<table class=\"epoch-overview\">"
        print "<thead><tr><th>Epoch</th><th>Correct</th><th>Total</th><th>Accuracy</th><th style=\"min-width:140px\"></th></tr></thead>"
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
        print "</tbody></table>"
        print "===SPLIT==="

        # Print collapsible details
        print "<h3 style=\"color:var(--muted);margin:1.5rem 0 0.8rem;font-size:0.95rem;\">Click an epoch to see sample details:</h3>"
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

    # Split the output
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

    # Collect files
    local IMAGES_REL=()
    local CSV_REL=()
    local OTHER_TXT_REL=()
    local EPOCH_COUNT=0

    while IFS= read -r f; do
        [ -n "$f" ] && IMAGES_REL+=("$f")
    done < <(cd "$DIR" && find . -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.svg" \) | sort)

    while IFS= read -r f; do
        [ -n "$f" ] && CSV_REL+=("$f")
    done < <(cd "$DIR" && find . -type f -iname "*.csv" | sort)

    while IFS= read -r f; do
        if [ -n "$f" ]; then
            local base
            base=$(basename "$f")
            if [[ "$base" =~ ^epoch_[0-9]+\.txt$ ]]; then
                EPOCH_COUNT=$((EPOCH_COUNT + 1))
            else
                OTHER_TXT_REL+=("$f")
            fi
        fi
    done < <(cd "$DIR" && find . -type f -iname "*.txt" | sort)

    # Parse epochs with awk (fast!)
    local OVERVIEW_TMP="/tmp/_epoch_overview.html"
    local DETAILS_TMP="/tmp/_epoch_details.html"
    generate_epoch_html "$DIR" "$OVERVIEW_TMP" "$DETAILS_TMP"

    # --- Write HTML ---
    cat > "$OUT" <<'HEADER'
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="15">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🧠 Grokking Dashboard</title>
<style>
  :root {
    --bg: #0f1117; --card: #1a1d2e; --border: #2a2d3e;
    --accent: #7c5cfc; --accent2: #00d4aa;
    --text: #e2e4f0; --muted: #8888aa;
    --glow: rgba(124, 92, 252, 0.15);
    --green: #00d4aa; --red: #ff5c72;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
  .hero { background: linear-gradient(135deg, #1a1040 0%, #0f1117 50%, #0a1a1a 100%); border-bottom: 1px solid var(--border); padding: 2.5rem 2rem 2rem; text-align: center; }
  .hero h1 { font-size: 2.4rem; font-weight: 800; background: linear-gradient(135deg, #7c5cfc, #00d4aa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .hero .subtitle { color: var(--muted); font-size: 0.95rem; }
  .hero .timestamp { display: inline-block; margin-top: 0.8rem; padding: 0.3rem 1rem; background: var(--card); border: 1px solid var(--border); border-radius: 999px; font-size: 0.8rem; color: var(--accent2); }
  .container { max-width: 1400px; margin: 0 auto; padding: 2rem 1.5rem; }
  .section-title { font-size: 1.3rem; font-weight: 700; margin: 2.5rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid var(--border); display: flex; align-items: center; gap: 0.5rem; }
  .section-title .badge { background: var(--accent); color: #fff; font-size: 0.7rem; padding: 0.15rem 0.55rem; border-radius: 999px; font-weight: 600; }
  .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 1.2rem; }
  .image-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; transition: transform 0.2s, box-shadow 0.2s; }
  .image-card:hover { transform: translateY(-3px); box-shadow: 0 8px 30px var(--glow); }
  .image-card img { width: 100%; display: block; background: #000; }
  .image-card .label { padding: 0.6rem 1rem; font-size: 0.8rem; color: var(--muted); font-family: 'Courier New', monospace; border-top: 1px solid var(--border); }
  .epoch-overview { width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; }
  .epoch-overview th { background: #151728; text-align: left; padding: 0.6rem 1rem; font-size: 0.8rem; color: var(--accent); border-bottom: 2px solid var(--border); position: sticky; top: 0; }
  .epoch-overview td { padding: 0.45rem 1rem; font-size: 0.82rem; border-bottom: 1px solid var(--border); font-family: 'Courier New', monospace; }
  .epoch-overview tr:hover td { background: rgba(124,92,252,0.05); }
  .acc-good { color: var(--green); font-weight: 700; }
  .acc-mid { color: #f0c040; font-weight: 700; }
  .acc-bad { color: var(--red); font-weight: 700; }
  .acc-bar-bg { width: 120px; height: 10px; background: #222; border-radius: 5px; overflow: hidden; display: inline-block; vertical-align: middle; margin-right: 0.5rem; }
  .acc-bar-fg { height: 100%; border-radius: 5px; }
  details { margin-bottom: 0.5rem; }
  details summary { cursor: pointer; padding: 0.5rem 1rem; background: var(--card); border: 1px solid var(--border); border-radius: 8px; font-size: 0.85rem; font-family: 'Courier New', monospace; display: flex; align-items: center; gap: 0.8rem; transition: background 0.15s; }
  details summary:hover { background: #1f2238; }
  details[open] summary { border-radius: 8px 8px 0 0; border-bottom: none; }
  details .detail-body { background: var(--card); border: 1px solid var(--border); border-top: none; border-radius: 0 0 8px 8px; overflow-x: auto; }
  .sample-table { width: 100%; border-collapse: collapse; }
  .sample-table th { background: #151728; text-align: left; padding: 0.4rem 0.8rem; font-size: 0.75rem; color: var(--muted); border-bottom: 1px solid var(--border); }
  .sample-table td { padding: 0.35rem 0.8rem; font-size: 0.78rem; border-bottom: 1px solid rgba(42,45,62,0.5); font-family: 'Courier New', monospace; }
  .row-correct td { color: var(--green); }
  .row-wrong td { color: var(--red); }
  .file-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 1rem; }
  .file-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; transition: transform 0.2s, box-shadow 0.2s; }
  .file-card:hover { transform: translateY(-2px); box-shadow: 0 6px 24px var(--glow); }
  .file-card .file-header { padding: 0.6rem 1rem; font-size: 0.85rem; font-weight: 600; background: linear-gradient(90deg, rgba(124,92,252,0.08), transparent); border-bottom: 1px solid var(--border); font-family: 'Courier New', monospace; display: flex; align-items: center; gap: 0.4rem; }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .dot-csv { background: var(--accent2); }
  .dot-txt { background: var(--accent); }
  .file-card pre { padding: 0.8rem 1rem; font-size: 0.75rem; line-height: 1.5; overflow-x: auto; max-height: 300px; overflow-y: auto; color: #ccc; background: #12141f; }
  .file-card a { display: block; padding: 0.7rem 1rem; text-align: center; color: var(--accent); text-decoration: none; font-size: 0.85rem; font-weight: 600; }
  .file-card a:hover { text-decoration: underline; }
  .empty-state { color: var(--muted); font-style: italic; padding: 1rem 0; }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  @media (max-width: 600px) { .image-grid, .file-grid { grid-template-columns: 1fr; } .hero h1 { font-size: 1.6rem; } }
</style>
</head>
<body>
HEADER

    cat >> "$OUT" <<EOF
<div class="hero">
  <h1>🧠 Grokking Dashboard</h1>
  <div class="subtitle">Live training monitor &middot; auto-refreshes every 15s</div>
  <div class="timestamp">⏱ Last updated: ${TIMESTAMP}</div>
</div>
<div class="container">
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

    # --- EPOCH SAMPLES ---
    cat >> "$OUT" <<EOF
<div class="section-title">🧪 Epoch Samples <span class="badge">${EPOCH_COUNT} epochs</span></div>
EOF
    cat "$OVERVIEW_TMP" >> "$OUT"
    cat "$DETAILS_TMP" >> "$OUT"

    # --- CSV ---
    cat >> "$OUT" <<EOF
<div class="section-title">📄 CSV Data <span class="badge">${#CSV_REL[@]}</span></div>
EOF
    if [ ${#CSV_REL[@]} -gt 0 ]; then
        echo '<div class="file-grid">' >> "$OUT"
        for csv in "${CSV_REL[@]}"; do
            local clean="${csv#./}"
            local preview
            preview=$(head -20 "${DIR}${clean}" 2>/dev/null | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')
            cat >> "$OUT" <<EOF
  <div class="file-card">
    <div class="file-header"><span class="dot dot-csv"></span>${clean}</div>
    <pre>${preview}</pre>
    <a href="${clean}" target="_blank">Open full file ↗</a>
  </div>
EOF
        done
        echo '</div>' >> "$OUT"
    else
        echo '<div class="empty-state">No CSV files found yet.</div>' >> "$OUT"
    fi

    # --- OTHER TXT ---
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
    <a href="${clean}" target="_blank">Open full file ↗</a>
  </div>
EOF
        done
        echo '</div>' >> "$OUT"
    fi

    echo '</div></body></html>' >> "$OUT"
    echo "HTML Dashboard generiert: $OUT"
}

# --- Main ---
echo "Überwache: $RUN_DIR"
echo "Intervall: ${INTERVAL}s"
[ -n "$COPY_TO" ] && echo "Sync zu:   $COPY_TO"
echo "---"

if [ -n "$COPY_TO" ]; then
    ensure_remote_dir
fi

while true; do
    bash plot_errors.sh "$RUN_DIR"

    generate_html "$RUN_DIR"

    FILES=$(find "$RUN_DIR" -type f \( \
        -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.svg" \
        -o -iname "*.csv" -o -iname "*.txt" -o -iname "*.html" \
    \))

    if [ -n "$FILES" ]; then
        COUNT=$(echo "$FILES" | wc -l)
        echo "Gefundene Dateien ($COUNT)"

        if [ -n "$COPY_TO" ]; then
            echo "Synce nach $COPY_TO ..."
            rsync -az \
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
