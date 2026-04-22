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

# --- Argumente parsen ---
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
# Generate a beautiful single-page HTML dashboard
# -------------------------------------------------------
generate_html() {
    local DIR="$1"
    local OUT="${DIR}index.html"
    local TIMESTAMP
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S %Z')

    # Collect files (relative paths)
    local IMAGES_REL=()
    local CSV_REL=()
    local TXT_REL=()

    while IFS= read -r f; do
        [ -n "$f" ] && IMAGES_REL+=("$f")
    done < <(cd "$DIR" && find . -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.svg" \) | sort)

    while IFS= read -r f; do
        [ -n "$f" ] && CSV_REL+=("$f")
    done < <(cd "$DIR" && find . -type f -iname "*.csv" | sort)

    while IFS= read -r f; do
        [ -n "$f" ] && TXT_REL+=("$f")
    done < <(cd "$DIR" && find . -type f -iname "*.txt" | sort)

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
    --bg: #0f1117;
    --card: #1a1d2e;
    --border: #2a2d3e;
    --accent: #7c5cfc;
    --accent2: #00d4aa;
    --text: #e2e4f0;
    --muted: #8888aa;
    --glow: rgba(124, 92, 252, 0.15);
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 0;
  }
  .hero {
    background: linear-gradient(135deg, #1a1040 0%, #0f1117 50%, #0a1a1a 100%);
    border-bottom: 1px solid var(--border);
    padding: 2.5rem 2rem 2rem;
    text-align: center;
  }
  .hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #7c5cfc, #00d4aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
  }
  .hero .subtitle {
    color: var(--muted);
    font-size: 0.95rem;
  }
  .hero .timestamp {
    display: inline-block;
    margin-top: 0.8rem;
    padding: 0.3rem 1rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 999px;
    font-size: 0.8rem;
    color: var(--accent2);
    letter-spacing: 0.03em;
  }
  .container { max-width: 1400px; margin: 0 auto; padding: 2rem 1.5rem; }

  /* Section titles */
  .section-title {
    font-size: 1.3rem;
    font-weight: 700;
    margin: 2.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .section-title .badge {
    background: var(--accent);
    color: #fff;
    font-size: 0.7rem;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-weight: 600;
  }

  /* Image grid */
  .image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
    gap: 1.2rem;
  }
  .image-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .image-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px var(--glow);
  }
  .image-card img {
    width: 100%;
    display: block;
    background: #000;
  }
  .image-card .label {
    padding: 0.6rem 1rem;
    font-size: 0.8rem;
    color: var(--muted);
    font-family: 'Courier New', monospace;
    border-top: 1px solid var(--border);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* File cards (CSV / TXT) */
  .file-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 1rem;
  }
  .file-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .file-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px var(--glow);
  }
  .file-card .file-header {
    padding: 0.6rem 1rem;
    font-size: 0.85rem;
    font-weight: 600;
    background: linear-gradient(90deg, rgba(124,92,252,0.08), transparent);
    border-bottom: 1px solid var(--border);
    font-family: 'Courier New', monospace;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  .file-card .file-header .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
  }
  .dot-csv { background: var(--accent2); }
  .dot-txt { background: var(--accent); }
  .file-card pre {
    padding: 0.8rem 1rem;
    font-size: 0.75rem;
    line-height: 1.5;
    overflow-x: auto;
    max-height: 300px;
    overflow-y: auto;
    color: #ccc;
    background: #12141f;
  }
  .file-card a {
    display: block;
    padding: 0.7rem 1rem;
    text-align: center;
    color: var(--accent);
    text-decoration: none;
    font-size: 0.85rem;
    font-weight: 600;
  }
  .file-card a:hover { text-decoration: underline; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  .empty-state {
    color: var(--muted);
    font-style: italic;
    padding: 1rem 0;
  }

  @media (max-width: 600px) {
    .image-grid { grid-template-columns: 1fr; }
    .file-grid { grid-template-columns: 1fr; }
    .hero h1 { font-size: 1.6rem; }
  }
</style>
</head>
<body>
HEADER

    # Hero section
    cat >> "$OUT" <<EOF
<div class="hero">
  <h1>🧠 Grokking Dashboard</h1>
  <div class="subtitle">Live training monitor &middot; auto-refreshes every 15s</div>
  <div class="timestamp">⏱ Last updated: ${TIMESTAMP}</div>
</div>
<div class="container">
EOF

    # --- PLOTS / IMAGES ---
    cat >> "$OUT" <<EOF
<div class="section-title">📊 Plots &amp; Visualizations <span class="badge">${#IMAGES_REL[@]}</span></div>
EOF
    if [ ${#IMAGES_REL[@]} -gt 0 ]; then
        echo '<div class="image-grid">' >> "$OUT"
        for img in "${IMAGES_REL[@]}"; do
            local clean="${img#./}"
            local cache_bust="?t=$(date +%s)"
            cat >> "$OUT" <<EOF
  <div class="image-card">
    <a href="${clean}${cache_bust}" target="_blank"><img src="${clean}${cache_bust}" alt="${clean}" loading="lazy"></a>
    <div class="label">${clean}</div>
  </div>
EOF
        done
        echo '</div>' >> "$OUT"
    else
        echo '<div class="empty-state">No images found yet.</div>' >> "$OUT"
    fi

    # --- CSV FILES ---
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

    # --- TXT FILES ---
    cat >> "$OUT" <<EOF
<div class="section-title">📝 Text Logs <span class="badge">${#TXT_REL[@]}</span></div>
EOF
    if [ ${#TXT_REL[@]} -gt 0 ]; then
        echo '<div class="file-grid">' >> "$OUT"
        for txt in "${TXT_REL[@]}"; do
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
    else
        echo '<div class="empty-state">No text files found yet.</div>' >> "$OUT"
    fi

    # Footer
    cat >> "$OUT" <<'FOOTER'
</div>
</body>
</html>
FOOTER

    echo "HTML Dashboard generiert: $OUT"
}

# -------------------------------------------------------
# Parse remote user@host and path from COPY_TO
# -------------------------------------------------------
parse_remote() {
    # e.g. root@blabla.de:/var/www/grok_test/
    REMOTE_USERHOST="${COPY_TO%%:*}"
    REMOTE_PATH="${COPY_TO#*:}"
    REMOTE_PATH="${REMOTE_PATH%/}/"
}

# -------------------------------------------------------
# Ensure remote directory exists
# -------------------------------------------------------
ensure_remote_dir() {
    if [ -n "$COPY_TO" ]; then
        parse_remote
        echo "Stelle sicher, dass Remote-Verzeichnis existiert: $REMOTE_PATH"
        ssh "$REMOTE_USERHOST" "mkdir -p '$REMOTE_PATH'"
    fi
}

# --- Hauptschleife ---
echo "Überwache: $RUN_DIR"
echo "Intervall: ${INTERVAL}s"
[ -n "$COPY_TO" ] && echo "Sync zu:   $COPY_TO"
echo "---"

# Create remote dir once at startup
if [ -n "$COPY_TO" ]; then
    ensure_remote_dir
fi

while true; do
    # 1. Plots neu generieren
    bash plot_errors.sh "$RUN_DIR"

    # 2. HTML Dashboard generieren
    generate_html "$RUN_DIR"

    # 3. Alle relevanten Dateien finden
    FILES=$(find "$RUN_DIR" -type f \( \
        -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.svg" \
        -o -iname "*.csv" -o -iname "*.txt" -o -iname "*.html" \
    \))

    if [ -n "$FILES" ]; then
        COUNT=$(echo "$FILES" | wc -l)
        echo ""
        echo "Gefundene Dateien ($COUNT):"
        echo "$FILES" | sed 's/^/  /'

        # 4. Optional: rsync zu Remote-Server
        if [ -n "$COPY_TO" ]; then
            echo ""
            echo "Synce $COUNT Dateien nach $COPY_TO ..."

            # Sync contents of RUN_DIR directly into remote target (flat, no nesting)
            rsync -az \
                --include='*/' \
                --include='*.png' \
                --include='*.jpg' \
                --include='*.jpeg' \
                --include='*.svg' \
                --include='*.csv' \
                --include='*.txt' \
                --include='*.html' \
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
