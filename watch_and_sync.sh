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

# --- Hauptschleife ---
echo "Überwache: $RUN_DIR"
echo "Intervall: ${INTERVAL}s"
[ -n "$COPY_TO" ] && echo "Sync zu:   $COPY_TO"
echo "---"

while true; do
    # 1. Plots neu generieren
    bash plot_errors.sh "$RUN_DIR"

    # 2. Alle relevanten Dateien im Run-Ordner finden (Bilder + csv + txt + html)
    FILES=$(find "$RUN_DIR" -type f \( \
        -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.svg" \
        -o -iname "*.csv" -o -iname "*.txt" -o -iname "*.html" \
    \))

    if [ -n "$FILES" ]; then
        COUNT=$(echo "$FILES" | wc -l)
        echo ""
        echo "Gefundene Dateien ($COUNT):"
        echo "$FILES" | sed 's/^/  /'

        # 3. Optional: rsync zu Remote-Server
        if [ -n "$COPY_TO" ]; then
            REMOTE_TARGET="${COPY_TO%/}/"

            echo ""
            echo "Synce $COUNT Dateien nach $REMOTE_TARGET ..."

            # Sync contents of RUN_DIR directly into REMOTE_TARGET (no extra subdirectory)
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
                "${RUN_DIR}" "$REMOTE_TARGET"

            echo "Sync abgeschlossen."
        fi
    else
        echo "Keine Dateien gefunden in $RUN_DIR"
    fi

    echo "--- Warte ${INTERVAL}s ---"
    sleep "$INTERVAL"
done
