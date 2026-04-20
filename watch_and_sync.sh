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

    # 2. Alle Bilder im Run-Ordner finden (egal wo)
    IMAGES=$(find "$RUN_DIR" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.svg" \))

    if [ -n "$IMAGES" ]; then
        COUNT=$(echo "$IMAGES" | wc -l)
        echo ""
        echo "Gefundene Bilder ($COUNT):"
        echo "$IMAGES" | sed 's/^/  /'

        # 3. Optional: rsync zu Remote-Server
        if [ -n "$COPY_TO" ]; then
            # Zielverzeichnis mit Run-Ordnername erstellen
            RUN_NAME=$(basename "${RUN_DIR%/}")
            REMOTE_TARGET="${COPY_TO%/}/${RUN_NAME}/"

            echo ""
            echo "Synce $COUNT Bilder nach $REMOTE_TARGET ..."

            # Alle Bilder mit relativer Struktur hochladen
            rsync -az --relative --include='*/' --include='*.png' --include='*.jpg' --include='*.jpeg' --include='*.svg' --exclude='*' "$RUN_DIR" "$REMOTE_TARGET"

            echo "Sync abgeschlossen."
        fi
    else
        echo "Keine Bilder gefunden in $RUN_DIR"
    fi

    echo "--- Warte ${INTERVAL}s ---"
    sleep "$INTERVAL"
done
