#!/bin/bash

# Verzeichnis als Argument übernehmen (Standard: aktuelles Verzeichnis)
DIR="${1:-.}"
# Trailing Slash sicherstellen
DIR="${DIR%/}/"

# Samples-Unterverzeichnis
SAMPLES_DIR="${DIR}samples/"

# Plot-Ausgabeverzeichnis erstellen
PLOT_DIR="${DIR}plots/"
mkdir -p "$PLOT_DIR"

# Prüfen ob das Verzeichnis existiert
if [ ! -d "$SAMPLES_DIR" ]; then
    echo "Fehler: Verzeichnis '$SAMPLES_DIR' nicht gefunden!"
    echo "Erwartete Struktur: <run_folder>/samples/epoch_*.txt"
    exit 1
fi

# 1. Daten extrahieren und relative Werte berechnen
data=$(cat "${SAMPLES_DIR}"epoch_*.txt | awk -F': ' '
/expected/ {e=$2}
/predicted/ {
    p=$2;
    if (e != 0 && e != "") {
        print ((p - e) / (e < 0 ? -e : e)) * 100
    }
}')

# 2. Median berechnen
median=$(echo "$data" | sort -n | awk '
{ a[NR]=$1 }
END {
    if (NR % 2) {
        print a[(NR + 1) / 2]
    } else {
        print (a[(NR / 2)] + a[(NR / 2) + 1]) / 2
    }
}')

echo "Berechneter Median: $median%"

# 3. Filterung
echo "$data" | awk -v m="$median" '
{ if ($1 > (m - 50) && $1 < (m + 50)) print $1 }
' > "${DIR}filtered_data.dat"

# Alle Daten auch speichern (für ungefiltertes Plot)
echo "$data" > "${DIR}all_data.dat"

# 4. Dateinamen definieren
PLOT_FILTERED="${PLOT_DIR}relative_error_filtered.png"
PLOT_ALL="${PLOT_DIR}relative_error_all.png"

# 5. Plotting mit Gnuplot als PNG-Dateien
gnuplot <<EOF
set terminal pngcairo size 1200,600 enhanced font "Arial,12"

# Plot 1: Gefilterte Daten
set output "${PLOT_FILTERED}"
set title "Relative Abweichung (Gefiltert: Median ±50%)"
set ylabel "Abweichung in %"
set xlabel "Datenpunkt"
set grid
plot "${DIR}filtered_data.dat" with linespoints title "Rel. Error (filtered)"

# Plot 2: Alle Daten (ungefiltert)
set output "${PLOT_ALL}"
set title "Relative Abweichung (Alle Datenpunkte)"
set ylabel "Abweichung in %"
set xlabel "Datenpunkt"
set grid
plot "${DIR}all_data.dat" with linespoints title "Rel. Error (all)"
EOF

# 6. Ergebnis ausgeben
echo ""
echo "Plots gespeichert:"
echo "  $PLOT_FILTERED"
echo "  $PLOT_ALL"
