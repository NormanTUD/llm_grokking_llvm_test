#!/bin/bash

# Verzeichnis als Argument übernehmen (Standard: aktuelles Verzeichnis)
DIR="${1:-.}"
# Trailing Slash sicherstellen
DIR="${DIR%/}/"

# 1. Daten extrahieren und relative Werte berechnen
data=$(cat "${DIR}"epoch_*.txt | awk -F': ' '
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

# 3. Filterung und Plotting mit Gnuplot
# Filter: wert > (median - 50) && wert < (median + 50)
echo "$data" | awk -v m="$median" '
{ if ($1 > (m - 50) && $1 < (m + 50)) print $1 }
' > "${DIR}filtered_data.dat"

gnuplot -p <<EOF
set title "Relative Abweichung (Gefiltert: Median ±50%)"
set ylabel "Abweichung in %"
set xlabel "Datenpunkt"
set grid
plot "${DIR}filtered_data.dat" with linespoints title "Rel. Error"
EOF
