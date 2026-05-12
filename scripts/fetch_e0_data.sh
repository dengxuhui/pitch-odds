#!/usr/bin/env bash

set -euo pipefail

OUTPUT_DIR="data/samples/e0"
BASE_URL="https://www.football-data.co.uk/mmz4281"

mkdir -p "$OUTPUT_DIR"

SEASONS=(1819 1920 2021 2122 2223 2324 2425)

for season in "${SEASONS[@]}"; do
  target="$OUTPUT_DIR/E0_${season}.csv"
  url="$BASE_URL/${season}/E0.csv"
  echo "downloading ${url} -> ${target}"
  curl -k --retry 5 --retry-delay 2 -fsSL "$url" -o "$target"
done

echo "done: downloaded ${#SEASONS[@]} E0 seasons to ${OUTPUT_DIR}"
