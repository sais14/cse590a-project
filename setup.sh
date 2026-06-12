#!/usr/bin/env bash
set -euo pipefail

SYNTHEA_URL="https://raw.githubusercontent.com/synthetichealth/synthea-sample-data/main/downloads/synthea_sample_data_csv_apr2020.zip"
ZIP="synthea_sample.zip"
CSV_DIR="synthea_data/csv"
DB="data/synthea.db"
NOTES_DIR="data/notes"

PY="${PYTHON:-python3}"

mkdir -p data

if [ -f "$DB" ]; then
  echo "[skip] $DB already exists — skipping download + DB build."
else
  if [ ! -d "$CSV_DIR" ] || [ -z "$(ls -A "$CSV_DIR"/*.csv 2>/dev/null)" ]; then
    if [ ! -f "$ZIP" ]; then
      echo "[download] fetching Synthea sample data..."
      curl -fSL "$SYNTHEA_URL" -o "$ZIP"
    fi
    echo "[extract] unzipping into synthea_data/ ..."
    mkdir -p synthea_data
    unzip -oq "$ZIP" -d synthea_data
  else
    echo "[skip] $CSV_DIR already populated — skipping download/extract."
  fi
  echo "[build] running build_db.py ..."
  "$PY" build_db.py
fi

if [ -d "$NOTES_DIR" ] && [ -n "$(ls -A "$NOTES_DIR"/note_*.txt 2>/dev/null)" ]; then
  echo "[skip] $NOTES_DIR already populated — skipping note generation."
else
  echo "[generate] running generate_notes.py ..."
  "$PY" generate_notes.py
fi

echo "[done] data pipeline ready."
