#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://127.0.0.1:8000"
SAMPLES=(
  "samples/core_high.json"
  "samples/enhanced_high.json"
  "samples/core_low.json"
  "samples/unknown_sex.json"
)

for sample in "${SAMPLES[@]}"; do
  echo "==> POST ${sample}"
  curl -s -X POST "${BASE_URL}/predict" \
    -H "Content-Type: application/json" \
    --data @"${sample}" | python -m json.tool
  echo
  echo
  done

echo "==> POST samples/invalid_out_of_range.json (show headers)"
curl -i -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  --data @"samples/invalid_out_of_range.json"
