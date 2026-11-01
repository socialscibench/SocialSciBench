#!/usr/bin/env bash
# Download the PatentsView tables used by the real-data workloads of task 001.
#
# Source: USPTO Open Data Portal, product PVGPATDIS (PatentsView Granted Patent
# Disambiguated Data). Requires a free USPTO.gov account and an ODP API key
# (data.uspto.gov -> Manage API Key). Save the key in a file, not in this script.
#
# usage: ODP_KEY_FILE=~/.uspto_odp_key ./download_patentsview.sh [out_dir]
set -euo pipefail

KEY_FILE="${ODP_KEY_FILE:-$HOME/.uspto_odp_key}"
OUT="${1:-$HOME/Data/patentsview}"
BASE="https://api.uspto.gov/api/v1/datasets/products/files/PVGPATDIS"

FILES=(
  g_patent.tsv.zip                      #   232 MB  grant date, patent type
  g_application.tsv.zip                 #    72 MB  application dates
  g_inventor_disambiguated.tsv.zip      #   708 MB  inventor ids (inventor x assignee network)
  g_assignee_disambiguated.tsv.zip      #   363 MB  firm ids
  g_location_disambiguated.tsv.zip      #     3 MB  location x year FE
  g_cpc_current.tsv.zip                 #   503 MB  technology class x year FE
  g_examiner_not_disambiguated.tsv.zip  #   199 MB  examiner FE
  g_us_patent_citation.tsv.zip          # 2,254 MB  forward citations, citing x cited firm dyads
  data_dictionary_PVGPATDIS.xlsx        #   0.2 MB
)

[ -r "$KEY_FILE" ] || { echo "API key file not found: $KEY_FILE" >&2; exit 1; }
mkdir -p "$OUT"
for f in "${FILES[@]}"; do
  if [ -s "$OUT/$f" ]; then echo "skip $f (exists)"; continue; fi
  echo "downloading $f"
  curl -fL --retry 3 -C - -H "X-API-KEY: $(cat "$KEY_FILE")" -o "$OUT/$f.part" "$BASE/$f"
  mv "$OUT/$f.part" "$OUT/$f"
done
ls -l "$OUT"
