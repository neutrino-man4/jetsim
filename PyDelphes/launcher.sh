#!/usr/bin/env bash
# Usage: bash launch_parallel.sh --config /path/to/config.json --jobs J --cores N

#set -euo pipefail

CONFIG=""
J=1
N=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --jobs)   J="$2";      shift 2 ;;
        --cores)  N="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "$CONFIG" ]] && { echo "ERROR: --config is required"; exit 1; }
[[ ! -f "$CONFIG" ]] && { echo "ERROR: config file not found: $CONFIG"; exit 1; }

TMPDIR_CONFIGS=$(mktemp -d)
trap 'rm -rf "$TMPDIR_CONFIGS"' EXIT

for i in $(seq 1 "$J"); do
    run_id=$(printf "%02d" "$i")
    tmp_cfg="${TMPDIR_CONFIGS}/config_run_${run_id}.json"
    sed "s/run_XX/run_${run_id}/g" "$CONFIG" > "$tmp_cfg"
    echo "$tmp_cfg"
done | xargs -P "$N" -I {} python3 gen_reco.py --config {} --no-verbose --randomize
