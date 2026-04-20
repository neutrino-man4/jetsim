#!/usr/bin/env bash
# =============================================================================
# run_ntuplizer.sh
#
# Launch J parallel instances of the makeNtuples.C ROOT macro, one per run
# directory (run_01_decayed_1 ... run_JJ_decayed_1), using at most N cores
# via xargs.
#
# Each job gets its own temporary working directory under
# /tmp/abal/ntuplizer_runs/run_XX/ so that ROOT's compilation artefacts
# (.so, .pcm, .d files) do not collide across parallel workers.
# Temporary directories are removed on exit (including on SIGINT/SIGTERM).
#
# Usage:
#   ./run_ntuplizer.sh <J> <N> <path/to/makeNtuples.C>
#
#   J  – total number of runs to process (01 … JJ, zero-padded to 2 digits)
#   N  – maximum number of parallel worker processes
#   makeNtuples.C – path to the ROOT macro source file
#
# Author: Aritra Bal (ETP)
# Date  : 2026-02-23
# =============================================================================

# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------
if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <J: number of runs> <N: max cores> <path/to/makeNtuples.C> <seed(qcd_dijet, ttbar, etc)> " >&2
    exit 1
fi

J="${1}"
N="${2}"
MACRO_SRC="$(realpath "${3}")"
SEED="$4"
if [[ ! -f "${MACRO_SRC}" ]]; then
    echo "ERROR: Macro not found: ${MACRO_SRC}" >&2
    exit 1
fi

if ! [[ "${J}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: J must be a positive integer, got '${J}'" >&2
    exit 1
fi

if ! [[ "${N}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: N must be a positive integer, got '${N}'" >&2
    exit 1
fi
RUN_STR="run_XX"
# if TTBar or WToQQ is in the seed then add a decayed_1 suffix to the run directories, otherwise leave it out (for QCD)
if [[ "${SEED}" == *"TTBar"* ]] || [[ "${SEED}" == *"WToQQ"* ]]; then
    RUN_STR="run_XX_decayed_1"
fi
# ---------------------------------------------------------------------------
# Path templates
# ---------------------------------------------------------------------------
INPUT_TEMPLATE="/ceph/abal/QFIT/MC/ROOTFILES/${SEED}/${RUN_STR}/delphes.root" ## Remove the _decayed_1 suffix when running for QCD. TODO: FIX at some point
OUTPUT_BASE="/ceph/abal/QFIT/MC/nTuples/${SEED}"
TMP_BASE="/tmp/abal/ntuplizer_runs"

echo "[INFO] Configuration:"
echo "       J = ${J}"
echo "       N = ${N}"
echo "       Macro = ${MACRO_SRC}"
echo "       Input template = ${INPUT_TEMPLATE}"
echo "       Output base   = ${OUTPUT_BASE}"
echo "############################################################################"

# ---------------------------------------------------------------------------
# Cleanup: remove all per-run temp dirs on exit
# ---------------------------------------------------------------------------
cleanup() {
    echo "[INFO] Cleaning up temporary directories under ${TMP_BASE} ..."
    rm -rf "${TMP_BASE}"
    echo "[INFO] Done."
}
trap cleanup EXIT SIGINT SIGTERM

# ---------------------------------------------------------------------------
# Create the top-level tmp directory
# ---------------------------------------------------------------------------
mkdir -p "${TMP_BASE}"

# ---------------------------------------------------------------------------
# Worker function executed by each xargs process
# ---------------------------------------------------------------------------
# xargs passes a single argument: the zero-padded run index (e.g. "03")
run_one() {
    local xx="${1}"                  # e.g. "03"
    local run_dir="run_${xx}"        # e.g. "run_03"
    # If the seed contains "TTBar" or "WToQQ", we need to add the "_decayed_1" suffix to match the input template
    if [[ "${SEED}" == *"TTBar"* ]] || [[ "${SEED}" == *"WToQQ"* ]]; then
        run_dir="run_${xx}_decayed_1"
    fi
    local tmp_dir="${TMP_BASE}/${run_dir}"
    local input_file="${INPUT_TEMPLATE/XX/${xx}}"
    local output_dir="${OUTPUT_BASE}/${run_dir}"
    local output_file="${output_dir}/output.root"
    local macro_copy="${tmp_dir}/makeNtuples.C"

    # ---- Validate input ----------------------------------------------------
    if [[ ! -f "${input_file}" ]]; then
        echo "[ERROR] run_${xx}: input file not found: ${input_file}" >&2
        return 1
    fi

    # ---- Set up isolated working directory ---------------------------------
    mkdir -p "${tmp_dir}"
    mkdir -p "${output_dir}"

    # Copy macro into the temp dir so ROOT writes its .so/.d/.pcm there
    cp "${MACRO_SRC}" "${macro_copy}"
    echo "[INFO] run_${xx}: Reading input from ${input_file}"
    echo "[INFO]  run_${xx}: Will write to ${output_file}"

    # ---- Run ROOT in the isolated temp dir ---------------------------------
    # 'cd' is done in a subshell so it does not affect the parent process.
    (
        cd "${tmp_dir}"
        root -l -b -q \
            "makeNtuples.C+(\"${input_file}\", \"${output_file}\", \"FatJet\")" \
            > "${tmp_dir}/root_stdout.log"
    )
    local exit_code=$?

    if [[ ${exit_code} -ne 0 ]]; then
        echo "[ERROR] run_${xx}: ROOT exited with code ${exit_code}." \
             "See ${tmp_dir}/root_stderr.log" >&2
        return "${exit_code}"
    fi
    echo "[INFO] run_${xx}: Delphes output at ${input_file} processed successfully using ${macro_copy}."
    echo "[INFO]  run_${xx}: finished  →  ${output_file}"
}

# Export the function and variables so the bash subprocess spawned by
# xargs can see them.
export -f run_one
export INPUT_TEMPLATE OUTPUT_BASE TMP_BASE MACRO_SRC SEED

# ---------------------------------------------------------------------------
# Build the list of zero-padded run indices and dispatch via xargs
# ---------------------------------------------------------------------------
echo "[INFO] Launching ${J} job(s) with up to ${N} parallel worker(s)."
echo "[INFO] Macro : ${MACRO_SRC}"
echo "[INFO] Tmp   : ${TMP_BASE}"
echo ""

# seq produces 1 2 … J; printf zero-pads each to 2 digits
seq -f "%02g" 1 "${J}" \
    | xargs -P "${N}" -I {XX} bash -c 'run_one "{XX}"'

echo ""
echo "[INFO] All ${J} job(s) completed."