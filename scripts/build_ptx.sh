#!/bin/bash
# Regenerate the checked-in PTX.
#
# ⚠️ **The PTX is compiled into the shipped binary via include_str!, so anything
# nvcc writes into it ends up in the wheel.** `-lineinfo` emits a `.file`
# directive naming the .cu, and nvcc canonicalises that to an absolute path —
# which put the build machine's directory layout inside every released wheel.
# The directive is debug metadata only, so it is rewritten to a repo-relative
# path afterwards; ncu still maps line numbers.
#
# The arch is part of the contract, not an incidental flag: sm_75 keeps Turing
# and newer working. Building for the local card instead would silently raise
# the minimum supported GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
# Resolve a bare `NVCC=nvcc` to its real path: the leak check below matches the
# toolkit directory, and `dirname nvcc` would be `.`.
NVCC=$(command -v "$NVCC") || { echo "ERROR: nvcc not found: ${NVCC}" >&2; exit 1; }
SRC=src/backend/cuda/histogram.cu
OUT=src/backend/cuda/histogram.ptx
EXPECTED_PTX_VERSION=8.0
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

# nvcc preprocesses with the host C++ compiler even for --ptx, and the CUDA
# release that produces the required PTX ISA accepts only older GCC releases.
# NVCC_CCBIN selects that host compiler without hard-coding a machine path.
CCBIN_ARGS=()
if [ -n "${NVCC_CCBIN:-}" ]; then
  CCBIN_ARGS=(-ccbin "$NVCC_CCBIN")
fi

"$NVCC" "${CCBIN_ARGS[@]}" --ptx -std=c++14 -arch=compute_75 -lineinfo "$SRC" -o "$TMP"

# Relativise .file paths, drop the nvcc banner, embed the source hash.
python3 scripts/sanitize_ptx.py "$TMP" "$SRC" "$EXPECTED_PTX_VERSION"

# Fixed strings, not a regex: paths may contain `.` or other metacharacters.
LEAK=(-e "$HOME" -e "$(pwd)" -e /tmp/ -e /usr/local/cuda -e "$(dirname "$(dirname "$NVCC")")")
if grep -qF "${LEAK[@]}" "$TMP"; then
  echo "ERROR: generated PTX still contains build-machine paths" >&2
  grep -nF "${LEAK[@]}" "$TMP" | head -3 >&2
  exit 1
fi
mv "$TMP" "$OUT"
trap - EXIT
echo "  $OUT regenerated and sanitised"
