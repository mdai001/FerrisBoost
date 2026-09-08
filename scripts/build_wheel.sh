#!/bin/bash
# Build the release wheel.
#
# Two things this does that a bare `maturin build` does not:
#
#   1. --profile dist  : release keeps `debug = true` for nsys/ncu/perf, and those
#      symbols embed the build machine's absolute paths. An unstripped wheel
#      contained 630 occurrences of /home/<user> and a 131 MB .so.
#
#   2. --remap-path-prefix : strips anything left that names the checkout or the
#      user's home, so the artifact does not carry a build-machine signature.
#
# ⚠️ --features REPLACES the list in pyproject.toml rather than appending, so all
# three must be named. Omitting `python` drops pyo3 from the artifact, maturin
# then falls back to cffi, and the result is a 22-byte broken wheel whose error
# message ("No module named 'cffi'") is nowhere near the cause.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${FB_PY:-python3}
WHEEL_OUT=${FB_WHEEL_OUT:-target/wheels}
export RUSTFLAGS="${RUSTFLAGS:-} --remap-path-prefix=$HOME=/build --remap-path-prefix=$PWD=/src"
"${MATURIN:-maturin}" build \
  --profile dist \
  --features "cuda,python,pyo3/extension-module" \
  --strip \
  -i "$PY" \
  -o "$WHEEL_OUT" \
  "$@"

# maturin writes the absolute build directory into the SBOM's bom-ref fields.
# RUSTFLAGS remapping cannot reach that, because maturin generates it rather
# than rustc, so it is rewritten after the fact and RECORD is regenerated.
WHEEL=$(ls -t "$WHEEL_OUT"/*.whl | head -1)
"$PY" scripts/sanitize_wheel.py "$WHEEL"
"$PY" scripts/sanitize_wheel.py "$WHEEL" --check

# Stable wheel tags cannot express x86-64 microarchitecture levels.  An
# optimized build therefore uses a separate distribution identity while still
# installing the exact same `ferrisboost` import package and native module.
# The default is deliberately unchanged: `pip install ferrisboost` receives
# the generic build.
if [[ -n "${FB_DIST_NAME:-}" && "${FB_DIST_NAME}" != "ferrisboost" ]]; then
  WHEEL=$("$PY" scripts/rebrand_wheel.py "$WHEEL" "$FB_DIST_NAME")
fi
echo "$WHEEL"
