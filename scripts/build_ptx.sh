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
SRC=src/backend/cuda/histogram.cu
OUT=src/backend/cuda/histogram.ptx
EXPECTED_PTX_VERSION=8.0
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

"$NVCC" --ptx -std=c++14 -arch=compute_75 -lineinfo "$SRC" -o "$TMP"

# Rewrite the absolute source path to the repo-relative one.
python3 - "$TMP" "$SRC" "$EXPECTED_PTX_VERSION" <<'PY'
import re, sys
out, src, expected_version = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(out, encoding="utf-8").read()
new, n = re.subn(r'(\.file\s+\d+\s+")[^"]*/(histogram\.cu")', rf'\g<1>{src[:-len("histogram.cu")]}\g<2>', text)
# nvcc 的 banner 含 compiler build ID 和精确工具链版本。它们对 PTX
# 执行没有作用,却会把构建环境指纹带进仓库和 wheel。只保留从
# `.version` 开始的实际 PTX,再加下面的源码内容 hash。
ptx_start = new.find(".version ")
if ptx_start < 0:
    raise SystemExit("ERROR: generated PTX has no .version directive")
new = new[ptx_start:]
actual_version = new.splitlines()[0].strip()
expected_directive = f".version {expected_version}"
if actual_version != expected_directive:
    raise SystemExit(
        f"ERROR: expected {expected_directive}, generated {actual_version}")
new = "\n".join(line.rstrip() for line in new.splitlines()) + "\n"
source = open(src, "rb").read()
fnv = 0xcbf29ce484222325
for byte in source:
    fnv = ((fnv ^ byte) * 0x100000001b3) & 0xffffffffffffffff
new = f"// FerrisBoost source FNV-1a64: {fnv:016x}\n\n" + new
open(out, "w", encoding="utf-8").write(new)
print(f"  rewrote {n} .file directive(s) to a relative path")
PY

if grep -q "$HOME\|$(pwd)" "$TMP"; then
  echo "ERROR: generated PTX still contains build-machine paths" >&2
  grep -n "$HOME\|$(pwd)" "$TMP" | head -3 >&2
  exit 1
fi
mv "$TMP" "$OUT"
trap - EXIT
echo "  $OUT regenerated and sanitised"
