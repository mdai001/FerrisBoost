"""Strip build-machine fingerprints from a PTX file in place.

Usage: sanitize_ptx.py <ptx> <histogram.cu repo path> <expected PTX version>

Called by scripts/build_ptx.sh on freshly generated nvcc output. It is also
idempotent on an already-sanitised checked-in PTX, so a metadata-only fix can
be applied without regenerating (the kernel body is never touched).
"""
import re
import sys

out, src, expected_version = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(out, encoding="utf-8").read()

# `-lineinfo` 的 `.file` 被 nvcc 规范化成绝对路径。源文件改成仓库相对路径。
new, n = re.subn(r'(\.file\s+\d+\s+")[^"]*/(histogram\.cu")',
                 rf'\g<1>{src[:-len("histogram.cu")]}\g<2>', text)
# CUDA 头文件的 `.file` 同样会带出 toolkit 的安装/解压目录(曾经是临时目录)。
# 只保留 `include/` 之后的相对部分。
new, n_hdr = re.subn(r'(\.file\s+\d+\s+")[^"]*/include/([^"]*")',
                     r'\g<1>cuda/include/\g<2>', new)
n += n_hdr
leftover = [line.strip() for line in new.splitlines()
            if re.match(r'\s*\.file\s+\d+\s+"(/|[A-Za-z]:[\\/])', line)]
if leftover:
    raise SystemExit("ERROR: absolute .file path left in PTX: " + leftover[0])

# nvcc 的 banner 含 compiler build ID 和精确工具链版本。它们对 PTX
# 执行没有作用,却会把构建环境指纹带进仓库和 wheel。只保留从
# `.version` 开始的实际 PTX,再加下面的源码内容 hash。
ptx_start = new.find(".version ")
if ptx_start < 0:
    raise SystemExit("ERROR: PTX has no .version directive")
new = new[ptx_start:]
actual_version = new.splitlines()[0].strip()
expected_directive = f".version {expected_version}"
if actual_version != expected_directive:
    raise SystemExit(f"ERROR: expected {expected_directive}, found {actual_version}")
new = "\n".join(line.rstrip() for line in new.splitlines()) + "\n"

# nvcc names internal-linkage symbols `_INTERNAL_<8 hex>_<file>_cu_...`, where the
# hash depends on the absolute source path. It only appears in debug-info name
# strings (encoded as decimal .b8 bytes) but makes the PTX differ between
# checkouts. Pin it to a constant so regeneration is path-independent.
_PINNED_ID = "73eb3007"
_INTERNAL_PREFIX = ",".join(str(b) for b in b"_INTERNAL_")
_PINNED_BYTES = ",".join(str(b) for b in (_PINNED_ID + "_").encode())
new = re.sub(
    rf"({re.escape(_INTERNAL_PREFIX)},)(?:\d+,){{8}}95\b",
    lambda m: m.group(1) + _PINNED_BYTES,
    new,
)

source = open(src, "rb").read()
fnv = 0xcbf29ce484222325
for byte in source:
    fnv = ((fnv ^ byte) * 0x100000001b3) & 0xffffffffffffffff
new = f"// FerrisBoost source FNV-1a64: {fnv:016x}\n\n" + new
open(out, "w", encoding="utf-8").write(new)
print(f"  rewrote {n} .file directive(s) to a relative path")
