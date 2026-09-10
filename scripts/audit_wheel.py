"""Audit a built wheel for personal or machine-specific fingerprints.

Run against the real release artifact, not the source tree: several leaks only
appear after compilation (rustc debug info) or after packaging (maturin's SBOM),
and one arrived through a checked-in CUDA PTX rather than any Rust code.
"""
import getpass, glob, json, os, re, socket, sys, zipfile

wheel = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob("target/wheels/*.whl"), key=os.path.getmtime)[-1]
z = zipfile.ZipFile(wheel)
home, user, host = os.path.expanduser("~"), getpass.getuser(), socket.gethostname()
cwd = os.getcwd()

# The canonical public project URL intentionally contains the maintainer's
# account name. Remove only these exact, approved identifiers for the username
# check; a bare username anywhere else remains a finding.
APPROVED_PUBLIC_IDENTIFIERS = [
    b"https://github.com/mdai001/FerrisBoost",
    b"github.com/mdai001/FerrisBoost",
    b"https://mdai001.github.io/FerrisBoost",
    b"mdai001.github.io/FerrisBoost",
]

CHECKS = [
    ("home directory",        [home.encode()]),
    ("username",              [user.encode()]),
    ("hostname",              [host.encode(), host.split(".")[0].encode()]),
    # dirname, not cwd: the leak-prone part is the directory tree above the
    # project. Matching on the project name itself would flag the module name.
    ("checkout path",         [cwd.encode(), os.path.dirname(cwd).encode()]),
    ("temp dirs",             [b"/tmp/claude", b"/var/tmp/", b"TMPDIR"]),
    ("benchmark scratch",     [b"scratchpad", b"full.log"]),
    # An embedded dataset path is absolute, so it is already caught by the
    # home / checkout / cargo-registry checks. Matching bare extensions here
    # would flag the format strings the reader legitimately contains.
    ("git worktree paths",    [b"/.git/", b"refs/heads/"]),
    ("editor/IDE metadata",   [b".vscode", b".idea", b".DS_Store"]),
    ("env/credentials",       [b".env", b"AWS_SECRET", b"API_KEY", b"BEGIN PRIVATE KEY",
                               b"password=", b"token="]),
    ("shell history",         [b".bash_history", b".zsh_history"]),
    ("cargo registry (raw)",  [b"/home/", b"/Users/", b"C:\\\\Users"]),
    ("embedded GPU name",     [b"GeForce", b"Quadro", b"Tesla V", b"RTX "]),
    ("compiler fingerprint",  [b"Compiler Build ID:",
                               b"Cuda compilation tools, release",
                               b"Based on NVVM "]),
    ("debug logs",            [b".log\x00"]),
]

print(f"wheel : {os.path.basename(wheel)}")
print(f"size  : {os.path.getsize(wheel)/2**20:.2f} MB")
print(f"member count: {len(z.infolist())}\n")

print("── file list ──")
for i in z.infolist():
    print(f"  {i.file_size/1024:10.1f} KB  {i.filename}")

blob = {i.filename: z.read(i.filename) for i in z.infolist()}
print("\n── fingerprint scan ──")
findings = 0
for label, pats in CHECKS:
    hits = []
    for fn, data in blob.items():
        scan_data = data
        if label == "username":
            for approved in APPROVED_PUBLIC_IDENTIFIERS:
                scan_data = scan_data.replace(approved, b"")
        for pat in pats:
            if pat and pat in scan_data:
                hits.append(f"{fn}({pat.decode(errors='replace')[:24]})")
                break
    if hits:
        findings += 1
        print(f"  LEAK  {label:22} {', '.join(sorted(set(hits))[:3])}")
    else:
        print(f"  ok    {label:22} absent")

print("\n── METADATA ──")
meta = next(v for k, v in blob.items() if k.endswith("dist-info/METADATA")).decode()
for line in meta.splitlines():
    if line.strip():
        print("   ", line[:110])

print("\n── RECORD integrity ──")
rec = next(v for k, v in blob.items() if k.endswith("dist-info/RECORD")).decode()
import base64, hashlib, csv, io
ok = bad = 0
for row in csv.reader(io.StringIO(rec)):
    if not row or not row[1]:
        continue
    want = row[1].split("=", 1)[1]
    got = base64.urlsafe_b64encode(hashlib.sha256(blob[row[0]]).digest()).decode().rstrip("=")
    ok, bad = (ok + 1, bad) if want == got else (ok, bad + 1)
print(f"    {ok} hashes verified, {bad} mismatched")

print("\n── WHEEL tags ──")
print("   ", next(v for k, v in blob.items() if k.endswith("dist-info/WHEEL")).decode().replace("\n", " | "))

print(f"\nRESULT: {'CLEAN' if findings == 0 and bad == 0 else f'{findings} fingerprint categories, {bad} hash mismatches'}")
sys.exit(1 if (findings or bad) else 0)
