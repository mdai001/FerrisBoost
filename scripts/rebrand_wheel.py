"""Give a wheel a different distribution name without changing its import package.

This is used for the explicit x86-64-v3 distribution. CPU ISA levels are not
part of the stable wheel compatibility-tag vocabulary, so the optimized build
must not share the ``ferrisboost`` distribution/version/tag identity with the
generic wheel.

    rebrand_wheel.py wheel.whl ferrisboost-v3

The script renames the wheel and its ``.dist-info`` directory, updates the
``Name`` field in METADATA, and regenerates RECORD.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
from pathlib import Path
import re
import sys
import zipfile


NAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


def wheel_name(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def metadata_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def digest(data: bytes) -> str:
    raw = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(raw).decode().rstrip("=")


def rebrand(path: Path, requested_name: str) -> Path:
    if not NAME_RE.fullmatch(requested_name):
        raise SystemExit(f"invalid distribution name: {requested_name!r}")

    normalized = wheel_name(requested_name)
    filename = path.name
    parts = filename[:-4].split("-")
    if len(parts) not in (5, 6):
        raise SystemExit(f"unsupported wheel filename: {filename}")
    parts[0] = normalized
    destination = path.with_name("-".join(parts) + ".whl")

    with zipfile.ZipFile(path) as src:
        members = [(info, src.read(info.filename)) for info in src.infolist()]

    dist_info_roots = {
        info.filename.split("/", 1)[0]
        for info, _ in members
        if ".dist-info/" in info.filename
    }
    if len(dist_info_roots) != 1:
        raise SystemExit(f"expected one .dist-info directory, found {sorted(dist_info_roots)}")
    old_root = dist_info_roots.pop()
    version_suffix = old_root.split("-", 1)[1]
    new_root = f"{normalized}-{version_suffix}"

    rewritten: list[tuple[zipfile.ZipInfo, str, bytes]] = []
    record_name = f"{new_root}/RECORD"
    saw_metadata = False
    for info, data in members:
        name = info.filename
        if name == old_root or name.startswith(old_root + "/"):
            name = new_root + name[len(old_root) :]
        if name == f"{new_root}/METADATA":
            text = data.decode("utf-8")
            text, count = re.subn(
                r"(?m)^Name: .+$", f"Name: {metadata_name(requested_name)}", text, count=1
            )
            if count != 1:
                raise SystemExit("METADATA has no unique Name field")
            data = text.encode("utf-8")
            saw_metadata = True
        rewritten.append((info, name, data))
    if not saw_metadata:
        raise SystemExit("wheel has no METADATA")

    rows = []
    for _, name, data in rewritten:
        if name != record_name:
            rows.append([name, digest(data), str(len(data))])
    rows.append([record_name, "", ""])
    record = io.StringIO()
    csv.writer(record, lineterminator="\n").writerows(rows)
    record_bytes = record.getvalue().encode()

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as out:
        for info, name, data in rewritten:
            info.filename = name
            out.writestr(info, record_bytes if name == record_name else data)
    os.replace(temporary, destination)
    if destination != path:
        path.unlink()
    return destination


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: rebrand_wheel.py WHEEL DISTRIBUTION")
    print(rebrand(Path(sys.argv[1]), sys.argv[2]))
