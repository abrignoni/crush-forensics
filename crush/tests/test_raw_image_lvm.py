#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
"""RawImageVFS on the logical volumes of an LVM2 volume group, and on a LUKS
container.

The images are abrignoni/qnxprobe's own MIT-licensed test fixtures
(tests/fixtures at the vendored commit), renamed with a `raw_` prefix.
`raw_lvm` is lvm-fixture: lvm2 wrote a volume group on two GPT partitions
with logical volumes that are one run of extents, split into two runs,
striped across both physical volumes, and RAID1, each holding an ext4
filesystem. `raw_lvm.sha256` lists every file hashed from the mounted
logical volumes, as `<logical volume>/<path>`. `raw_luks1` is luks1-cbc, a
LUKS1 container cryptsetup wrote.
"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pytest

from crush.core.raw_image import RawImageOpenError
from crush.core.vfs import RawImageVFS, VFSNode
from crush.tests.conftest import FIXTURES_DIR


def _open(tmp_path: Path, stem: str) -> tuple[RawImageVFS, Path]:
    img = tmp_path / f"{stem}.img"
    img.write_bytes(gzip.decompress((FIXTURES_DIR / f"{stem}.img.gz").read_bytes()))
    return RawImageVFS(img), img


def _files(node: VFSNode) -> list[VFSNode]:
    out: list[VFSNode] = []
    for child in node.children:
        if child.is_dir:
            out.extend(_files(child))
        else:
            out.append(child)
    return out


def test_every_file_lvm_wrote_reads_back_under_its_logical_volume(tmp_path: Path) -> None:
    """Each logical volume, whether it is one run, split into runs, striped or
    RAID1, is a folder of its own, and every file in it has the bytes hashed
    from the mounted volume."""
    want: dict[str, str] = {}
    for line in (FIXTURES_DIR / "raw_lvm.sha256").read_text(encoding="utf-8").splitlines():
        digest, _, relpath = line.partition("  ")
        want[relpath] = digest
    vfs, _img = _open(tmp_path, "raw_lvm")
    try:
        folder = {
            str(v["label"]).removeprefix("LVM fixvg/"): f"/{v['name']}"
            for v in vfs._handle.volumes
            if str(v.get("label", "")).startswith("LVM fixvg/")
        }
        got: dict[str, str] = {}
        for node in vfs.root().children:
            lv = next((k for k, p in folder.items() if p == node.path), None)
            if lv is None or not node.is_dir:
                continue
            prefix = node.path.rstrip("/") + "/"
            for f in _files(node):
                got[f"{lv}/{f.path.removeprefix(prefix)}"] = hashlib.sha256(vfs.read(f)).hexdigest()
        assert len(want) == 10
        assert {path: got.get(path) for path in want} == want
        assert {"data", "striped", "mirror"} <= {p.split("/")[0] for p in want}
    finally:
        vfs.close()


def test_a_volume_read_through_a_map_leaves_no_gap_past_the_image(tmp_path: Path) -> None:
    """A logical volume split into runs or striped is placed past the image's
    end. The space no region covers is still listed, and ends at the image's
    end: the placed volume adds no gap of its own."""
    vfs, img = _open(tmp_path, "raw_lvm")
    try:
        size = img.stat().st_size
        gaps = [v for v in vfs._handle.volumes if v["kind"] == "unallocated"]
        assert gaps
        assert max(v["base"] + v["size"] for v in gaps) == size
        for v in gaps:
            node = next(c for c in vfs.root().children if c.name == v["name"])
            assert len(vfs.read(node)) == v["size"]
        mapped = [v for v in vfs._handle.volumes if v.get("lba") is None and v["kind"] == "ext4"]
        assert sorted(str(v["label"]) for v in mapped) == ["LVM fixvg/data", "LVM fixvg/striped"]
    finally:
        vfs.close()


def test_a_luks_container_is_named_as_not_read(tmp_path: Path) -> None:
    """Crush takes no LUKS passphrase: the container is refused with the
    reader's own reason rather than as an unrecognised file."""
    img = tmp_path / "raw_luks1.img"
    img.write_bytes(gzip.decompress((FIXTURES_DIR / "raw_luks1.img.gz").read_bytes()))
    with pytest.raises(RawImageOpenError, match="is luks: LUKS-encrypted and not read"):
        RawImageVFS(img)
