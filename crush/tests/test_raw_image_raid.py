# SPDX-License-Identifier: Apache-2.0
"""RawImageVFS on RAID: LVM2 RAID logical volumes inside one image, and one
member of a RAID array opened on its own.

The images are abrignoni/qnxprobe's own MIT-licensed test fixtures
(tests/fixtures/raid at the vendored commit), renamed with a `raw_` prefix.
Linux md and lvm2 wrote them (qnxprobe's tools/make_raid_fixtures.sh):

- `raw_lvm_raid` is lvm-raid-single-0: one volume group on five GPT
  partitions of one disk, with RAID4, RAID5 (four layouts and raid5_n),
  RAID6 (three layouts) and RAID10 logical volumes, each holding ext4.
- `raw_md_member` is md-r1-0, one member of a two-member md RAID1 array.
- `raw_ddf_member` is md-ddf-0, one member of a RAID5 array mdadm wrote with
  DDF metadata.
- `raw_raid_trees.sha256` is raid-trees.sha256: every file of every array,
  hashed through the assembled device, as `<array>/<path>`. The arrays of
  `raw_lvm_raid` are `lvm-raid-single-<logical volume>`.
"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pytest

from crush.core.issues import render_value
from crush.core.raw_image import RawImageOpenError
from crush.core.vfs import RawImageVFS, VFSNode
from crush.tests.conftest import FIXTURES_DIR


def _image(tmp_path: Path, stem: str) -> Path:
    img = tmp_path / f"{stem}.img"
    img.write_bytes(gzip.decompress((FIXTURES_DIR / f"{stem}.img.gz").read_bytes()))
    return img


def _files(node: VFSNode) -> list[VFSNode]:
    out: list[VFSNode] = []
    for child in node.children:
        if child.is_dir:
            out.extend(_files(child))
        else:
            out.append(child)
    return out


@pytest.mark.forensic(
    category="Known-output Verification",
    subject="Raw disk image",
    desc="Every file on the LVM2 RAID4, RAID5, RAID6 and RAID10 logical volumes of one "
    "image reads back with the bytes hashed through the device lvm2 assembled",
)
def test_every_file_on_lvm_raid_volumes_in_one_image_reads_back(tmp_path: Path) -> None:
    prefix = "lvm-raid-single-"
    want: dict[str, str] = {}
    for line in (FIXTURES_DIR / "raw_raid_trees.sha256").read_text(encoding="utf-8").splitlines():
        digest, _, relpath = line.partition("  ")
        if relpath.startswith(prefix):
            want[relpath.removeprefix(prefix)] = digest
    vfs = RawImageVFS(_image(tmp_path, "raw_lvm_raid"))
    try:
        folder = {
            str(v["label"]).removeprefix("LVM qpraid1/"): f"/{v['name']}"
            for v in vfs._handle.volumes if str(v.get("label", "")).startswith("LVM qpraid1/")
        }
        assert sorted(folder) == sorted(
            ["r4", "r5ls", "r5la", "r5rs", "r5ra", "r5n", "r6zr", "r6nr", "r6nc", "r10"])
        got: dict[str, str] = {}
        for node in vfs.root().children:
            lv = next((k for k, p in folder.items() if p == node.path), None)
            if lv is None or not node.is_dir:
                continue
            base = node.path.rstrip("/") + "/"
            for f in _files(node):
                got[f"{lv}/{f.path.removeprefix(base)}"] = hashlib.sha256(vfs.read(f)).hexdigest()
        assert len(want) == 40
        assert {path: got.get(path) for path in want} == want
    finally:
        vfs.close()


def test_an_lvm_raid_volume_shows_its_layout(tmp_path: Path) -> None:
    """The Layout of a RAID logical volume names its level, stripe and image
    volumes, and where each image volume's bytes are in the image."""
    vfs = RawImageVFS(_image(tmp_path, "raw_lvm_raid"))
    try:
        r5 = next(v for v in vfs._handle.volumes if v.get("label") == "LVM qpraid1/r5ls")
        node = next(c for c in vfs.root().children if c.path == f"/{r5['name']}")
        layout = render_value((vfs.node_info(node) or {})["Layout"])
        assert layout.startswith(
            "raid5_ls volume in 32,768-byte stripes over 3 image volume(s), 0 of them not read; "
            "image volume 0: its bytes 0–1,572,863 at image offset(s) 4,718,592; "
        )
        assert layout.endswith("image volume 2: its bytes 0–1,572,863 at image offset(s) 55,050,240")
    finally:
        vfs.close()


@pytest.mark.parametrize(("stem", "says"), [
    ("raw_md_member", "a member of a Linux md RAID array"),
    ("raw_ddf_member", "a member of a DDF RAID set, whose array is not assembled here"),
])
def test_one_raid_member_alone_is_named_not_walked(tmp_path: Path, stem: str, says: str) -> None:
    """A disk that is one member of a RAID array holds only part of its data:
    the reader names it and walks nothing, and opening it says why rather
    than that nothing was recognised."""
    with pytest.raises(RawImageOpenError, match="no readable filesystem") as raised:
        RawImageVFS(_image(tmp_path, stem))
    assert says in str(raised.value)
