"""Deterministic serialization of processed ML data."""

from __future__ import annotations

import gzip
from pathlib import Path

from ml.common.io import read_jsonl_gz, write_jsonl_gz
from ml.common.provenance import file_sha256


def test_gzip_output_is_byte_identical_across_paths(tmp_path: Path) -> None:
    rows = [{"pair_id": f"p{i}", "scores": [0.5, i]} for i in range(100)]
    a = tmp_path / "one" / "pairs_dev.jsonl.gz"
    b = tmp_path / "two" / "other-name.jsonl.gz"
    write_jsonl_gz(a, rows)
    write_jsonl_gz(b, rows)
    write_jsonl_gz(a, rows)  # rewrite in place: new random temporary basename
    assert file_sha256(a) == file_sha256(b)
    header = a.read_bytes()[:10]
    assert header[3] & 0x08 == 0  # no FNAME field (no temporary file name embedded)
    assert header[4:8] == b"\0\0\0\0"  # mtime 0
    assert list(read_jsonl_gz(a)) == rows
    assert gzip.decompress(b.read_bytes()).count(b"\n") == 100
    assert a.stat().st_mode & 0o777 == 0o644
