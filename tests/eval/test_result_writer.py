from __future__ import annotations

import json

import pytest

import prism.utils.result_writer as result_writer


def test_atomic_result_update_preserves_previous_summary_on_replace_failure(tmp_path, monkeypatch):
    path = tmp_path / "results.json"
    result_writer.write_json_result_atomic(path, {"generation": 1})

    def fail_replace(_source, _destination):
        raise OSError("injected atomic replace failure")

    monkeypatch.setattr(result_writer.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected atomic replace failure"):
        result_writer.write_json_result_atomic(path, {"generation": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"generation": 1}
    assert not (tmp_path / ".results.json.tmp").exists()
