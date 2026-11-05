import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import run_experiment


def test_missing_real_input_stops_before_any_download_or_synthetic_analysis(tmp_path):
    with patch.object(run_experiment.subprocess, "run") as request:
        code = run_experiment.main(["--raw-dir", str(tmp_path/"absent"), "--run-dir", str(tmp_path/"run")])
        request.assert_not_called()
    assert code == 2
    state = json.loads((tmp_path/"run/execution.json").read_text())
    assert state["mode"] == "REAL_TNG_ONLY"
    assert state["status"] == "blocked_missing_real_data"
    assert len(state["missing_files"]) == 20
    assert not list((tmp_path/"run").glob("**/global.npz"))


def test_key_not_configured_does_not_try_download(tmp_path, monkeypatch):
    monkeypatch.delenv("TNG_API_KEY", raising=False)
    with patch.object(run_experiment.subprocess, "run") as request:
        code = run_experiment.main(["--download", "--run-dir", str(tmp_path/"run"), "--raw-dir", str(tmp_path/"raw")])
        request.assert_not_called()
    assert code == 2
    assert json.loads((tmp_path/"run/execution.json").read_text())["status"] == "blocked_missing_authentication"


def test_existing_execution_never_overwritten(tmp_path):
    (tmp_path/"run").mkdir()
    (tmp_path/"run/retained.txt").write_text("user data")
    with pytest.raises(SystemExit):
        run_experiment.main(["--run-dir", str(tmp_path/"run")])
    assert (tmp_path/"run/retained.txt").read_text() == "user data"
