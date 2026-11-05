import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from evidence import finite_json, paired_contrast, small_scale_mask, audit_saved_analysis


def test_small_scale_bins_fully_inside_predeclared_range():
    assert small_scale_mask([.1, .2, .5, 1., 2.], [.2, 1.]).tolist() == [False, True, True, False]
    with pytest.raises(ValueError):
        small_scale_mask([.1, 1.], [2., 1.])


def test_paired_covariance_keeps_shared_region_fluctuations():
    a = np.array([[1., 3.], [2., 4.], [3., 5.]])
    b = a-2.
    result = paired_contrast([2., 4.], [0., 2.], a, b, [True, True])
    assert result["mean_difference"] == 2.
    assert result["internal_jk_se"] == 0.
    assert result["status"] != "causal_evidence"


def test_missing_replicate_bin_is_not_interpolated():
    result = paired_contrast([1., 2.], [0., 0.], [[1., np.nan], [1., 2.]], [[0., 0.], [0., 0.]], [True, True])
    assert result["bin_indices"] == [0]
    with pytest.raises(ValueError):
        paired_contrast([1.], [0.], [[1.]], [[0.]], [True])


def test_nonfinite_json_is_strict_serializable():
    result = finite_json({"x": np.array([np.nan, np.inf, 1.]), "n": np.int64(2)})
    assert json.loads(json.dumps(result, allow_nan=False)) == {"x": [None, None, 1.], "n": 2}


def test_actual_saved_synthetic_case_never_certifies_physical_cv_claim():
    root = Path(__file__).resolve().parents[1]
    saved = root/"results"/"synthetic_validation_v2"/"synthetic_clustered"
    if not saved.exists():
        pytest.skip("saved end-to-end synthetic fixture not available")
    cfg = json.loads((root/"config.json").read_text())
    output = audit_saved_analysis(saved, cfg)
    assert output["physical_claim_status"] == "pending_real_data"
    assert np.isclose(sum(v["raw_pair_share"] for v in output["joint_4_raw_DD_sources"].values()), 1.)
    json.dumps(output, allow_nan=False)
