import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from run_analysis import density_groups, mass_matched_catalogues, synthetic_catalog, serial, jk_cov


def test_density_ranks_and_ties():
    group = density_groups(np.ones(10), np.arange(10))
    assert list(group[:2]) == ["low", "low"]
    assert list(group[-2:]) == ["high", "high"]
    assert np.sum(group=="medium") == 6


def test_mass_matching_and_original_unchanged():
    cats = [synthetic_catalog(1,120), synthetic_catalog(2,140)]
    cats[0]["snapshot"] = 99
    cats[1]["snapshot"] = 50
    cfg = {"seed":1, "cross_redshift_mass_bins_log10":[9,10,11,12]}
    matched, record = mass_matched_catalogues(cats,cfg)
    assert len(cats[0]["coords"]) == 120
    assert len(matched[0]["coords"]) == len(matched[1]["coords"])
    hist = [np.histogram(np.log10(c["stellar_mass_msun"]), record["edges_log10_Msun"])[0] for c in matched]
    assert np.array_equal(hist[0],hist[1])


def test_jk_cov_and_strict_json():
    data = np.array([[1.,np.nan],[2.,np.nan],[3.,np.nan]])
    cov = jk_cov(data)
    assert np.isclose(cov[0,0],4/3)
    assert np.isnan(cov[1,1])
    assert serial({"v":np.nan,"a":np.array([1,np.inf])}) == {"v":None,"a":[1.,None]}
