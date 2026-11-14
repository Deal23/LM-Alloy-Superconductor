from lmsc.paths import ROOT, public_path


def test_public_path_hides_external_directories(tmp_path):
    assert public_path(ROOT / "results" / "figure.png") == "results/figure.png"
    assert public_path(tmp_path / "private" / "data.csv") == "data.csv"
