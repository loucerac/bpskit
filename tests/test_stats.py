import numpy as np
import pandas as pd
import pytest

from bpskit.delivery import parse_date
from bpskit.stats import adjust, bh


def test_bh_matches_hand_computation_and_skips_nan():
    q = bh(pd.Series([0.01, 0.04, np.nan, 0.03]))
    np.testing.assert_allclose(q.dropna().to_numpy(), [0.03, 0.04, 0.04])
    assert np.isnan(q.iloc[2])
    np.testing.assert_allclose(adjust(pd.Series([0.01, 0.5]), "bonferroni"), [0.02, 1.0])
    with pytest.raises(ValueError):
        adjust(pd.Series([0.1]), "holm")


def test_parse_date_mid_month_and_sentinels():
    s = parse_date(pd.Series(["201503", "-1", "201512.0"]), "%Y%m", ["-1"], mid_month_day=15)
    assert s.iloc[0] == pd.Timestamp("2015-03-15")
    assert pd.isna(s.iloc[1])
    assert s.iloc[2] == pd.Timestamp("2015-12-15")


def test_parse_date_refuses_a_wrong_format():
    with pytest.raises(ValueError):
        parse_date(pd.Series(["2015-03-01", "2016-01-01"]), "%Y%m%d", [], mid_month_day=15)
