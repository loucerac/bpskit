import numpy as np
import pandas as pd
import pytest

from bpskit.icd import Icd9Mapper, expand_spec, harmonise, icd10_chapter, match_table


def test_expand_spec_ranges_and_codes():
    assert expand_spec("A15-A17, B18, H35.3") == ["A15", "A16", "A17", "B18", "H353"]
    with pytest.raises(ValueError):
        expand_spec("A19-A15")
    with pytest.raises(ValueError):
        expand_spec("A1-A15")


def test_match_table_applies_exclusions_and_overlap():
    conds = pd.DataFrame(
        {"condition_id": ["dm", "endo"], "codes": ["E10-E14", "E00-E90"], "exclude": ["E13", ""]}
    )
    m = match_table(pd.Series(["E119", "E139", "E039", "I10"]), conds)
    assert set(map(tuple, m.to_numpy())) == {
        ("E119", "dm"),
        ("E119", "endo"),
        ("E139", "endo"),
        ("E039", "endo"),
    }


def test_chapters_including_u_between_t_and_v():
    np.testing.assert_array_equal(
        icd10_chapter(["A00", "H60", "U07", "V01", "Z99", "E119"]), [1, 8, 22, 20, 21, 0]
    )


class _FakeMapper(Icd9Mapper):
    def __init__(self):
        self._mapper, self._cache = None, {"4019": "I10", "V4581": "Z951"}


def test_harmonise_places_ve_codes_by_date():
    df = pd.DataFrame(
        {
            "code": ["4019", "V4581", "V4581", "E119", "9999"],
            "day": [100.0, 100.0, 300.0, 100.0, 100.0],
        }
    )
    system = harmonise(df, changeover_day=200.0, mapper=_FakeMapper())
    # Digits are always ICD-9; V45.81 only before the changeover; E11.9 is ICD-10 diabetes.
    assert list(system) == ["icd9_mapped", "icd9_mapped", "icd10", "icd10", "icd9_unmapped"]
    assert list(df.code) == ["I10", "Z951", "V4581", "E119", "9999"]
