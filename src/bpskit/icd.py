"""ICD-10 code specifications, chapters, and ICD-9 -> ICD-10 mapping.

Codes are handled in normalised form: upper case, no dot ('E119').
"""

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from bpskit.delivery import normalise_code

__all__ = [
    "expand_spec",
    "prefix_mask",
    "match_table",
    "icd10_chapter",
    "CHAPTER_ROMAN",
    "Icd9Mapper",
    "harmonise",
]

_CAT = re.compile(r"^[A-Z]\d{2}$")


def _cat_index(c: str) -> int:
    """'A00' -> 0, 'B12' -> 112: 3-character categories as integers, for ranges."""
    if not _CAT.match(c):
        raise ValueError(f"range end {c!r} is not a 3-character ICD-10 category")
    return (ord(c[0]) - 65) * 100 + int(c[1:])


def _cat_from_index(i: int) -> str:
    return f"{chr(65 + i // 100)}{i % 100:02d}"


def expand_spec(spec: str) -> list[str]:
    """Expand a code specification into code prefixes.

    'A15-A19, B18, H35.3' -> ['A15', 'A16', 'A17', 'A18', 'A19', 'B18', 'H353'].
    Range ends must be 3-character categories; a single code may be longer and
    matches as a prefix. Raises ValueError on a malformed or empty range.
    """
    out: list[str] = []
    for tok in str(spec).split(","):
        tok = tok.strip().upper().replace(".", "")
        if not tok:
            continue
        if "-" in tok:
            a, b = (t.strip() for t in tok.split("-"))
            ia, ib = _cat_index(a), _cat_index(b)
            if ib < ia:
                raise ValueError(f"empty range {tok!r}")
            out += [_cat_from_index(i) for i in range(ia, ib + 1)]
        else:
            out.append(tok)
    return out


def prefix_mask(codes: pd.Series, prefixes: list[str]) -> np.ndarray:
    """True where a code starts with any of `prefixes`."""
    mask = np.zeros(len(codes), dtype=bool)
    # One isin per prefix length is far faster than one startswith per prefix.
    for length in sorted({len(p) for p in prefixes}):
        want = {p for p in prefixes if len(p) == length}
        mask |= codes.str[:length].isin(want).to_numpy()
    return mask


def match_table(unique_codes: pd.Series, conditions: pd.DataFrame) -> pd.DataFrame:
    """(code, condition_id) for every code each condition's `codes` spec matches.

    A condition row may carry an `exclude` spec, removed from what `codes`
    matched. A code can belong to several conditions (a chapter row and a
    condition inside it), so this is a table, not a mapping.
    """
    codes = pd.Series(unique_codes, dtype=str).reset_index(drop=True)
    rows = []
    for _, c in conditions.iterrows():
        m = prefix_mask(codes, expand_spec(c["codes"]))
        if c.get("exclude"):
            m &= ~prefix_mask(codes, expand_spec(c["exclude"]))
        rows.append(pd.DataFrame({"code": codes[m].to_numpy(), "condition_id": c["condition_id"]}))
    return pd.concat(rows, ignore_index=True)


# First 3-character category of each WHO ICD-10 chapter, in code order. U
# (chapter XXII, special purposes) sorts between T and V.
_CHAPTER_STARTS = [
    ("A00", 1),
    ("C00", 2),
    ("D50", 3),
    ("E00", 4),
    ("F00", 5),
    ("G00", 6),
    ("H00", 7),
    ("H60", 8),
    ("I00", 9),
    ("J00", 10),
    ("K00", 11),
    ("L00", 12),
    ("M00", 13),
    ("N00", 14),
    ("O00", 15),
    ("P00", 16),
    ("Q00", 17),
    ("R00", 18),
    ("S00", 19),
    ("U00", 22),
    ("V00", 20),
    ("Z00", 21),
]
CHAPTER_ROMAN = {
    i: r
    for i, r in enumerate(
        "- I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI XVII XVIII XIX XX XXI XXII".split()
    )
}


def icd10_chapter(codes: Iterable[str]) -> np.ndarray:
    """Chapter number (1-22) of 3-character ICD-10 categories; 0 for anything else."""
    codes = pd.Series(codes, dtype=str)
    ok = codes.str.fullmatch(r"[A-Z]\d{2}").to_numpy()
    idx = np.full(len(codes), -1)
    idx[ok] = [(ord(c[0]) - 65) * 100 + int(c[1:]) for c in codes[ok]]
    starts = np.array([_cat_index(s) for s, _ in _CHAPTER_STARTS])
    nums = np.array([n for _, n in _CHAPTER_STARTS])
    pos = np.searchsorted(starts, idx, side="right") - 1
    return np.where(ok & (pos >= 0), nums[np.clip(pos, 0, None)], 0)


# --------------------------------------------------------------------------- #
# ICD-9 -> ICD-10
# --------------------------------------------------------------------------- #
class Icd9Mapper:
    """Map ICD-9-CM codes to ICD-10-CM with the CMS GEM bundled in icd-mappings.

    Results are cached per code, because a delivery repeats a few thousand
    codes hundreds of millions of times. With `enabled=False`, or when
    icd-mappings is not installed, every code maps to None and is dropped as
    unmapped rather than passed through as if it were ICD-10.
    """

    def __init__(self, enabled: bool = True):
        self._mapper = None
        if enabled:
            try:
                from icdmappings import Mapper

                self._mapper = Mapper()
            except ImportError:
                print("[WARN ] icd-mappings not installed: ICD-9 records are dropped, not mapped")
        self._cache: dict[str, str | None] = {}

    def map(self, codes: pd.Series) -> pd.Series:
        new = sorted(set(codes.unique()) - self._cache.keys())
        if new:
            got = (
                self._mapper.map(new, source="icd9", target="icd10")
                if self._mapper
                else [None] * len(new)
            )
            self._cache.update(zip(new, got))
        return codes.map(self._cache)


def harmonise(df: pd.DataFrame, changeover_day: float | None, mapper: Icd9Mapper) -> pd.Series:
    """Map ICD-9 codes in `df.code` to ICD-10 in place; return each row's system.

    A code is ICD-9 when it is all digits. Before `changeover_day` (None: never)
    the letter-led supplementary (V01-V91) and external-cause (E800-E999) codes
    count as ICD-9 too. Those shapes are also valid ICD-10 (E11 is diabetes),
    so only a date can place them, and only where the changeover was measured
    on that file.

    Returns:
        "icd10", "icd9_mapped" or "icd9_unmapped" per row. Unmapped rows keep
        their ICD-9 code; the caller drops them.
    """
    is9 = df.code.str.fullmatch(r"\d+")
    if changeover_day is not None:
        is9 |= (df.day < changeover_day) & df.code.str.match(r"^(V\d{2}|E[89]\d{2})")
    system = pd.Series(np.where(is9, "icd9", "icd10"), index=df.index)
    if is9.any():
        m = mapper.map(df.code[is9])
        hit = m.notna()
        system[is9] = np.where(hit, "icd9_mapped", "icd9_unmapped")
        df.loc[hit[hit].index, "code"] = normalise_code(m[hit])
    return system
