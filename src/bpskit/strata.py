"""Strata for indirect standardisation, and every within-person pair of records.

`Strata` splits person-time by sex, age band, calendar year and, optionally,
prior-year healthcare-contact intensity; expected counts are person-time
times a reference rate per stratum. `person_pairs` enumerates ordered pairs of
a person's records in bounded blocks. Both were written for the disease-
trajectory method (Jensen et al. 2014) and work for any indirect standardisation.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd

from bpskit.persontime import DAYS_PER_YEAR, age_bands, band_index, day_of, lexis, to_days

__all__ = ["Strata", "person_pairs"]


class Strata:
    """Sex x age band x calendar year x (optionally) prior-year contact intensity.

    Stratum index s = ((sex * B + band) * Y + year) * C + level, sex 0 = F,
    1 = M, so the F strata come first ("sex-major").

    `contact_days` is a dense (persons x years) array of distinct record days in
    the setting, first column = year `contact_year0`, cut at `contact_bins`
    into C = len(bins) + 1 levels.

    Why intensity, and why the previous year: people who visit more accumulate
    more first records, so without it every pair looks associated. A yes/no
    flag stratifies nothing when nearly everyone has a record every year. A
    same-year count is partly made by the outcome, since D2's own first record
    is a contact.
    """

    BLOCK = 1_000_000  # rows per Lexis split: the (rows x bands) arrays stay under ~200 MB

    def __init__(
        self,
        bands: pd.DataFrame,
        year_starts: np.ndarray,
        contact_days=None,
        contact_year0: int = 0,
        contact_bins=(1,),
    ):
        self.bands = bands
        self.B = len(bands)
        self.ys = np.asarray(year_starts, float)  # Y + 1 boundaries, in days
        self.Y = len(self.ys) - 1
        self.year0 = int(pd.to_datetime(self.ys[0], unit="D").year)
        self.coff = self.year0 - contact_year0 - 1  # column of the previous year
        if contact_days is None:
            self.levels, self.C = None, 1
        else:
            if self.coff < 0:
                raise ValueError(
                    "contact strata need a calendar year of records before the "
                    "first stratum year: set index_from a year after washout_from"
                )
            bins = np.asarray(sorted(contact_bins))
            self.levels = np.searchsorted(bins, contact_days, side="right").astype(np.uint8)
            self.C = len(bins) + 1
        self.S = 2 * self.B * self.Y * self.C

    @classmethod
    def for_analysis(
        cls,
        traj: dict,
        ages: dict,
        washout_from,
        index_from,
        last_date,
        contacts=None,
        year_starts=None,
    ):
        """The strata a trajectory analysis uses, built the same way by every rule.

        `year_starts` defaults to 1 January of each year from `index_from`, with
        the last boundary at the day after `last_date`.
        """
        if year_starts is None:
            end = day_of(last_date) + 1
            y0, y1 = pd.Timestamp(index_from).year, pd.Timestamp(last_date).year
            year_starts = to_days(
                pd.Series(pd.to_datetime([f"{y}-01-01" for y in range(y0, y1 + 2)]))
            )
            year_starts[-1] = end
        return cls(
            age_bands(ages["band_width"], ages["open_band_from"]),
            year_starts,
            contacts,
            contact_year0=pd.Timestamp(washout_from).year,
            contact_bins=traj["contact_bins"],
        )

    def level_shares(self) -> np.ndarray:
        """Share of persons at each intensity level, per stratum year (Y x C)."""
        if self.levels is None:
            return np.ones((self.Y, 1))
        cols = self.levels[:, self.coff : self.coff + self.Y]
        return np.stack([(cols == c).mean(axis=0) for c in range(self.C)], axis=1)

    def _contact(self, person, y):
        if self.levels is None:
            return 0
        return self.levels[person, y + self.coff].astype(np.int64)

    def at(self, day, sex, birth, person) -> np.ndarray:
        """Stratum of single points in time (days outside the years are clipped)."""
        y = np.clip(np.searchsorted(self.ys, day, side="right") - 1, 0, self.Y - 1)
        b = band_index((day - birth) / DAYS_PER_YEAR, self.bands)
        return ((sex * self.B + b) * self.Y + y) * self.C + self._contact(person, y)

    def accumulate(self, acc: np.ndarray, key, sex, birth, entry, exit_, person) -> None:
        """acc[key, s] += person-years of [entry, exit) falling in stratum s, in place."""
        for lo in range(0, len(entry), self.BLOCK):
            sl = slice(lo, lo + self.BLOCK)
            self._accumulate(acc, key[sl], sex[sl], birth[sl], entry[sl], exit_[sl], person[sl])

    def _accumulate(self, acc, key, sex, birth, entry, exit_, person) -> None:
        flat = acc.reshape(-1)
        for y in range(self.Y):
            lo = np.maximum(entry, self.ys[y])
            hi = np.minimum(exit_, self.ys[y + 1])
            m = hi > lo
            if not m.any():
                continue
            bm = birth[m]
            pt = lexis((lo[m] - bm) / DAYS_PER_YEAR, (hi[m] - bm) / DAYS_PER_YEAR, self.bands)
            r, b = np.nonzero(pt)
            s = ((sex[m][r] * self.B + b) * self.Y + y) * self.C + self._contact(person[m][r], y)
            flat += np.bincount(key[m][r] * self.S + s, weights=pt[r, b], minlength=flat.size)

    def integrate(self, rate: np.ndarray, col, sex, birth, entry, exit_, person) -> np.ndarray:
        """Expected count per row: sum over strata of person-years of [entry, exit) in s
        times rate[s, col[row]].

        Built exactly the way `accumulate` builds the bulk term, so a per-pair
        correction computed with it can never exceed the bulk and drive an
        expected count negative.
        """
        if len(entry) > self.BLOCK:
            return np.concatenate(
                [
                    self.integrate(
                        rate,
                        col[i : i + self.BLOCK],
                        sex[i : i + self.BLOCK],
                        birth[i : i + self.BLOCK],
                        entry[i : i + self.BLOCK],
                        exit_[i : i + self.BLOCK],
                        person[i : i + self.BLOCK],
                    )
                    for i in range(0, len(entry), self.BLOCK)
                ]
            )
        out = np.zeros(len(entry))
        for y in range(self.Y):
            lo = np.maximum(entry, self.ys[y])
            hi = np.minimum(exit_, self.ys[y + 1])
            m = np.flatnonzero(hi > lo)
            if not len(m):
                continue
            bm = birth[m]
            pt = lexis((lo[m] - bm) / DAYS_PER_YEAR, (hi[m] - bm) / DAYS_PER_YEAR, self.bands)
            r, b = np.nonzero(pt)
            rows = m[r]
            s = ((sex[rows] * self.B + b) * self.Y + y) * self.C + self._contact(person[rows], y)
            out += np.bincount(rows, weights=pt[r, b] * rate[s, col[rows]], minlength=len(out))
        return out


def person_pairs(person: np.ndarray, max_pairs: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Every ordered pair of distinct rows of the same person.

    `person` must be sorted. Yields (row_a, row_b) index arrays in blocks of
    about `max_pairs` pairs; one person's pairs are never split across blocks,
    so a single person with more than sqrt(max_pairs) records makes a block
    bigger than `max_pairs`.
    """
    if not len(person):
        return
    starts = np.flatnonzero(np.r_[True, person[1:] != person[:-1]])
    sizes = np.diff(np.r_[starts, len(person)]).astype(np.int64)
    cum = np.cumsum(sizes * sizes)
    g0 = 0
    while g0 < len(starts):
        base = cum[g0 - 1] if g0 else 0
        g1 = max(int(np.searchsorted(cum, base + max_pairs, side="right")), g0 + 1)
        st, sz = starts[g0:g1], sizes[g0:g1]
        rows = np.arange(st[0], st[-1] + sz[-1])
        rs, rz = np.repeat(st, sz), np.repeat(sz, sz)
        a = np.repeat(rows, rz)
        b = np.repeat(rs, rz) + np.arange(len(a)) - np.repeat(np.cumsum(rz) - rz, rz)
        keep = a != b
        yield a[keep], b[keep]
        g0 = g1
