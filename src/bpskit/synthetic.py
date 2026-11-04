"""A synthetic BPS delivery with known structure, for tests and examples.

The files have the BPS column names and formats (pipe-delimited, `AAAAMM` or
`AAAAMMDD` dates, ICD-10 codes with and without dots), so a workflow written
for the real delivery runs on it unchanged. What is planted, so a correct
analysis can be checked against it:

    E11 -> N18 -> I50 -> death   directional: hazard x4 after the predecessor
                                 (x5 for death after I50)
    J44 ~ M81                    associated through a shared frailty, no direction
    150 codes (A/B/C/G/K/L00-24) independent of everything, with rates scaled by a
                                 person-level healthcare-use intensity (lognormal)
    R53                          routine visits, Poisson in that intensity
    U07                          from 2020 only, independent of everything

Directional tests need power: at 50,000 persons the planted pairs are found;
at a few thousand the files are only good for exercising code.

    from bpskit.synthetic import make_delivery
    cfg = make_delivery("data/synthetic", n_persons=2_000)   # writes the files
    cfg["universe"], cfg["sources"]                           # how to read them
"""

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["FILES", "config", "make_delivery"]

# One people file and three diagnosis sources, named by what they hold.
FILES = {
    "universe": "sociodemograficos.csv",
    "ap": "atencion_primaria.csv",
    "cmbd": "hospitalizaciones.csv",
    "bps": "patologias_bps.csv",
}


def config(delivery_root: str | Path = ".") -> dict:
    """The `delivery_root`, `universe` and `sources` config of a synthetic delivery.

    The same shape a BPS workflow config uses, so it can be merged into one:
    `universe` names the people file's columns and sex labels, and each source
    its id, date and ICD-code columns.
    """
    return {
        "delivery_root": str(delivery_root),
        "universe": {
            "file": FILES["universe"],
            "id": "COD_ID",
            "sex": "DESC_SEXO",
            "sex_values": {"Mujer": "F", "Hombre": "M"},
            "birth": "ANYO_NAC",
            "death": "ANYO_FAC",
            "date_format": "%Y%m",
        },
        "sources": {
            "ap": {
                "file": FILES["ap"],
                "id": "COD_ID",
                "date": "FECHA_INICIO_DIAGNOSTICO",
                "date_format": "%Y%m%d",
                "code": "COD_CIE",
            },
            "cmbd": {
                "file": FILES["cmbd"],
                "id": "COD_ID",
                "date": "COD_FEC_INGRESO",
                "date_format": "%Y%m%d",
                "code": "COD_CIE",
                "position": "COD_POSICION",
            },
            "bps": {
                "file": FILES["bps"],
                "id": "COD_ID",
                "date": "FEC_INI_PATOLOGIA",
                "date_format": "%Y%m",
                "code": "COD_CIE",
            },
        },
    }


def make_delivery(out_dir: str | Path, n_persons: int = 50_000, seed: int = 1) -> dict:
    """Write a synthetic delivery of `n_persons` into `out_dir`; return its config.

    The same `n_persons` and `seed` always write the same bytes. Persons are
    followed 2015-2025; identifiers look like `P00000042`.
    """
    n = n_persons
    N = int(n)
    rng = np.random.default_rng(int(seed))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    T0, T1 = 2015.0, 2026.0  # decimal years

    birth = rng.uniform(1930, 1975, N)
    sex = rng.choice(["Mujer", "Hombre"], N)
    frail = rng.random(N) < 0.1
    use = rng.lognormal(0, 0.8, N)

    def first_time(rate):
        t = T0 + rng.exponential(1 / rate)
        return np.where(t < T1, t, np.nan)

    def after(base, mult, trigger):
        """Hazard `base`, multiplied by `mult` from the trigger's time on."""
        t0 = first_time(np.full(N, base))
        t1 = trigger + rng.exponential(1 / (base * mult), N)
        t = np.where(np.isnan(trigger) | (t0 < trigger), t0, t1)
        return np.where(t < T1, t, np.nan)

    codes = {"E11": first_time(0.01 * np.exp((2020 - birth - 65) / 15))}
    codes["N18"] = after(0.004, 4, codes["E11"])
    codes["I50"] = after(0.004, 4, codes["N18"])
    codes["J44"] = first_time(np.where(frail, 0.03, 0.004))
    codes["M81"] = first_time(np.where(frail, 0.03, 0.004))
    for c in [f"{letter}{n:02d}" for letter in "ABCGKL" for n in range(25)]:
        codes[c] = first_time(rng.uniform(0.002, 0.02) * use)
    death = after(0.01, 5, codes["I50"])

    rows = []
    for c, t in codes.items():
        ids = np.flatnonzero(~np.isnan(t))
        for rep in range(2):  # first record + one later repeat
            tt = t[ids] + rep * rng.uniform(0, 2, len(ids))
            ok = (tt < T1) & (np.isnan(death[ids]) | (tt <= death[ids]))
            rows.append(pd.DataFrame({"i": ids[ok], "t": tt[ok], "code": c}))
    k = rng.poisson(3 * use * 11)
    vi = np.repeat(np.arange(N), k)
    vt = rng.uniform(T0, T1, len(vi))
    ok = np.isnan(death[vi]) | (vt <= death[vi])
    rows.append(pd.DataFrame({"i": vi[ok], "t": vt[ok], "code": "R53"}))
    m = rng.random(N) < 0.15
    rows.append(
        pd.DataFrame({"i": np.flatnonzero(m), "t": rng.uniform(2020, T1, m.sum()), "code": "U07"})
    )
    ev = pd.concat(rows, ignore_index=True)

    def fmt(t, f):
        d = pd.to_datetime("2015-01-01") + pd.to_timedelta((pd.Series(t) - T0) * 365.25, unit="D")
        return d.dt.strftime(f).to_numpy()

    pid = np.array([f"P{i:08d}" for i in range(N)])
    pd.DataFrame(
        {
            "COD_ID": pid,
            "DESC_SEXO": sex,
            "ANYO_NAC": fmt(birth, "%Y%m"),
            "ANYO_FAC": np.where(np.isnan(death), "", fmt(np.nan_to_num(death, nan=T0), "%Y%m")),
        }
    ).to_csv(out / FILES["universe"], sep="|", index=False)
    # routine visits and U07 are primary care; the rest is split across the sources
    src = np.where(
        ev.code.isin(["R53", "U07"]),
        "ap",
        rng.choice(["ap", "cmbd", "bps"], len(ev), p=[0.6, 0.25, 0.15]),
    )
    a, c, b = (ev[src == s] for s in ("ap", "cmbd", "bps"))
    pd.DataFrame(
        {
            "COD_ID": pid[a.i],
            "FECHA_INICIO_DIAGNOSTICO": fmt(a.t, "%Y%m%d"),
            "COD_CIE": a.code + ".9",
        }
    ).to_csv(out / FILES["ap"], sep="|", index=False)
    pd.DataFrame(
        {
            "COD_ID": pid[c.i],
            "COD_FEC_INGRESO": fmt(c.t, "%Y%m%d"),
            "COD_CIE": c.code,
            "COD_POSICION": rng.integers(1, 4, len(c)),
        }
    ).to_csv(out / FILES["cmbd"], sep="|", index=False)
    pd.DataFrame(
        {"COD_ID": pid[b.i], "FEC_INI_PATOLOGIA": fmt(b.t, "%Y%m"), "COD_CIE": b.code}
    ).to_csv(out / FILES["bps"], sep="|", index=False)
    return config(out)
