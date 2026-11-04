"""bpskit end to end on a synthetic delivery, in a few seconds.

Writes a 5,000-person synthetic BPS delivery, measures the incidence of type 2
diabetes (ICD-10 E11) in primary care in 2018-2025, compares women with the
whole population, and runs the transfer check on the result table: once on the
aggregate table (it passes), once on a file that names a patient (it is refused).

    python examples/quickstart.py [work_dir]
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from bpskit import transfer
from bpskit.delivery import normalise_code, parse_date, read_delivery, stream_csv
from bpskit.icd import expand_spec, prefix_mask
from bpskit.persontime import DAYS_PER_YEAR, day_of, follow_up, to_days
from bpskit.stats import poisson_rr
from bpskit.synthetic import make_delivery

work = Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp())
cfg = make_delivery(work / "delivery", n_persons=5_000)
root = Path(cfg["delivery_root"])
sentinels = ["", "-1", "-2", "-9"]

# 1. People: sex, and birth and death as days since 1970 (NaN = alive).
u = cfg["universe"]
people = read_delivery(root / u["file"], [u["id"], u["sex"], u["birth"], u["death"]])
is_f = people[u["sex"]].map(u["sex_values"]).eq("F").to_numpy()
birth = to_days(parse_date(people[u["birth"]], u["date_format"], sentinels, 15))
death = to_days(parse_date(people[u["death"]], u["date_format"], sentinels, 15))

# 2. Each person's first primary-care record of E11, streamed block by block.
ap = cfg["sources"]["ap"]
prefixes = expand_spec("E11")
first: dict[str, float] = {}
for block in stream_csv(root / ap["file"], [ap["id"], ap["date"], ap["code"]], 16, "utf-8"):
    hit = block[prefix_mask(normalise_code(block[ap["code"]]), prefixes)]
    days = to_days(parse_date(hit[ap["date"]], ap["date_format"], sentinels, 15))
    for pid, d in zip(hit[ap["id"]], days):
        first[pid] = min(d, first.get(pid, np.inf))
event = people[u["id"]].map(first).to_numpy(float)

# 3. Follow-up 2018-2025. A record before the start makes a person prevalent and
#    removes them from the risk set; an incident record ends their follow-up.
entry, exit_, in_period = follow_up(birth, death, day_of("2018-01-01"), day_of("2025-12-31"))
at_risk = in_period & ~(event < entry)
incident = at_risk & (event <= exit_)
exit_ = np.where(incident, event, exit_)
years = np.where(at_risk, (exit_ - entry) / DAYS_PER_YEAR, 0.0)

# 4. Women against the whole population: observed cases over those expected at
#    the overall rate, with an exact Poisson interval.
rate = incident.sum() / years.sum()
obs, exp = incident[is_f].sum(), years[is_f].sum() * rate
ratio, lo, hi, p = poisson_rr(obs, exp, alpha=0.05)
table = pd.DataFrame(
    {
        "group": ["women"],
        "cases": [obs],
        "expected": [exp],
        "ratio": [ratio],
        "ratio_lo": [lo],
        "ratio_hi": [hi],
        "p": [p],
    }
)
print(
    f"E11 incidence {rate * 1e5:,.0f} per 100,000 person-years; women vs all: "
    f"{float(ratio):.2f} ({float(lo):.2f}-{float(hi):.2f})"
)

# 5. The transfer check: no file leaving the machine may name a patient.
out = work / "results/out"
out.mkdir(parents=True, exist_ok=True)
table.to_csv(out / "e11_incidence.csv", index=False)
ids = set(people[u["id"]])
forbidden = transfer.forbidden_columns(cfg)
n, problems = transfer.check([out], ids, strict=True, forbidden=forbidden)
print(f"aggregate table: {n} file checked, {len(problems)} problem(s)")
assert not problems

leak = out / "first_cases.csv"
pd.DataFrame({"patient": list(first)[:3]}).to_csv(leak, index=False)
n, problems = transfer.check([out], ids, strict=True, forbidden=forbidden)
print("with a leaked file:", *problems, sep="\n  ")
assert problems and all(str(leak) in p for p in problems)
