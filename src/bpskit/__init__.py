"""Shared tools for studies on a BPS (Base Poblacional de Salud, SAS) delivery.

    bpskit.delivery    read the pipe-delimited delivery: sentinels, dates, streaming
    bpskit.icd         ICD-10 code specs and chapters, ICD-9 -> ICD-10 mapping (CMS GEM)
    bpskit.persontime  dates as numbers, age bands, Lexis splitting, person streaming
    bpskit.stats       exact Poisson and binomial tests, BH / Bonferroni
    bpskit.transfer    the check that no file leaving the data machine names a patient

The command line (`bpskit pullable`, `bpskit check-logs`) runs the transfer
check on the data machine.
"""

__version__ = "0.1.0"
