"""Shared tools for studies on a BPS (Base Poblacional de Salud, SAS) delivery.

    bpskit.delivery    read the pipe-delimited delivery: sentinels, dates, streaming
    bpskit.universe    the people file into the universe table (one row per person)
    bpskit.ingest      a diagnosis file into dated, ICD-10-harmonised events
    bpskit.quality     ICD harmonisation and implausible-record tables
    bpskit.icd         ICD-10 code specs and chapters, ICD-9 -> ICD-10 mapping (CMS GEM)
    bpskit.persontime  dates as numbers, age bands, Lexis splitting, person streaming
    bpskit.stats       exact Poisson and binomial tests, exact two-group rate contrasts,
                       BH / Bonferroni
    bpskit.strata      strata for indirect standardisation, within-person record pairs
    bpskit.cluster     Jaccard similarity, Markov clustering (MCL), modularity, adjusted Rand
    bpskit.mixture     latent class analysis (Bernoulli mixture, EM) with unobserved cells
    bpskit.transfer    the check that no file leaving the data machine names a patient
    bpskit.synthetic   a synthetic BPS delivery with planted structure, for tests

The command line (`bpskit pullable`, `bpskit check-logs`) runs the transfer
check on the data machine.
"""

__version__ = "0.6.0"  # x-release-please-version
