"""Cohort and linkage subpackage for ppgfeas.

Intended scope: the code that links MIMIC-IV Waveform Database v0.1.0 records to
the MIMIC-IV clinical layer (ICU stays and the charted noninvasive blood
pressure cycles that anchor Signal 1) and assembles the analysis cohort.

Not yet included in this release: the cohort and linkage modules read
credentialed MIMIC-IV data and are being ported from the originating repository
only after a per-file data-use-agreement and secret review, so they are
deliberately absent here. This subpackage is the home they will land in.
"""
