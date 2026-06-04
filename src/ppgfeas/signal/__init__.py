"""Signal-extraction subpackage for ppgfeas.

Intended scope: the per-signal waveform-extraction code that turns raw
MIMIC-IV Waveform Database v0.1.0 records into the derived time series and
per-beat features used by the three candidate signals (cuff-anchored
perfusion-index recovery, the Mayer-band power ratio, and the per-beat
exponential time constant).

Not yet included in this release: the signal-extraction modules read
credentialed MIMIC-IV-WDB data and are being ported from the originating
repository only after a per-file data-use-agreement and secret review, so they
are deliberately absent here. This subpackage is the home they will land in.
"""
