# Manuscript — ppg-microvascular-feasibility-mimic

## Files

- `tex/manuscript.tex` — the LaTeX source.
- `references.bib` — Vancouver-style BibTeX (informational; the manuscript uses an inline numbered list).
- `figures/` — the figures referenced by the paper (vector PDFs).

The compiled PDF, the citation-verification log, and the figure build notes are kept out of the public release; see `.gitignore`.

## Build

```bash
cd tex
latexmk -pdf manuscript.tex
```

Or the canonical sequence:

```bash
cd tex
pdflatex manuscript.tex
pdflatex manuscript.tex
pdflatex manuscript.tex
```

The bibliography is currently an inline `\begin{enumerate}` block in the .tex file, so `bibtex` is not strictly required. If we move to a BibTeX-driven bibliography, run `bibtex manuscript` between the first and second `pdflatex` passes.
