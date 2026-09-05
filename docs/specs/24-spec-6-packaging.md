# SPEC 6 — Wire-up, verification and packaging (final)

No model or scoring changes. Pre-registration hash must not change.

## 1. Reports become generated, prose included
Add `narrative_path` to make_report / make_markdown_report and to the
comparison report. Each reads a markdown file and inlines it in place of
the TODO block (convert to HTML for the .html twin). Narrative files
under docs/narrative/. Generated layout must match the supplied
model_comparison.md and validation.md section-for-section. Fill every
{{generated}} cell from the run. Diff the output against the supplied
files and list every number that differs.

## 2. Confirm three numbers taken from summaries, not tables
- gjr_skewt persistence at every non-flagged origin is within
  0.983-0.997; quote the actual min and max.
- gjr_skewt nu at origin 2011 = 10.80 and at 2021 = 5.56.
- Rerun diagnostics.run_all and report every figure the diagnostics
  prose quotes. Also state the mean-excess shape on losses and whether
  ACF(r^2) at lag 20 is positive.
If any differ from the prose, change the prose to the table, never the
reverse, and list the changes.

## 3. Small open items
Bootstrap-seed row in the HTML config table; one sentence under the
plots about pooled marginals; cross-links between the two reports.

## 4. Repository packaging
docs/specs/ with every spec and implementer report, numbered in the order
written. README.md following the brief's component order. AIUSAGE.md and
aws_deployment.md at repo root; leave the [confirm] markers untouched —
they are Shahram's to resolve.

## Done when
Both scoreboards appear, hash regenerated and committed, leakage test
extended and passing, the numbers in the report, repo pushed.
