# Report — markdown validation report

## Status
Added. 120 tests passing (6 new).

`make_markdown_report` mirrors `make_report` section for section, driven
by the same figures and the same summary counts. Markdown cannot portably
inline an image, so the five plots are written to `<stem>_figures/` and
linked relatively. `run_validation` writes both; `markdown_path=None`
skips it.

## Two defects found while doing it

### A test was overwriting the real report
`test_run_validation_end_to_end` passed `out_path` to `tmp_path` but let
`markdown_path` default, so its reduced-size run (horizon 60, 150 paths)
silently wrote over the project's `reports/validation.md`. The result was
a markdown report claiming 0 spread mismatches next to an HTML report
claiming 5, because they came from different runs. Fixed, with a
regression test that snapshots `reports/` before and after a full
`run_validation` call.

### .DS_Store got committed
Swept up by `git add -A`. Removed via amend and added to `.gitignore`.

## One test of mine was wrong
The first determinism test wrote to `first.md` and `second.md` and
compared bytes — but the figure directory is named after the document
stem, so the two legitimately link to different paths. Rewritten to write
the same path twice.
