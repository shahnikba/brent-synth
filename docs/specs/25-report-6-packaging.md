# Report — SPEC 6, wire-up, verification and packaging

## Status
Complete. 218 tests passing.

## 1. Reports are generated, prose included
`narrative_path` threaded through `make_report`, `make_markdown_report`,
`make_comparison_report` and `run_validation`. Six prose files under
`docs/narrative/`; HTML twins convert the same markdown (`markdown`
added as a dependency) rather than carrying a second copy. Missing prose
falls back to the TODO stub so a fresh checkout still generates.

`reports/validation.md` regenerates **byte-identical** to
`docs/validation.md` — every heading on the reference's exact line
number, zero differing numbers. `model_comparison.md` matches
section-for-section with all 14 `{{generated}}` cells filled.

### Differences from the supplied reference
1. **§7.1 "0.983–0.997" is wrong.** Actual gjr_skewt range at
   non-flagged origins is **0.98281–0.99891**. Both ends outside the
   claim. The sentence is now generated from the table.
2. **§2 gjr_skewt_vt rank stability 0.333 -> 0.125.** The reference row
   paired the amended score with the original rule's stability.
3. **§2 iid_t 4.313 -> 4.312** (4.3125, round-half-to-even).
4. **§5.2 two cells**: gjr_skewt calm 0.0001 -> 0.000 (3dp);
   garch_normal stress 0.068 -> 0.067 (0.0675).
5. **§6 figure filename** renamed to `path-ranks.png` to match.

## 2. Three number sets verified
Persistence claim wrong, as above. nu confirmed: 10.80161 at 2011,
5.55814 at 2021. Full `diagnostics.run_all` rerun; ACF(r²) at lag 20 is
**+0.08164 (positive)** and the mean-excess function on losses **rises**
(slope +0.274; 0.02087 -> 0.03795). Both `[check:]` markers in the
diagnostics prose resolved and the prose corrected: n 4754 -> 4753, end
date 2026-09-03, Ljung-Box 1.7e-213 -> 1.9e-213, and the gain tail's
"1/xi = 7.9" removed since xi/se = 1.25 fails the significance gate.

## 3. Small items
Bootstrap-seed row added to the HTML config table; cross-links verified
both directions and both formats; the pooled-marginals sentence renders
from `validation_thresholds.md`.

## 4. Packaging
`AIUSAGE.md` and `aws_deployment.md` moved to the repo root with
`git mv`; the `[confirm]` markers were not touched. `docs/specs/`
populated with all 25 documents. README follows the brief's component
order with working relative links.

Nothing unexpected was found in `docs/` beyond a gitignored `.DS_Store`.
