# SPEC 5.2 — Numbers needed for the narrative (no model changes)

## 1. Per-loss breakdown, champion vs FIGARCH
For each of the four ranking losses, the mean rank of gjr_skewt and
figarch_skewt over selection and over confirmation origins separately.
The narrative needs to say *which* loss GJR wins the composite on.

## 2. PIT by regime and on common days
- PIT chi-square and shape per model, split by regime label.
- Same restricted to the origins where gjr_skewt_vt is available, so all
  six models are judged on identical days.
- An effect size next to every chi-square p: KS distance from
  Uniform(0,1) and the largest single-bin deviation from 1/20.
- One PIT histogram per model per regime (3 x 6 small multiples).

## 3. Persistence by origin
Origin year x GARCH-family model, flagged when >= 1; plus nu and gamma
for gjr_skewt. A line under the parameter-drift figure marking 1.0.

## 4. Stress origins, per loss
For every stress-labelled origin: the four losses per model (not just
ranks), plus drawdown rank and worst-day rank. Same for the 2019 -> 2020
origin even if its label changes.

## 5. Report placement
Items 1-4 go into model_comparison.md as new subsections of
sec 2 (1), sec 5 (2), sec 7 (3), sec 4 (4). No change to
preregistration.md or the hash.
