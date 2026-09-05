# SPEC 5.1 — Follow-ups to the backtest

## 1. Amend the ranking rule for unavailable models
A candidate whose fit raises at an origin receives rank n_models (the
worst) on all four losses at that origin, instead of being dropped and
the others re-ranked. Rationale: a generator that cannot produce
scenarios has failed at that origin, and scoring it only where it
succeeds is survivorship bias. Record as "Amendment 1 (post-hoc)" in
docs/preregistration.md; regenerate the hash; commit separately. Report
shows both rulings side by side.

## 2. Remove the backcast leak
Compute the variance backcast from the training slice only and pass it
explicitly. If unsupported, keep current behaviour and the caveat.
Assert the filtered sigma at day 1 equals sigma^2_{T+1} from the
train-only fit exactly.

## 3. Quote the missing numbers in the report summary
Per model: persistence at each origin (GARCH family), flagging >= 1;
Diebold-Mariano p-values vs champion on nll and tail_crps, confirmation
origins; a one-line PIT verdict (U / hump / flat) from a chi-square
against uniform at 20 bins; the stress-origin rows.

## 4. Tag and push
`spec5-preregistered` on the pre-registration commit and `spec5-run` on
the results commit; push tags with the branch.
