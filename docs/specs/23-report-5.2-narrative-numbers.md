# Report — SPEC 5.2

## Status
All four complete. 216 tests passing (7 new). No change to the
pre-registration hash.

## 1. Which loss the champion won on
On selection, gjr_skewt wins three of four including both density losses.
**On confirmation it wins neither density loss** — FIGARCH takes NLL
(1.833 vs 2.000) and tail CRPS (2.000 vs 2.500) — and holds the composite
entirely on exceedance error (1.917 vs 2.167) and path coverage
(2.333 vs 2.833).

That reconciles the SPEC 5.1 puzzle: the Diebold-Mariano statistics were
negative for FIGARCH because FIGARCH genuinely wins those. The champion
survives confirmation as a *risk* model, not a density model.

## 2. PIT — this softens my SPEC 5.1 caveat
Split by regime, the champion's "significantly under-confident" verdict
comes almost entirely from calm years: hump in calm (p 0.0001), **flat in
normal (0.099) and stress (0.269)**.

The split also exposes a static-width failure invisible in the pooled
view: `iid_t` and `ms_variance` flip from hump in calm to **U in stress**.
`iid_t`'s stress max-bin deviation is 0.065 — a bin holding 11.5% where it
should hold 5%.

On common days (9 origins), FIGARCH reads flat (0.122) and the champion
still reads humped (0.008).

Effect sizes earn their place: a seeded uniform draw trips the chi-square
at p = 0.040 with a KS distance of 0.007 — a textbook 5% false positive.

## 3 and 4
Persistence table with `non_stationary` column, nu and gamma alongside;
the four loss values at every stress origin with 2019 pinned in.
