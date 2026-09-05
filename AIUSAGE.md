# AI usage

This document describes how AI tools were used to build this repository,
what I decided myself, how AI output was verified, and where it was wrong.
It is written to be checkable against the git history and the test suite.

## Tools and roles

Two AI systems were used, in distinct roles.

**Claude (chat) — design partner and spec author.** I used it to discuss
modelling choices, to write implementation specifications, to interpret
results, and to draft the prose sections of the two reports from the
numbers those reports generated. Every spec is a self-contained markdown
document that names interfaces, invariants, tests and a "done when"
condition; five were written (data, diagnostics, model, validation, and
the model ladder with temporal validation) plus three follow-ups.

**Claude Code — implementer.** I call this session "Brian". It only
implements: it is given a spec and writes the code and tests for it, and
it makes no modelling or methodological decisions. It implemented each
spec, wrote the tests the spec required and additional ones it judged
necessary, and returned a written report per spec listing status, test
counts, deviations from the spec, and defects found. Those reports were
the input to the next design discussion.

I sat between the two: I set the objectives, made every modelling and
methodological decision, reviewed each report, and decided what to keep,
amend or reject.

## What I decided

The decisions that shape the submission were mine, made in discussion
with the chat model but not delegated to it:

- **The model family.** GJR-GARCH with skewed-t innovations, chosen
  because each term maps to a diagnostic: leverage to the asymmetric tail
  indices, skewed-t to the skew and kurtosis, GARCH to the Ljung–Box
  result on squared returns.
- **Validation as the priority.** After the first working pipeline I
  redirected the project from a single-model, in-sample report to a
  pre-registered, out-of-time comparison of six candidates. My concern
  was that any model can match the marginal distribution of one series
  in sample, and that the question that matters — does the model cover a
  year it has not seen — is only answerable out of time. That reframing,
  and the three kinds of overfitting the pipeline defends against, are
  set out in `docs/` and were the design brief for SPEC 5.
- **Pre-registration and the post-hoc amendment.** The candidate list,
  origins, split and ranking rule were fixed and committed before the
  first out-of-time result. When the rule turned out not to handle a
  candidate that could not be fitted at some origins, I chose to amend
  it (failed candidates take the worst rank) rather than leave a
  survivorship-biased table, and required the amendment to be recorded
  and both rulings reported.
- **Keeping the champion.** The composite rule selected GJR-GARCH and
  confirmed it on held-out origins. The density scores and calibration
  diagnostics then favoured FIGARCH slightly. I kept the pre-registered
  selection and reported the tension, rather than switch models after
  seeing the confirmation results — which would have been the exact
  selection overfit the design exists to prevent.

## How AI output was verified

I did not take generated code or generated numbers on trust.

- **Ground-truth recovery tests.** Every estimator is tested on data
  simulated from known parameters: the diagnostics recover the tail
  index of a t(4) sample; the GJR fit recovers parameters from a
  20,000-point simulation of itself; the PIT of the true model passes a
  uniformity test.
- **Leakage test.** Every post-origin return is multiplied by ten and the
  backtest rerun; fitted parameters and simulated paths must be
  byte-identical. This is the test the temporal validation depends on.
- **Regression test.** The model-ladder wrapper's simulation must equal
  the original `model.simulate` output under `np.array_equal`.
- **Independent formulas.** CRPS is checked against the Gaussian closed
  form; Kupiec against an LR computed separately in the test.
- **Reading the code.** I read the implementation of `data.py`,
  `diagnostics.py`, `model.py` and `validation.py` and checked it against
  what the spec asked for, looking for the things that are silent when
  wrong: the percent/raw unit boundary in `model.py`, the per-path versus
  pooled statistics in `validation.py`, the adjacency handling in all
  three. This was done early. I did not read the later modules — the
  candidate ladder, `scoring.py`, `backtest.py` and the report
  generators — line by line, for lack of time. Those rest on the
  automated checks above rather than on my own reading, and that is a
  real limit on how much of this code I have personally verified.

218 tests pass at submission.

## Where the AI was wrong, and what I did about it

Listing these matters more than the list of what went right.

- **The GJR persistence formula.** The first spec wrote the stationarity
  condition as α + γ/2 + β, the textbook form. That is wrong for skewed
  innovations: the weight on γ is the partial second moment of the
  innovation below zero, 0.542 at the fitted skew, not 0.5. Using 0.5
  understates the long-run variance by 21%. Caught in review; the
  implementation carries the correct weight and a docstring explaining
  why.
- **Pooled synthetic statistics.** An early version of the validation
  pooled all 1.26 million synthetic days into one sample and compared it
  against bands built from 252-day windows. For anything nonlinear in the
  sample that is a category error — pooled excess kurtosis read 99.6
  where the per-path median was 2.0. Rewritten so both sides are
  computed per path at the same horizon.
- **A spec that named a non-existent argument.** SPEC 5 specified
  `search_rng` for `MarkovRegression.fit`; the parameter is `rng`.
  The implementer flagged it rather than silently substituting.
- **A spec that contradicted itself.** SPEC 5.1 specified a test that
  post-origin perturbations must not move density scores, then corrected
  itself mid-sentence — they legitimately must. The implementer built the
  corrected version and said so.
- **A leak that was real in principle and zero in effect.** The
  out-of-sample filter computed arch's variance backcast from the whole
  series, including the test window. Measured, its influence had decayed
  below float64 resolution before the test window began, so no reported
  number changed; it was nonetheless replaced with a train-only backcast
  so the property holds structurally rather than incidentally.
- **Two library-naming defects.** arch names the skew-t degrees of
  freedom `eta`; one wrapper assumed `nu`. `pd.qcut` on fewer than three
  distinct values crashes; the regime labeller was rewritten to
  rank-based terciles.
- **Claims marked for checking.** Drafted prose contained statements the
  chat model believed but could not verify without the data (which years
  had the largest single-day moves, whether the real ACF is still
  positive at lag 20). These were marked `[check]` in the draft and
  confirmed or corrected against a rerun of `diagnostics.run_all` before
  use. All four were resolved and three changed the prose: the
  autocorrelation of squared returns is still positive at lag 20
  (+0.082); the mean-excess function on losses rises rather than being
  flat (slope +0.27 over the top fifth of thresholds); the observation
  count fell from 4754 to 4753 and the Ljung–Box p-value from 1.7e−213
  to 1.9e−213 after the provisional-bar fix; and the gain tail's index
  of 7.9 was withdrawn entirely, because the significance gate added
  later found ξ/se = 1.25, short of the 95% mark. The corrections are
  listed in `docs/specs/25-report-6-packaging.md`.

## What the prose is

The interpretation sections of both reports were drafted by the chat
model from the numbers in the generated reports, in response to my
questions about what the results meant, and then edited by me. The
argument they make — that a GARCH(1,1) approximates the data's long
memory by integrating, and that this one mechanism produces every
mismatch the validation found — emerged from that discussion; I hold it
and can defend it. I have not presented the drafting as unaided.

## What was not done with AI

Repository setup, the choice of the 4-Xtra brief's Brent series, and the
decision of what to submit were done without AI assistance. So was the
design of the pipeline itself: the shape of the validation, the decision
to make it temporal rather than in-sample only, and the structure of the
out-of-time comparison were mine. The AI implemented that design; it did
not choose it.

An earlier draft of this section also claimed data ingestion and caching
were done without AI. That is not correct and I have removed it.
`data.py` — the fetch, the log-return calculation and the parquet cache —
was written by the implementer session against SPEC 1, and three
cache-lifecycle defects in it were fixed against a later review. What is
true is that I read that code manually and checked it was implemented
correctly, which is a different claim and the one I am making.

## Reproducibility of the AI trail

Specs and implementer reports are kept under `docs/specs/` in the order
they were written: six specs, three follow-ups, four defect reviews and
every report returned against them, numbered 01–25 with an index. The
pre-registration file carries a
hash of the scoring plan and is committed before the results commit;
tags `spec5-preregistered` and `spec5-run` mark the two. Git cannot prove
the ordering of commits made in one session, and I do not claim it does.
