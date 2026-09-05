# SPEC 2 — Statistical Diagnostics

## Goal
Compute the diagnostics that characterise Brent log returns and justify
the model choice. Pure computation: return numbers, no plotting here.
Consumes `load_returns()` from SPEC 1.

## Module (src/brent_synth/diagnostics.py)

### 1. Empirical moments
    def moments(returns: pd.Series) -> dict:
        """Return {'mean','std','skew','kurtosis','excess_kurtosis',
                   'annualised_vol','n_obs'}."""
- Use sample skew and Fisher (excess) kurtosis.
- annualised_vol = std * sqrt(252).

### 2. Autocorrelation
    def acf_returns(returns: pd.Series, nlags: int = 40) -> np.ndarray:
        """ACF of r_t. Expected ~0 at all lags."""

    def acf_squared(returns: pd.Series, nlags: int = 40) -> np.ndarray:
        """ACF of r_t^2. Expected positive, slow decay (clustering)."""
- Return the ACF values (lag 0..nlags). Use statsmodels acf.
- Also return the Ljung-Box p-value on r_t^2 at lag 10 as a
  clustering significance check:
    def ljung_box_squared(returns, lags=10) -> float: ...

### 3. Tail diagnostics
    def hill_estimator(returns: pd.Series, tail: str = "left") -> dict:
        """Hill tail-index estimates across a range of k.
        Return {'k': np.ndarray, 'alpha': np.ndarray} for a Hill plot.
        tail='left' uses losses (negative returns, sign-flipped);
        tail='right' uses gains."""

    def pot_gpd_fit(returns: pd.Series, tail: str = "left",
                    threshold_quantile: float = 0.95) -> dict:
        """Peaks-over-threshold GPD fit on exceedances.
        Return {'threshold','xi','beta','n_exceedances','tail_index'}.
        tail_index = 1/xi when xi>0."""

    def mean_excess(returns: pd.Series, tail: str = "left") -> dict:
        """Mean-excess function for the mean-excess plot.
        Return {'thresholds': np.ndarray, 'mean_excess': np.ndarray}."""

- Run Hill/POT/mean-excess on the LEFT tail (losses) primarily.
- Left tail = work with -returns where return < 0, standard convention.
- Use scipy.stats.genpareto for the GPD fit.

### 4. Convenience
    def run_all(returns: pd.Series) -> dict:
        """Run every diagnostic, return one nested dict of results.
        This is what the report layer will consume."""

## Tests (tests/test_diagnostics.py)
- moments() returns all keys; excess_kurtosis > 0 for real data.
- acf_returns lag-1 is small (|.| < 0.1); acf_squared lag-1 clearly
  larger than acf_returns lag-1.
- ljung_box_squared p-value < 0.05 (clustering is significant).
- hill_estimator returns matching-length k and alpha arrays, alpha > 0.
- pot_gpd_fit returns xi; tail_index > 0.
- Cross-check: on a synthetic Student-t(df=4) sample, hill alpha
  recovers ~4 within tolerance (validates the estimator itself).

## Notes
- No matplotlib in this module. Plots belong to the report layer,
  which will consume run_all() output.
- Keep return types plain (dict / np.ndarray / float) so they
  serialise cleanly for the report.
- The t(df=4) recovery test matters: it proves the Hill code is
  correct, not just that it runs.

## Done when
`run_all(load_returns())` returns a populated results dict and all
tests pass, including the t(df=4) estimator-recovery check.
