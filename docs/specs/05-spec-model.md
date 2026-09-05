# SPEC 3 — Generative Model (GJR-GARCH + skewed-t)

## Goal
Fit a volatility-aware model to Brent log returns and simulate forward
paths. Clean fit->simulate split, fully seeded/reproducible. Consumes
`load_returns()` from SPEC 1.

## Model
GJR-GARCH(1,1,1), constant mean, skewed-t innovations.
    r_t   = mu + eps_t
    eps_t = sigma_t * z_t,   z_t ~ skew-t(nu, lambda)
    sigma_t^2 = omega + (alpha + gamma*I[eps_{t-1}<0]) eps_{t-1}^2
                + beta * sigma_{t-1}^2
Use the `arch` package (pin it). Fit by MLE.

## Module (src/brent_synth/model.py)

### Fit
    def fit(returns: pd.Series) -> ModelFit:
        """Fit GJR-GARCH(1,1,1)-skewt by MLE. Returns a ModelFit holding:
        params (mu, omega, alpha, gamma, beta, nu, lambda), loglik, aic,
        bic, the last conditional variance, the unconditional variance,
        and the fitted arch result object."""

- arch config: mean='Constant', vol='GARCH', p=1, o=1, q=1, dist='skewt'.
- Scale note: arch expects returns in percent. Multiply by 100 before
  fitting and divide simulated output by 100 on the way out, so the
  public interface is always in raw log-return units. Document this
  in a comment — it's a common silent bug.
- ModelFit is a dataclass. Store enough to simulate without refitting.

### Simulate
    def simulate(fit: ModelFit, horizon: int = 252, n_paths: int = 5000,
                 seed: int = 42, initial_var: str = "last") -> np.ndarray:
        """Simulate returns. Returns array shape (n_paths, horizon)
        in raw log-return units."""

- Seed the RNG explicitly (np.random default_rng(seed)); same seed ->
  identical output. This is a hard requirement.
- Return raw returns, NOT prices.

### Convenience
    def fit_and_simulate(returns, **kwargs) -> tuple[ModelFit, np.ndarray]

## Tests (tests/test_model.py)
- fit() returns finite params; beta and (alpha+beta) in (0,1); nu > 2.
- Persistence alpha + gamma/2 + beta < 1.
- simulate() returns shape (n_paths, horizon), all finite.
- Determinism: same seed -> identical; different seed -> different.
- Scale sanity: std of simulated returns within 3x of real.
- initial_var='last' vs 'unconditional' produce different day-1 variance.
- Estimator recovery: simulate a long path from KNOWN params, refit,
  check recovered params are close to the originals.

## Done when
`fit_and_simulate(load_returns())` returns a ModelFit and a (5000, 252)
array, all model tests pass including the round-trip recovery test.
