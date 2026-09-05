# brent-synth on AWS — deployment design

## What is being deployed

The pipeline is a batch job: fetch prices → compute returns → fit → simulate
→ validate → write reports, with an out-of-time backtest alongside. Cold, it
runs in under five minutes on one CPU with ~2 GB of memory; it has no
serving component, no state beyond files, and it must be reproducible — the
same code, data snapshot and seeds must produce byte-identical reports. The
design follows from that: **scheduled containers writing to object
storage**, and nothing more elaborate until the workload asks for it.

The one structural choice is to split the codebase's own `fit → simulate`
boundary across two runtimes. Fitting is the expensive, scheduled, gated
step; simulating from a stored fit takes seconds and is what a consumer
would call on demand.

## Architecture

```
EventBridge Scheduler (daily, after settlement)
        │
        ▼
ECS Fargate task ── image from ECR (pinned by digest)
   fetch → returns → fit → simulate → validate → backtest → reports
        │
        ├── S3  s3://brent-synth/data/snapshots/<date>/prices.parquet
        ├── S3  s3://brent-synth/fits/<run-id>/{params.json, manifest.json}
        ├── S3  s3://brent-synth/reports/<run-id>/{validation, model_comparison}.{html,md}
        └── S3  s3://brent-synth/backtest_cache/   (keyed by pre-registration hash)
        │
        ▼ (only if validation gate passes)
S3  s3://brent-synth/fits/current  →  copy of the promoted fit

API Gateway → Lambda `simulate` ── loads fits/current, returns (n_paths × horizon)
CloudFront → S3 reports/           ── read-only static hosting, IAM-restricted
CloudWatch + SNS                   ── task failure, gate failure, runtime > 10 min
```

| Component | Choice | Why |
|---|---|---|
| Compute (batch) | ECS Fargate, 2 vCPU / 4 GB, EventBridge-triggered | Minutes-long CPU job; no cluster to manage; Lambda's 15-minute cap would fit today but not the backtest at more origins or paths |
| Compute (on demand) | Lambda, container image | `simulate` from a stored fit is seconds and stateless; scales to zero |
| Storage | S3, versioned, one bucket, prefix per artefact type | Everything the pipeline produces is a file; versioning gives a free audit trail |
| Image | ECR, built in CI from the `uv.lock`, referenced by digest not tag | Reproducibility: the manifest records the digest, so a report can be regenerated from exactly the code that made it |
| Secrets | Secrets Manager for the data-vendor key (none needed for yfinance) | Never in the image or environment |
| Orchestration | Step Functions only if stages need independent retry; a single task otherwise | Do not add a workflow engine for a linear five-minute job |
| Observability | CloudWatch logs from the task; structured JSON; SNS alert on failure or gate rejection | The interesting failures are model failures, not infrastructure ones |
| Access | Private VPC subnets, no inbound; task role limited to its own S3 prefixes; reports behind CloudFront with signed URLs or IAM | Least privilege; nothing is public |

## Reproducibility and the validation gate

Every run writes a `manifest.json`: git SHA, image digest, data snapshot
hash, seeds, package versions, the pre-registration hash, and the pass/fail
counts from the validation report. **A fit is promoted to `fits/current`
only if the validation report's location test passes on every independent
statistic and the champion is unchanged by the backtest's confirmation
rule.** A run that fits fine but fails the gate is kept, alerted, and not
promoted — the thresholds the report justifies are the same thresholds the
deployment enforces. The backtest cache is keyed on the pre-registration
hash, so a change to the scoring plan invalidates it automatically, as in
the repository today.

Data snapshots are stored per run rather than fetched at simulate time:
yfinance is an unofficial source and revises history; a production
deployment would replace it with a licensed vendor behind the same
`load_prices` interface, and the snapshot layout is what makes that swap
invisible to everything downstream.

## Cost

Fargate at 2 vCPU / 4 GB for five minutes daily is roughly **$1/month**;
S3 for a few MB of artefacts per run is cents; Lambda on-demand simulation
is cents until it reaches millions of calls; CloudFront and CloudWatch are
negligible at this volume. Total well under **$10/month**. The cost that
matters is engineering time, which is why the design has no Kubernetes,
no workflow engine, no database and no queue.

## What changes when the workload changes

- **More models or origins in the backtest** → runtime grows linearly;
  Fargate task size goes up before anything else does. Past ~30 minutes,
  fan the origins out as parallel Fargate tasks under a Step Functions map
  state; the cache already keys per `(model, origin)`.
- **On-demand *fitting*** (a user supplies their own series) → the fit
  moves into a queued Fargate task with SQS, and the API returns a job id.
  Not built now because nobody has asked for it.
- **Conditional stress scenarios** (start paths from a stressed variance
  state, per model_comparison.md §9.4) → a parameter on the existing
  `simulate` Lambda, no infrastructure change.
- **Multiple instruments** → one scheduled task per ticker, same image,
  ticker as a parameter; the S3 layout gains a ticker prefix.

## Deliberately not built

No GPU (the models are MLE fits, not neural). No model-serving framework.
No feature store. No real-time ingestion: settlement prices arrive once a
day and the stress-testing use case is not latency-sensitive. Each of these
is a day of work and a recurring cost to solve a problem the workload does
not have.
