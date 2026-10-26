# Haiku LLM-baseline evaluation plan

Status (2026-10-25): user authorized execution after reviewing the $2 limit and the
transmission of 3128 public ContractNLI examples to Anthropic. The Haiku run has started;
complete results and independent evaluation are pending. Progress, cached answers and
cumulative spending are saved in `../kaggle-recovery/provider-haiku-baseline/`.
This document does not accept or close any phase.

## Completed DEV check

All 1037 DEV requests answered, with no API errors or unknown usage. Recorded cost:
$0.126527; mean request latency: 1.1905 seconds. Independent label recomputation:

| Pipeline | DEV accuracy | DEV macro-F1 |
|---|---:|---:|
| Haiku, retrieved top-8 excerpts | 0.742527 | 0.691604 |
| Trained DeBERTa, full pipeline, selected k=4 | 0.848602 | 0.802028 |

The input scopes differ, so these are pipeline comparisons. Independent records:
`../kaggle-recovery/logs/codex-haiku-dev-check.json` and
`../kaggle-recovery/logs/codex-trained-dev-check.json`. TEST and the final cascade
evaluation remain pending. Do not assume Haiku escalation improves the trained pipeline.

## Scope

One zero-shot LLM baseline run over all 1037 DEV + 2091 TEST ContractNLI pairs, on
`claude-haiku-5-5` with effort `low`, `max_tokens` 4096 and the official Anthropic endpoint.
It uses the production code path (`ml.baselines.llm.run_pairs` / `call` / `save_system`)
with an injected budget-enforcing client. SDK `max_retries=0`, project `MAX_ATTEMPTS = 1`
and concurrency 1. The prepared runner, offline tests (10 passing checks), `plan.json` and
commands are in `../kaggle-recovery/provider-haiku-baseline/`. The Sonnet runner
(`provider-baseline/`, $30 cap) and the one-request diagnostic are unchanged and still
available. Production `DEFAULT_MODEL` stays `claude-sonnet-5-5`; the request hash for Haiku
is `0226cd53…`.

## Why the baseline is needed

`ml.eval.evaluate` selects a final cascade tau only against a usable LLM baseline: complete,
non-partial DEV and TEST predictions, each with at least 95% of pairs answered. The rule is
the smallest tau whose DEV cascade accuracy is within tolerance of the LLM-only DEV
accuracy. Without that baseline:

- tau is provisional (0.83 in the 2026-10-25 dry check) and `llm_baseline_usable` is false;
- `recover_fp32 --params` serves no tau, so the model-service returns `escalate: null`;
- the baseline comparison has no LLM row.

The old `llm_zero_shot` files in the Kaggle bundle answered no pairs and are excluded.

Haiku's accuracy may differ from Sonnet's. A tau selected
against Haiku's DEV accuracy calibrates escalation relative to Haiku, not Sonnet. The
reported LLM row, its cost and its latency describe Haiku only. If Haiku's accuracy is
low, the selected tau will be correspondingly permissive. Whether that is acceptable is a
review decision, made after results exist.

## Budget

- Hard cap: **$2, cumulative, never reset**. The prior Sonnet diagnostic ($0.0032375) counts
  against the cap as committed spend. The runner refuses while a Sonnet baseline ledger exists.
- Prices are from `ml/baselines/anthropic_pricing.yaml`, per MTok: input $0.10, output $0.50,
  5-minute cache write $0.125, cache read $0.01. They match the official pricing page,
  re-checked 2026-10-25, for prompts up to 100k tokens. The runner refuses any request above
  100k tokens.
- Before each call, a worst-case reservation of at most $0.0040 is fsynced to an append-only
  ledger. Each call is settled to its reported usage. A failed or interrupted call is charged
  its full reservation.
- Rough expected cost is $0.4–1.3, which is not a bound. If Haiku's thinking output averages
  more than about 1,000 tokens per request, the cap stops the run and no predictions are
  written for the unfinished split.

## Data transmitted to Anthropic

Running the evaluation sends Anthropic, for each of the 3128 pairs, the production system
prompt and a user prompt. The user prompt holds the hypothesis text and the top-8 retrieved
excerpts from that NDA. These are public ContractNLI NDA excerpts and hypotheses. No
customer documents, keys or other data are sent. The user should confirm this transmission
before authorizing the run.

## Pending

1. Complete the authorized `haiku_baseline.py run` under its existing cumulative $2 cap
   (see the runner README). Do not delete or reset its ledger or cached answers.
2. After a complete authorized run: continue with offline evaluation (no provider calls),
   review the DEV-selected tau, then update and verify the local fp32 artifact if valid.
3. Review of the results and the Haiku caveat above before any phase acceptance.
