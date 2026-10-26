# Haiku LLM-baseline evaluation plan

Status (2026-10-25, final): **complete.** The user authorized the run after reviewing the
$2 limit and the transmission of 3128 public ContractNLI examples to Anthropic. The run and
the independent offline evaluation both exited 0. No further provider requests are allowed
or planned. The ledger, cached answers and run summary are in
`../kaggle-recovery/provider-haiku-baseline/`.
This document does not accept or close any phase.

## Final run (2026-10-26 00:23:57–01:25:09 UTC, exit 0)

Source: `../kaggle-recovery/provider-haiku-baseline/run-summary.json`. The run used a real
client, recorded `invariant_failed: null` and left 0 pending reservations.

| Split | Pairs answered | Errors | Usage-less attempts | Cost (USD) | Mean / max latency (s) | Wall (s) |
|---|---:|---:|---:|---:|---|---:|
| DEV | 1037 / 1037 | 0 | 0 | 0.126527 | 1.1905 / 3.4496 | 1234.9 |
| TEST | 2091 / 2091 | 0 | 0 | 0.241506 | 1.1648 / 8.785 | 2436.5 |

- Committed spend against the $2 cap was **$0.371269985**. That total includes the
  $0.0032375 Sonnet diagnostic. The cost is exact, not a lower bound.
- Tokens on TEST: input 2,019,504, cache read 1,260,873, cache write 0, output 53,893.
- Two invalid evidence ids were claimed on DEV and none on TEST; they are scored as false
  positives.

The independent label recomputation
(`../kaggle-recovery/logs/codex-haiku-complete-check.json`) matches the evaluation output:

| Split | Accuracy | Macro-F1 | F1 entail / contra / not_mentioned |
|---|---:|---:|---|
| DEV | 0.742527 | 0.691604 | 0.807 / 0.541 / 0.727 |
| TEST | 0.739359 | 0.703409 | 0.774 / 0.586 / 0.750 |

## Final evaluation (offline, no provider calls)

Output: `../kaggle-recovery/eval-haiku-final/results`, copied to `ml/eval/results/`. The
command and provenance are in `KAGGLE_LOCAL_VALIDATION.md`. TEST macro-F1:

- trained DeBERTa, full pipeline: 0.813778;
- the same weights on the BGE top 8: 0.769167;
- TF-IDF: 0.724374;
- Haiku: 0.703409.

Cascade:

- The DEV rule selected tau **0.3** (`provisional: false`, `llm_baseline_usable: true`;
  Haiku-only DEV accuracy 0.7425).
- Coverage is **1.0** on DEV and TEST. No pair is escalated to Haiku, so the cascade equals
  the trained model, with no accuracy improvement over it and $0 LLM cost.
- The cascade is not faster. The model's per-pair latency in `cascade.json` is 10.01 s, the
  Kaggle CPU bench's p50 for the original dynamic int8 file (since replaced, see
  `INT8_QUANTIZATION.md`); the served fp32 p50 is 13.36 s. Haiku averaged 1.165 s per
  TEST request (p50 1.164 s, p95 1.658 s) at $0.00196 per document.
- The Haiku caveat below still applies: tau is calibrated against Haiku, not Sonnet.

## DEV check (recorded while TEST was running; superseded by the final run above)

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

1. Done: the authorized `haiku_baseline.py run` completed at $0.3713 of the $2 cap. Keep its
   ledger and cached answers; do not delete or reset them.
2. Done: the offline evaluation, and the local fp32 artifact now serves its tau 0.3
   (installed and smoke-checked; see `KAGGLE_LOCAL_VALIDATION.md`).
3. Review of the results and the Haiku caveat above before any phase acceptance.
