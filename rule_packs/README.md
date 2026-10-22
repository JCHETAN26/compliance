# rule_packs/ — versioned YAML rule packs

| File | Status |
|---|---|
| `nda_standard.yaml` | **Phase 3.** The 17 ContractNLI hypotheses, verbatim, as `model_supported: true` per-document rules. Validated against the official dataset by `ml/tests/test_rule_pack.py`. |
| `vendor_confidentiality_review.yaml` | Not written yet. It belongs to Phase 4 of `docs/BUILD_HANDOFF.md`, together with the cross-document rules, the rule-pack tables and API. |

Only the YAML and its validation exist at this point. Rule packs are not yet stored in
the database or exposed through the API (Phase 4).

Each rule maps `contractnli_key` to the dataset hypothesis. A model-supported rule's
`hypothesis_text` must equal the training hypothesis exactly; a custom rule whose text
differs must be `model_supported: false` and always goes to the LLM (handoff §7.6).
