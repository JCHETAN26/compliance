# rule_packs/ — versioned YAML rule packs

| File | Status |
|---|---|
| `nda_standard.yaml` | **Phase 3.** The 17 ContractNLI hypotheses, verbatim, as `model_supported: true` per-document rules. Validated against the official dataset by `ml/tests/test_data.py`. No `compliant_statuses`: findings report the status only ("not assessed" for compliance). |
| `vendor_confidentiality_review.yaml` | **Phase 4.** The flagship pack: three verbatim ContractNLI rules (trained model), one custom rule (LLM), and three cross-document rules evaluated per vendor-register row (contract on file, required protections for high-access vendors, contract term covering the renewal date). |

Both packs are loaded into the database as read-only **built-in** packs when the API and the
worker start (idempotent; `compliance_core.services.rule_packs.sync_builtin_packs`). The
files ship inside the `compliance-core` wheel, so installed services read the same YAML.
Workspaces can import, copy and edit their own packs (UI **Rule packs**, or
`POST /api/v1/rule-packs/import`, `PUT /api/v1/rule-packs/{id}`). Every review run
snapshots the rules it checks, so later edits never change an existing run.

## Schema (validated strictly; unknown fields are errors)

- pack: `key`, `name`, `version` (x.y.z), `description`, optional `source` (provenance),
  optional `register.columns` (aliases for `vendor`, `data_access`, `contract_file`,
  `renewal_date`; `vendor` and `contract_file` are required when cross-document rules exist).
- rule: `key`, `title`, `description`, `hypothesis_text`, `scope`
  (`per_document` | `cross_document`), `applies_to_doc_types`, `severity`,
  `model_supported`, `contractnli_key`, `compliant_statuses`, `cross_document`, `remediation`.
- `cross_document.kind`: `contract_on_file` | `required_protections` (with
  `required_rules`) | `term_covers_date` (with `date_column`); optional `subject_filter`.

## Trained-model support cannot be claimed

`model_supported: true` is accepted only for a per-document rule whose `contractnli_key` is
one of the trained hypotheses **and** whose `hypothesis_text` equals that hypothesis
verbatim (the mapping is `nda_standard.yaml`). Custom or edited rules must use the LLM
(handoff section 7.6); the server rejects anything else with HTTP 422.

`compliant_statuses` lists the statuses that count as compliant. `needs_review` is never
compliant, and `not_mentioned` is compliant only if a pack explicitly says so.
`remediation` is the action text the report writer uses for non-compliant findings; the
report never generates remediation advice with a model.
