# Phase 0 independent review — pending issues

## Browser session-expiry check

Codex independently ran:

`env -u PYTHONPATH /tmp/cw-e2e/.venv/bin/python -I /tmp/cw-e2e/ui_check.py http://127.0.0.1:3300`

The walkthrough completed signup, workspace create, member management, rename,
delete, signout, and sign-in, then failed with a Playwright 30s timeout clicking
Create workspace during the mid-use expiry case. The button detached while the
click awaited stability. The script was also being edited by Claude during the
run (traceback source line no longer matched), so rerun a stable final version.

Do not dismiss the failure without checking whether auth Query invalidation or
focus refetch redirects before the intended mutation. Verify the desired UX and
adjust the test to assert it deterministically, without masking actual errors.
The acceptance criterion remains pending until independently rerun successfully.

## Environment constraints for next phase

- Free disk after Phase 0 build is only about 2.5 GB. Do not start large model
  downloads or delete unrelated Docker caches/images/volumes. Propose exact
  project-owned cache cleanup or an environment remedy if needed.
- User chose Colab for full GPU training and Anthropic API for real-provider
  application checks/baselines. Read docs/SUPERVISION.md.
- Runtime on this shared host uses WEB_PORT=3300; port 3000 belongs to another app.

## Resolution: browser check passed

After Claude finalized the test (including explicit page-load readiness), Codex
copied it to `/tmp/compliance-supervision/phase0-ui-verify.py` to prevent concurrent
edits, and ran:

`env -u PYTHONPATH /tmp/cw-e2e/.venv/bin/python -I /tmp/compliance-supervision/phase0-ui-verify.py http://127.0.0.1:3300`

Exit 0: `console/page errors: none`, `UI check passed`. The fixed script separately
checks expired-cookie reload and session loss before a client mutation. Codex also
visually inspected the full-page workspace-detail screenshot. The browser issue
is resolved; the pending section above is retained as the review record.
