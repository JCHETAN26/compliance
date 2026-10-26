import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { WorkspaceShell } from "@/components/workspaces/workspace-detail";
import { json, mockFetch, renderWithQuery } from "@/test/utils";

import { event, FINDINGS, report, run, TASKS, WORKSPACE } from "./fixtures";
import { totalCount } from "./labels";
import { citationHref, EMPTY_FILTERS, filterFindings, ReportView } from "./report-view";
import { bumpPatch, ModelSupportBadge } from "./rule-packs-page";
import { describeEvent, ProgressGrid, RunView, taskLabels } from "./run-view";
import { lastSeq, MAX_EVENTS, mergeEvents } from "./use-run-events";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), refresh: vi.fn() }),
  usePathname: () => "/workspaces/ws1/reviews/r1",
  useSearchParams: () => new URLSearchParams(),
}));

describe("event merging", () => {
  it("dedupes by sequence number and keeps order", () => {
    const merged = mergeEvents([event(1), event(3)], [event(2), event(3, { event_type: "error" }), event(4)]);
    expect(merged.map((e) => e.seq)).toEqual([1, 2, 3, 4]);
    expect(merged[2]?.event_type).toBe("error");
    expect(lastSeq(merged)).toBe(4);
    expect(lastSeq([])).toBe(0);
  });

  it("caps the buffer to the newest events", () => {
    const many = Array.from({ length: MAX_EVENTS + 10 }, (_, i) => event(i + 1));
    const merged = mergeEvents([], many);
    expect(merged).toHaveLength(MAX_EVENTS);
    expect(merged[0]?.seq).toBe(11);
  });

  it("describes events with readable labels and keeps raw ids out of the main text", () => {
    const labels = taskLabels(TASKS);
    const tokens = describeEvent(
      event(5, {
        event_type: "tokens",
        payload: { task_key: "a", tool: "llm.rule_verdict", input_tokens: 10, output_tokens: 2, cost_usd: 0, usage_source: "mock_estimate" },
      }),
      labels,
    );
    expect(tokens.text).toBe("Return or destruction — acme_nda.pdf · LLM verdict · tokens 10+2 · cost $0 · (mock estimate)");
    expect(tokens.ids).toBe("task_key: a");
    const verified = describeEvent(
      event(6, { agent: "citation_verifier", payload: { task_key: "cd|x|row:3f1d2c4e-1111-2222-3333-444455556666", finding_id: "f9", status: "needs_review", verified: false } }),
      labels,
    );
    expect(verified.text).toBe("a check · needs review · not verified");
    expect(verified.text).not.toMatch(/row:|f9/);
    expect(verified.ids).toContain("finding_id: f9");
  });
});

describe("progress grid", () => {
  it("renders one accessible cell per rule x subject with its state", () => {
    renderWithQuery(<ProgressGrid tasks={TASKS} />);
    expect(screen.getByLabelText("Return or destruction, acme_nda.pdf: satisfied")).toHaveTextContent("✓");
    expect(screen.getByLabelText("Notice on compelled disclosure, acme_nda.pdf: running")).toBeInTheDocument();
    expect(screen.getByLabelText("Return or destruction, globex_nda.docx: needs review")).toHaveTextContent("?");
    expect(screen.getByLabelText("not applicable")).toBeInTheDocument();
  });

  it("reads counts from run totals", () => {
    expect(totalCount({ findings_by_status: { a: 2, b: 3 } }, "findings_by_status")).toBe(5);
    expect(totalCount({ findings_by_status: { a: 2 } }, "findings_by_status", "b")).toBe(0);
    expect(totalCount({}, "x")).toBe(0);
  });
});

describe("report helpers", () => {
  it("filters and orders findings", () => {
    expect(filterFindings(FINDINGS, EMPTY_FILTERS).map((f) => f.id)).toEqual(["f1", "f3", "f2"]);
    expect(filterFindings(FINDINGS, { ...EMPTY_FILTERS, verifiedOnly: true }).map((f) => f.id)).toEqual(["f1", "f3"]);
    expect(filterFindings(FINDINGS, { ...EMPTY_FILTERS, status: "needs_review" }).map((f) => f.id)).toEqual(["f2"]);
    expect(filterFindings(FINDINGS, { ...EMPTY_FILTERS, decidedBy: "system" }).map((f) => f.id)).toEqual(["f3"]);
    expect(filterFindings(FINDINGS, { ...EMPTY_FILTERS, text: "umbrella" }).map((f) => f.id)).toEqual(["f3"]);
  });

  it("links citations to the highlighted viewer, never for deleted sources", () => {
    const [first, , third] = FINDINGS;
    expect(citationHref("ws1", first!.citations[0]!)).toBe("/workspaces/ws1/documents/d1?citation=c1");
    expect(citationHref("ws1", third!.citations[0]!)).toBeNull();
  });

  it("bumps rule pack versions and labels model support", () => {
    expect(bumpPatch("1.0.9")).toBe("1.0.10");
    expect(bumpPatch("x")).toBe("x");
    renderWithQuery(
      <>
        <ModelSupportBadge rule={{ model_supported: true, scope: "per_document", contractnli_key: "nda-16" }} />
        <ModelSupportBadge rule={{ model_supported: false, scope: "per_document", contractnli_key: null }} />
        <ModelSupportBadge rule={{ model_supported: false, scope: "cross_document", contractnli_key: null }} />
      </>,
    );
    expect(screen.getByText("Trained model")).toBeInTheDocument();
    expect(screen.getByText("LLM")).toBeInTheDocument();
    expect(screen.getByText("Cross-document")).toBeInTheDocument();
  });
});

function shell(ui: React.ReactElement, role: "owner" | "viewer" = "owner", extra: Record<string, () => Response> = {}) {
  const fetches = mockFetch({
    "GET /api/v1/workspaces/ws1": () => json(200, { ...WORKSPACE, role }),
    "GET /api/v1/settings/llm": () =>
      json(200, { deployment_provider: "mock", deployment_model: "mock-deterministic-v1", providers: [], rule_check_backend: "mock", mock_only: true }),
    ...extra,
  });
  const result = renderWithQuery(<WorkspaceShell workspaceId="ws1">{ui}</WorkspaceShell>);
  return { ...result, ...fetches };
}

describe("ReportView", () => {
  it("shows the summary, mock banner, filters and a deleted-source citation without a link", async () => {
    shell(<ReportView runId="r1" />, "owner", { "GET /api/v1/reviews/r1/report": () => json(200, report()) });
    expect(await screen.findByText(/1 finding is non-compliant/)).toBeInTheDocument();
    expect(await screen.findByText("Offline mock mode")).toBeInTheDocument();
    const row = screen.getByText("Umbrella GmbH").closest("tr")!;
    await userEvent.click(within(row).getByRole("button", { name: "Show details" }));
    expect(screen.getByText("source deleted")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Open in document" })).not.toBeInTheDocument();
    expect(screen.getByText(/register row:/i)).toBeInTheDocument();
  });

  it("records an audited decision with a required comment", async () => {
    const { calls } = shell(<ReportView runId="r1" />, "owner", {
      "GET /api/v1/reviews/r1/report": () => json(200, report()),
      "POST /api/v1/findings/f2/override": () => json(200, { ...FINDINGS[1], status: "violated", decided_by: "human" }),
    });
    const row = (await screen.findByText("globex_nda.docx")).closest("tr")!;
    await userEvent.click(within(row).getByRole("button", { name: "Show details" }));
    await userEvent.click(screen.getByRole("button", { name: "Record decision" }));
    const save = screen.getByRole("button", { name: "Save decision" });
    expect(save).toBeDisabled();
    await userEvent.selectOptions(screen.getByLabelText("Decision"), "violated");
    await userEvent.type(screen.getByLabelText("Comment (required)"), "No notice clause in section 2.");
    await userEvent.click(save);
    const post = calls.find((c) => c.method === "POST");
    expect(post).toBeDefined();
    expect(await post!.json()).toEqual({ status: "violated", comment: "No notice clause in section 2." });
  });

  it("does not offer decisions to viewers", async () => {
    shell(<ReportView runId="r1" />, "viewer", { "GET /api/v1/reviews/r1/report": () => json(200, report()) });
    const row = (await screen.findByText("globex_nda.docx")).closest("tr")!;
    await userEvent.click(within(row).getByRole("button", { name: "Show details" }));
    expect(screen.queryByRole("button", { name: "Record decision" })).not.toBeInTheDocument();
  });
});

describe("RunView", () => {
  it("shows counters, the grid, the event history and the awaiting-review prompt", async () => {
    shell(<RunView runId="r1" />, "owner", {
      "GET /api/v1/reviews/r1": () => json(200, run({ status: "awaiting_review" })),
      "GET /api/v1/reviews/r1/tasks": () => json(200, TASKS),
      "GET /api/v1/reviews/r1/events/history": () =>
        json(200, { items: [event(1, { agent: "planner" }), event(2, { event_type: "error", payload: { code: "budget_exhausted" } })], next_after: 2, has_more: false, run_status: "awaiting_review" }),
    });
    expect(await screen.findByText("Waiting for human review")).toBeInTheDocument();
    expect(screen.getByText("1,234 / 400,000")).toBeInTheDocument();
    expect(screen.getByText("mock estimate")).toBeInTheDocument();
    expect(await screen.findByText("code budget_exhausted")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Resume" })).toBeInTheDocument();
    expect(screen.getByLabelText("Return or destruction, acme_nda.pdf: satisfied")).toBeInTheDocument();
  });
});
