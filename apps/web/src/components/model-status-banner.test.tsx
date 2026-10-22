import { screen, waitFor } from "@testing-library/react";

import { json, mockFetch, renderWithQuery } from "@/test/utils";

import { ModelStatusBanner } from "./model-status-banner";

const status = (state: string, extra: Record<string, unknown> = {}) => ({
  rule_check: { state, message: `message for ${state}`, model_version: null, variant: null, ...extra },
});

describe("ModelStatusBanner", () => {
  it("announces untrained fallback mode", async () => {
    mockFetch({ "GET /api/v1/models/status": () => json(200, status("untrained_fallback")) });
    renderWithQuery(<ModelStatusBanner />);
    const banner = await screen.findByRole("status");
    expect(banner).toHaveTextContent("message for untrained_fallback");
    expect(banner).toHaveAttribute("data-state", "untrained_fallback");
  });

  it("marks a smoke artifact as not a trained model", async () => {
    mockFetch({
      "GET /api/v1/models/status": () => json(200, status("smoke", { model_version: "rule-check-smoke-1", variant: "int8" })),
    });
    renderWithQuery(<ModelStatusBanner />);
    const banner = await screen.findByRole("status");
    expect(banner).toHaveAttribute("data-state", "smoke");
    expect(banner).toHaveTextContent("message for smoke");
    expect(banner).toHaveTextContent("rule-check-smoke-1");
  });

  it("reports an unreachable model service", async () => {
    mockFetch({ "GET /api/v1/models/status": () => json(200, status("unreachable")) });
    renderWithQuery(<ModelStatusBanner />);
    const banner = await screen.findByRole("status");
    expect(banner).toHaveAttribute("data-state", "unreachable");
    expect(banner).toHaveTextContent("message for unreachable");
  });

  it("renders nothing when a trained model is serving", async () => {
    const { calls } = mockFetch({
      "GET /api/v1/models/status": () => json(200, status("trained", { model_version: "rule-check-x" })),
    });
    renderWithQuery(<ModelStatusBanner />);
    await waitFor(() => expect(calls).toHaveLength(1));
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });
});
