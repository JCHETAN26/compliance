import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import type { ApiToken, ApiTokenCreated } from "@/lib/api/client";
import { json, mockFetch, renderWithQuery } from "@/test/utils";

import { ApiTokensPanel, mcpConfigSnippets } from "./api-tokens-panel";

const SECRET = "cwmcp_" + "s".repeat(43);

function token(overrides: Partial<ApiToken> = {}): ApiToken {
  return {
    id: "t1",
    name: "Laptop",
    prefix: "cwmcp_abcdefgh",
    scopes: ["read"],
    workspace_ids: null,
    created_at: "2026-10-26T10:00:00Z",
    expires_at: "2027-01-24T10:00:00Z",
    last_used_at: null,
    revoked_at: null,
    status: "active",
    ...overrides,
  };
}

const WORKSPACES = [
  { id: "ws1", name: "Vendors", owner_id: "u1", created_at: "2026-10-01T00:00:00Z", role: "owner", member_count: 1, document_count: 3 },
  { id: "ws2", name: "Archive", owner_id: "u2", created_at: "2026-10-01T00:00:00Z", role: "viewer", member_count: 2, document_count: 0 },
];

describe("API tokens panel", () => {
  it("shows the empty state, then creates a read-only 90-day token and reveals it once", async () => {
    let listed: ApiToken[] = [];
    const created: ApiTokenCreated = { ...token({ name: "Desktop" }), token: SECRET };
    const { calls } = mockFetch({
      "GET /api/v1/settings/api-tokens": () => json(200, listed),
      "GET /api/v1/workspaces": () => json(200, WORKSPACES),
      "POST /api/v1/settings/api-tokens": () => {
        listed = [token({ name: "Desktop" })];
        return json(201, created);
      },
    });
    renderWithQuery(<ApiTokensPanel />);
    expect(await screen.findByText("You have no API tokens yet.")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /read/ })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: /read/ })).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: /review/ })).not.toBeChecked();
    expect(screen.getByLabelText("Expires after")).toHaveValue("90");

    await userEvent.type(screen.getByLabelText("Token name"), "Desktop");
    await userEvent.click(screen.getByRole("button", { name: "Create token" }));

    expect(await screen.findByTestId("new-token")).toHaveTextContent(SECRET);
    expect(screen.getByText(/it will not be shown again/)).toBeInTheDocument();
    const post = calls.find((c) => c.method === "POST")!;
    expect(await post.json()).toEqual({ name: "Desktop", scopes: ["read"], expires_in_days: 90, workspace_ids: null });

    const table = await screen.findByRole("table", { name: "Your API tokens" });
    expect(within(table).getByText("Desktop")).toBeInTheDocument();
    expect(within(table).queryByText(SECRET)).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: "I have copied it" }));
    expect(screen.queryByTestId("new-token")).toBeNull();
    expect(screen.queryByText(SECRET)).toBeNull();
  });

  it("sends opted-in scopes and the workspace allowlist", async () => {
    const { calls } = mockFetch({
      "GET /api/v1/settings/api-tokens": () => json(200, []),
      "GET /api/v1/workspaces": () => json(200, WORKSPACES),
      "POST /api/v1/settings/api-tokens": () => json(201, { ...token(), token: SECRET }),
    });
    renderWithQuery(<ApiTokensPanel />);
    await screen.findByText("You have no API tokens yet.");
    await userEvent.type(screen.getByLabelText("Token name"), "Reviewer bot");
    await userEvent.click(screen.getByRole("checkbox", { name: /review/ }));
    await userEvent.selectOptions(screen.getByLabelText("Expires after"), "7");
    await userEvent.click(screen.getByRole("checkbox", { name: /Only these workspaces/ }));
    const create = screen.getByRole("button", { name: "Create token" });
    expect(create).toBeDisabled(); // restricted to nothing yet
    await userEvent.click(screen.getByRole("checkbox", { name: /Vendors/ }));
    await userEvent.click(create);
    await screen.findByTestId("new-token");
    const post = calls.find((c) => c.method === "POST")!;
    expect(await post.json()).toEqual({
      name: "Reviewer bot",
      scopes: ["read", "review"],
      expires_in_days: 7,
      workspace_ids: ["ws1"],
    });
  });

  it("revokes after confirmation and shows inactive tokens without actions", async () => {
    let listed = [
      token(),
      token({ id: "t2", name: "Old", status: "expired" }),
      token({ id: "t3", name: "Gone", status: "revoked", workspace_ids: ["ws9"] }),
    ];
    const { calls } = mockFetch({
      "GET /api/v1/settings/api-tokens": () => json(200, listed),
      "GET /api/v1/workspaces": () => json(200, WORKSPACES),
      "DELETE /api/v1/settings/api-tokens/t1": () => {
        listed = [token({ status: "revoked" }), ...listed.slice(1)];
        return json(200, listed[0]);
      },
    });
    renderWithQuery(<ApiTokensPanel />);
    const table = await screen.findByRole("table", { name: "Your API tokens" });
    expect(within(table).getAllByRole("button", { name: "Revoke" })).toHaveLength(1);
    expect(within(table).getByText("a workspace you left")).toBeInTheDocument();
    await userEvent.click(within(table).getByRole("button", { name: "Revoke" }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText(/stop working on their next request/)).toBeInTheDocument();
    await userEvent.click(within(dialog).getByRole("button", { name: "Revoke token" }));
    await waitFor(() => expect(within(table).queryAllByRole("button", { name: "Revoke" })).toHaveLength(0));
    expect(calls.some((c) => c.method === "DELETE")).toBe(true);
  });

  it("shows a load error with retry", async () => {
    mockFetch({
      "GET /api/v1/settings/api-tokens": () => json(500, { error: { code: "internal_error", message: "Boom." } }),
      "GET /api/v1/workspaces": () => json(200, []),
    });
    renderWithQuery(<ApiTokensPanel />);
    expect(await screen.findByText("Could not load API tokens")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("builds client configs that keep the token out of URLs and command lines", () => {
    const snippets = mcpConfigSnippets(SECRET);
    const docker = JSON.parse(snippets.docker);
    const server = docker.mcpServers["compliance-workspace"];
    expect(server.command).toBe("docker");
    expect(server.args).toEqual([
      "exec", "-i", "-e", "COMPLIANCE_MCP_TOKEN", "compliance-mcp-server-1",
      "python", "-m", "compliance_mcp_server", "--transport", "stdio",
    ]);
    expect(server.args.join(" ")).not.toContain(SECRET);
    expect(server.env).toEqual({ COMPLIANCE_MCP_TOKEN: SECRET });
    expect(snippets.ssh).not.toContain(SECRET);
    expect(snippets.http).toContain(`Authorization: Bearer ${SECRET}`);
    expect(snippets.http).toContain("http://127.0.0.1:8020/mcp ");
  });
});
