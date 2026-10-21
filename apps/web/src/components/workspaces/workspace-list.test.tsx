import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import type { Workspace } from "@/lib/api/client";
import { json, mockFetch, renderWithQuery } from "@/test/utils";

import { WorkspaceList } from "./workspace-list";

const existing: Workspace = {
  id: "11111111-1111-1111-1111-111111111111",
  name: "Vendor review",
  owner_id: "u1",
  created_at: "2026-10-01T12:00:00Z",
  role: "reviewer",
  member_count: 3,
  document_count: 4,
};

describe("WorkspaceList", () => {
  it("shows an empty state when there are no workspaces", async () => {
    mockFetch({ "GET /api/v1/workspaces": () => json(200, []) });
    renderWithQuery(<WorkspaceList />);
    expect(await screen.findByText("No workspaces yet")).toBeInTheDocument();
  });

  it("renders workspace cards with role and member count", async () => {
    mockFetch({ "GET /api/v1/workspaces": () => json(200, [existing]) });
    renderWithQuery(<WorkspaceList />);
    const link = await screen.findByRole("link", { name: /Vendor review/ });
    expect(link).toHaveAttribute("href", `/workspaces/${existing.id}`);
    expect(within(link).getByText("Reviewer")).toBeInTheDocument();
    expect(within(link).getByText("3 members")).toBeInTheDocument();
    expect(within(link).getByText("4 documents")).toBeInTheDocument();
  });

  it("shows an error with a working retry", async () => {
    let attempts = 0;
    mockFetch({
      "GET /api/v1/workspaces": () => {
        attempts += 1;
        return attempts === 1
          ? json(500, { error: { code: "internal_error", message: "An unexpected error occurred." } })
          : json(200, [existing]);
      },
    });
    renderWithQuery(<WorkspaceList />);
    expect(await screen.findByText("Could not load workspaces")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("link", { name: /Vendor review/ })).toBeInTheDocument();
  });

  it("creates a workspace with the CSRF header and shows it", async () => {
    document.cookie = "cw_csrf=csrf-abc; path=/";
    const created: Workspace = {
      ...existing,
      id: "2",
      name: "Policy audit",
      role: "owner",
      member_count: 1,
      document_count: 0,
    };
    let listed: Workspace[] = [];
    const { calls } = mockFetch({
      "GET /api/v1/workspaces": () => json(200, listed),
      "POST /api/v1/workspaces": () => {
        listed = [created];
        return json(201, created);
      },
    });
    renderWithQuery(<WorkspaceList />);
    await screen.findByText("No workspaces yet");
    await userEvent.type(screen.getByLabelText("New workspace"), "  Policy audit ");
    await userEvent.click(screen.getByRole("button", { name: "Create workspace" }));
    expect(await screen.findByRole("link", { name: /Policy audit/ })).toBeInTheDocument();
    const post = calls.find((c) => c.method === "POST")!;
    expect(post.headers.get("X-CSRF-Token")).toBe("csrf-abc");
    expect(await post.json()).toEqual({ name: "Policy audit" });
    expect(screen.getByLabelText("New workspace")).toHaveValue("");
  });

  it("validates an empty name client-side and surfaces server validation", async () => {
    mockFetch({
      "GET /api/v1/workspaces": () => json(200, []),
      "POST /api/v1/workspaces": () =>
        json(422, {
          error: {
            code: "validation_error",
            message: "Request validation failed.",
            details: [{ loc: ["body", "name"], msg: "String should have at most 120 characters", type: "x" }],
          },
        }),
    });
    renderWithQuery(<WorkspaceList />);
    await screen.findByText("No workspaces yet");
    await userEvent.click(screen.getByRole("button", { name: "Create workspace" }));
    expect(screen.getByText("Enter a workspace name.")).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText("New workspace"), "x");
    await userEvent.click(screen.getByRole("button", { name: "Create workspace" }));
    expect(await screen.findByText("name: String should have at most 120 characters")).toBeInTheDocument();
  });
});
