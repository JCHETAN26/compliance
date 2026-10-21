import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import { json, mockFetch, renderWithQuery } from "@/test/utils";

import { AuthForm, validateCredentials } from "./auth-form";

const router = { replace: vi.fn(), refresh: vi.fn(), push: vi.fn() };
vi.mock("next/navigation", () => ({ useRouter: () => router }));

const session = {
  user: { id: "u1", email: "ada@example.com", created_at: "2026-10-21T00:00:00Z" },
  csrf_token: "c",
  expires_at: "2026-10-21T08:00:00Z",
};

beforeEach(() => {
  router.replace.mockReset();
  router.refresh.mockReset();
});

describe("validateCredentials", () => {
  it("requires a plausible email and a long enough sign-up password", () => {
    expect(validateCredentials("signup", "nope", "short")).toEqual({
      email: "Enter a valid email address.",
      password: "Use at least 8 characters.",
    });
    expect(validateCredentials("login", "a@b.co", "")).toEqual({ password: "Enter your password." });
    expect(validateCredentials("login", "a@b.co", "x")).toEqual({});
  });
});

describe("AuthForm", () => {
  it("shows field errors without calling the API", async () => {
    const { spy } = mockFetch({});
    renderWithQuery(<AuthForm mode="signup" />);
    await userEvent.click(screen.getByRole("button", { name: "Create account" }));
    expect(await screen.findByText("Enter a valid email address.")).toBeInTheDocument();
    expect(screen.getByText("Use at least 8 characters.")).toBeInTheDocument();
    expect(screen.getByLabelText("Email")).toHaveAttribute("aria-invalid", "true");
    expect(spy).not.toHaveBeenCalled();
  });

  it("registers and navigates to a safe next path", async () => {
    const { calls } = mockFetch({ "POST /api/v1/auth/register": () => json(201, session) });
    renderWithQuery(<AuthForm mode="signup" next="/workspaces/abc" />);
    await userEvent.type(screen.getByLabelText("Email"), "  ada@example.com ");
    await userEvent.type(screen.getByLabelText("Password"), "correct horse");
    await userEvent.click(screen.getByRole("button", { name: "Create account" }));
    await waitFor(() => expect(router.replace).toHaveBeenCalledWith("/workspaces/abc"));
    expect(await calls[0]!.json()).toEqual({ email: "ada@example.com", password: "correct horse" });
  });

  it("ignores an off-site next parameter", async () => {
    mockFetch({ "POST /api/v1/auth/login": () => json(200, session) });
    renderWithQuery(<AuthForm mode="login" next="https://evil.example" />);
    await userEvent.type(screen.getByLabelText("Email"), "ada@example.com");
    await userEvent.type(screen.getByLabelText("Password"), "whatever");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(router.replace).toHaveBeenCalledWith("/workspaces"));
  });

  it("shows the server's error message on failed sign-in", async () => {
    mockFetch({
      "POST /api/v1/auth/login": () =>
        json(401, { error: { code: "unauthenticated", message: "Invalid email or password." } }),
    });
    renderWithQuery(<AuthForm mode="login" />);
    await userEvent.type(screen.getByLabelText("Email"), "ada@example.com");
    await userEvent.type(screen.getByLabelText("Password"), "wrong");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByText("Invalid email or password.")).toBeInTheDocument();
    expect(router.replace).not.toHaveBeenCalled();
  });

  it("explains an expired session", () => {
    mockFetch({});
    renderWithQuery(<AuthForm mode="login" notice="Your session ended. Sign in again to continue." />);
    expect(screen.getByText("Your session ended. Sign in again to continue.")).toBeInTheDocument();
  });
});
