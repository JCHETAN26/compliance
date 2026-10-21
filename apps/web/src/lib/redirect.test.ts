import { safeNextPath } from "./redirect";

describe("safeNextPath", () => {
  it.each([
    ["/workspaces/abc", "/workspaces/abc"],
    ["/workspaces?tab=members", "/workspaces?tab=members"],
  ])("keeps same-origin path %s", (input, expected) => {
    expect(safeNextPath(input)).toBe(expected);
  });

  it.each([
    [null],
    [undefined],
    [""],
    ["https://evil.example/phish"],
    ["//evil.example"],
    ["/\\evil.example"],
    ["javascript:alert(1)"],
    ["/login"],
    ["/signup?next=/x"],
    ["/x\u0000y"],
    [`/${"a".repeat(600)}`],
  ])("falls back for unsafe value %s", (input) => {
    expect(safeNextPath(input)).toBe("/workspaces");
  });
});
