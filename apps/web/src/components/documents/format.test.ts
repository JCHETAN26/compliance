import { formatBytes, precheckFile, stageLabel } from "./format";

function file(name: string, size: number): File {
  return new File([new Uint8Array(size)], name);
}

describe("upload prechecks", () => {
  it("accepts supported extensions case-insensitively", () => {
    expect(precheckFile(file("Contract.PDF", 10))).toBeNull();
    expect(precheckFile(file("register.xlsx", 10))).toBeNull();
  });

  it("rejects unsupported, empty and oversized files", () => {
    expect(precheckFile(file("a.exe", 10))).toMatch(/Unsupported type/);
    expect(precheckFile(file("a.pdf", 0))).toMatch(/empty/);
    const big = { name: "a.pdf", size: 60 * 1024 * 1024 } as File;
    expect(precheckFile(big)).toMatch(/exceeds/);
  });

  it("formats sizes and stages", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(stageLabel({ status: "processing", stage: "embedding" } as never)).toBe("Embedding");
    expect(stageLabel({ status: "uploaded", stage: "queued" } as never)).toBe("Queued");
  });
});
