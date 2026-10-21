import { fireEvent, screen, waitFor } from "@testing-library/react";

import { renderWithQuery } from "@/test/utils";

import { UploadPanel } from "./upload-panel";

interface SentRequest {
  url: string;
  headers: Record<string, string>;
  body: FormData;
  xhr: FakeXHR;
}

const sent: SentRequest[] = [];

class FakeXHR {
  upload: { onprogress: ((e: { lengthComputable: boolean; loaded: number; total: number }) => void) | null } = {
    onprogress: null,
  };
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;
  status = 0;
  response: unknown = null;
  responseType = "";
  withCredentials = false;
  private url = "";
  private headers: Record<string, string> = {};
  open(_method: string, url: string) {
    this.url = url;
  }
  setRequestHeader(name: string, value: string) {
    this.headers[name] = value;
  }
  send(body: FormData) {
    sent.push({ url: this.url, headers: this.headers, body, xhr: this });
  }
  abort() {
    this.onabort?.();
  }
  respond(status: number, response: unknown) {
    this.upload.onprogress?.({ lengthComputable: true, loaded: 50, total: 100 });
    this.status = status;
    this.response = response;
    this.onload?.();
  }
}

beforeEach(() => {
  sent.length = 0;
  vi.stubGlobal("XMLHttpRequest", FakeXHR);
  document.cookie = "cw_csrf=csrf-xyz; path=/";
});

afterEach(() => vi.unstubAllGlobals());

function drop(files: File[]) {
  fireEvent.drop(screen.getByRole("button", { name: /Drop files here/ }), { dataTransfer: { files } });
}

const doc = (name: string) => ({ filename: name, outcome: "created", document: { id: "d1" }, error: null });

describe("UploadPanel", () => {
  it("uploads each file separately with the CSRF header and reports outcomes", async () => {
    renderWithQuery(<UploadPanel workspaceId="ws1" />);
    drop([new File(["%PDF-1.7"], "a.pdf"), new File(["x"], "b.exe"), new File(["a,b"], "c.csv")]);
    expect(await screen.findByText("Unsupported type. Upload PDF, DOCX, XLSX or CSV files.")).toBeInTheDocument();
    await waitFor(() => expect(sent).toHaveLength(2), { timeout: 2000 });
    expect(sent.every((r) => r.url === "/api/v1/workspaces/ws1/documents")).toBe(true);
    expect(sent.every((r) => r.headers["X-CSRF-Token"] === "csrf-xyz")).toBe(true);
    expect((sent[0]!.body.get("files") as File).name).toBe("a.pdf");

    sent[0]!.xhr.respond(200, { results: [doc("a.pdf")] });
    sent[1]!.xhr.respond(200, {
      results: [
        {
          filename: "c.csv",
          outcome: "rejected",
          document: null,
          error: { code: "unsupported_file", message: "The file content is not plain-text CSV." },
        },
      ],
    });
    expect(await screen.findByText("Uploaded - processing")).toBeInTheDocument();
    expect(await screen.findByText("The file content is not plain-text CSV.")).toBeInTheDocument();
  });

  it("shows duplicates and server errors per file", async () => {
    renderWithQuery(<UploadPanel workspaceId="ws1" />);
    drop([new File(["%PDF"], "dup.pdf"), new File(["%PDF"], "big.pdf")]);
    await waitFor(() => expect(sent).toHaveLength(2));
    sent[0]!.xhr.respond(200, { results: [{ ...doc("dup.pdf"), outcome: "duplicate" }] });
    sent[1]!.xhr.respond(413, { error: { code: "payload_too_large", message: "The upload exceeds the 200 MB request limit." } });
    expect(await screen.findByText("Identical file already in this workspace.")).toBeInTheDocument();
    expect(await screen.findByText("The upload exceeds the 200 MB request limit.")).toBeInTheDocument();
    expect(screen.getByText("Upload failed")).toBeInTheDocument();
  });

  it("is keyboard operable", () => {
    renderWithQuery(<UploadPanel workspaceId="ws1" />);
    const zone = screen.getByRole("button", { name: /Drop files here/ });
    const input = screen.getByLabelText("Choose files to upload") as HTMLInputElement;
    const click = vi.spyOn(input, "click");
    zone.focus();
    fireEvent.keyDown(zone, { key: "Enter" });
    expect(click).toHaveBeenCalled();
  });
});
