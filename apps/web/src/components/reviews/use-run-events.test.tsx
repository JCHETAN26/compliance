import { act, render, screen } from "@testing-library/react";

import { json, mockFetch } from "@/test/utils";

import { event } from "./fixtures";
import { useRunEvents } from "./use-run-events";

class FakeEventSource {
  static CLOSED = 2;
  static instances: FakeEventSource[] = [];
  readyState = 1;
  listeners = new Map<string, ((e: MessageEvent<string>) => void)[]>();
  constructor(public url: string) {
    FakeEventSource.instances.push(this);
  }
  addEventListener(type: string, fn: (e: MessageEvent<string>) => void) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), fn]);
  }
  emit(type: string, data: unknown) {
    for (const fn of this.listeners.get(type) ?? []) fn({ data: JSON.stringify(data) } as MessageEvent<string>);
  }
  close() {
    this.readyState = FakeEventSource.CLOSED;
  }
}

function Probe({ runId, active }: { runId: string; active: boolean }) {
  const { events, status } = useRunEvents(runId, active);
  return (
    <div>
      <p data-testid="seqs">{events.map((e) => e.seq).join(",")}</p>
      <p data-testid="status">{status}</p>
    </div>
  );
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => (resolve = r));
  return { promise, resolve };
}

describe("useRunEvents", () => {
  beforeEach(() => {
    FakeEventSource.instances = [];
    vi.stubGlobal("EventSource", FakeEventSource);
  });
  afterEach(() => vi.unstubAllGlobals());

  it("resets events, cursor and status when the run changes, ignoring late old-run data", async () => {
    const oldHistory = deferred<Response>();
    mockFetch({
      "GET /api/v1/reviews/old/events/history": () => oldHistory.promise,
      "GET /api/v1/reviews/new/events/history": () =>
        json(200, { items: [event(1), event(2)], next_after: 2, has_more: false, run_status: "running" }),
    });
    const { rerender } = render(<Probe runId="old" active />);
    expect(FakeEventSource.instances[0]?.url).toContain("/reviews/old/events?after=0");
    await act(async () => FakeEventSource.instances[0]!.emit("agent_event", event(90)));
    expect(screen.getByTestId("seqs")).toHaveTextContent("90");

    rerender(<Probe runId="new" active />);
    expect(screen.getByTestId("seqs")).toHaveTextContent(/^$/);
    expect(FakeEventSource.instances[0]!.readyState).toBe(FakeEventSource.CLOSED);
    const fresh = FakeEventSource.instances[1]!;
    expect(fresh.url).toContain("/reviews/new/events?after=0"); // not the old cursor (90)
    await act(async () => {
      // Late answers for the old run arrive after the switch.
      oldHistory.resolve(json(200, { items: [event(95)], next_after: 95, has_more: false, run_status: "completed" }));
      FakeEventSource.instances[0]!.emit("agent_event", event(96));
    });
    expect(await screen.findByText("1,2")).toBeInTheDocument();
    await act(async () => fresh.emit("agent_event", event(3)));
    expect(screen.getByTestId("seqs")).toHaveTextContent("1,2,3");
  });

  it("reports a truthful status when the run stops", async () => {
    mockFetch({
      "GET /api/v1/reviews/r/events/history": () =>
        json(200, { items: [], next_after: 0, has_more: false, run_status: "running" }),
    });
    const { rerender } = render(<Probe runId="r" active />);
    await act(async () => FakeEventSource.instances[0]!.emit("open", {}));
    expect(screen.getByTestId("status")).toHaveTextContent("live");
    rerender(<Probe runId="r" active={false} />);
    expect(screen.getByTestId("status")).toHaveTextContent("idle");
    expect(FakeEventSource.instances[0]!.readyState).toBe(FakeEventSource.CLOSED);
  });
});
