"use client";

import { useEffect, useRef, useState } from "react";

import type { AgentEvent } from "@/lib/api/client";
import { eventStreamUrl, fetchEventHistory } from "@/lib/api/reviews";

export const MAX_EVENTS = 3000;

/** Merge events by their per-run sequence number (dedupe, keep order, cap size). */
export function mergeEvents(existing: AgentEvent[], incoming: AgentEvent[]): AgentEvent[] {
  if (incoming.length === 0) return existing;
  const bySeq = new Map<number, AgentEvent>();
  for (const event of existing) bySeq.set(event.seq, event);
  for (const event of incoming) bySeq.set(event.seq, event);
  const merged = [...bySeq.values()].sort((a, b) => a.seq - b.seq);
  return merged.length > MAX_EVENTS ? merged.slice(merged.length - MAX_EVENTS) : merged;
}

export function lastSeq(events: AgentEvent[]): number {
  return events.at(-1)?.seq ?? 0;
}

export type StreamStatus = "loading" | "live" | "reconnecting" | "idle" | "error";

interface RunEvents {
  runId: string;
  events: AgentEvent[];
  status: StreamStatus;
}

/**
 * Agent events for a run: the persisted history first, then (while the run is active) a
 * Server-Sent Events stream from the last seen sequence number. The stream resumes from
 * the cursor after drops (EventSource sends Last-Event-ID; a closed stream is reopened
 * with ``?after=``), so no event is missed or duplicated.
 *
 * Everything is scoped to ``runId``: switching runs starts from an empty list and cursor
 * 0, and late responses or stream messages for a previous run are discarded.
 */
export function useRunEvents(runId: string, active: boolean, onEnd?: () => void) {
  const [state, setState] = useState<RunEvents>({ runId, events: [], status: "loading" });
  const cursor = useRef<{ runId: string; seq: number }>({ runId, seq: 0 });
  const onEndRef = useRef(onEnd);
  useEffect(() => {
    onEndRef.current = onEnd;
  }, [onEnd]);

  const view: RunEvents = state.runId === runId ? state : { runId, events: [], status: "loading" };

  useEffect(() => {
    let cancelled = false;
    // Changing the run resets the cursor; this effect runs before the stream effect.
    if (cursor.current.runId !== runId) cursor.current = { runId, seq: 0 };
    const update = (fn: (prev: RunEvents) => RunEvents) =>
      setState((prev) => fn(prev.runId === runId ? prev : { runId, events: [], status: "loading" }));
    (async () => {
      try {
        let after = 0;
        for (;;) {
          const page = await fetchEventHistory(runId, after, 1000);
          if (cancelled) return;
          update((prev) => {
            const events = mergeEvents(prev.events, page.items);
            if (cursor.current.runId === runId) {
              cursor.current.seq = Math.max(cursor.current.seq, lastSeq(events));
            }
            return { ...prev, events };
          });
          after = page.next_after;
          if (!page.has_more) break;
        }
        if (!cancelled) update((prev) => ({ ...prev, status: prev.status === "loading" ? "idle" : prev.status }));
      } catch {
        if (!cancelled) update((prev) => ({ ...prev, status: "error" }));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [runId]);

  useEffect(() => {
    if (!active || typeof EventSource === "undefined") return;
    let source: EventSource | null = null;
    let retry: ReturnType<typeof setTimeout> | null = null;
    let stopped = false;
    const update = (fn: (prev: RunEvents) => RunEvents) => {
      if (stopped) return; // a message from a stream we already closed
      setState((prev) => (prev.runId === runId ? fn(prev) : prev));
    };

    const open = () => {
      if (stopped) return;
      const after = cursor.current.runId === runId ? cursor.current.seq : 0;
      source = new EventSource(eventStreamUrl(runId, after));
      source.addEventListener("open", () => update((prev) => ({ ...prev, status: "live" })));
      source.addEventListener("agent_event", (message) => {
        if (stopped) return;
        const event = JSON.parse((message as MessageEvent<string>).data) as AgentEvent;
        if (cursor.current.runId === runId) cursor.current.seq = Math.max(cursor.current.seq, event.seq);
        update((prev) => ({ ...prev, events: mergeEvents(prev.events, [event]) }));
      });
      source.addEventListener("end", (message) => {
        source?.close();
        let reason = "stopped";
        try {
          reason = (JSON.parse((message as MessageEvent<string>).data) as { reason?: string }).reason ?? reason;
        } catch {
          /* keep the default */
        }
        if (reason === "reconnect") {
          // The server ends long streams so clients reconnect with their cursor.
          retry = setTimeout(open, 500);
          return;
        }
        update((prev) => ({ ...prev, status: "idle" }));
        onEndRef.current?.();
      });
      source.addEventListener("error", () => {
        update((prev) => ({ ...prev, status: "reconnecting" }));
        if (source?.readyState === EventSource.CLOSED) retry = setTimeout(open, 3000);
      });
    };
    open();
    return () => {
      // Leaving the active state (or the run): the stream is closed, so say so.
      setState((prev) => (prev.runId === runId && prev.status !== "error" ? { ...prev, status: "idle" } : prev));
      stopped = true;
      if (retry) clearTimeout(retry);
      source?.close();
    };
  }, [runId, active]);

  return { events: view.events, status: view.status };
}
