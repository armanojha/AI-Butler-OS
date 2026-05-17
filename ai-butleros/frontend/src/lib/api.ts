// src/lib/api.ts

const BASE_URL = "http://localhost:8000/api/v1";

// ── Types ──────────────────────────────────────────────────────────────────────

export interface InteractResponse {
  status: string;
  trace_id: string;
}

export interface TraceRow {
  id: string;
  timestamp: string;
  agent_name: string;
  step_name: string;
  input_data: Record<string, unknown> | null;
  output_data: Record<string, unknown> | null;
  monologue: string | null;
}

export interface TraceResponse {
  trace_id: string;
  completed: boolean;
  rows: TraceRow[];
}

// Discriminated union — callers can switch on `.status` without try/catch
export type TraceResult =
  | { status: "queued" }
  | { status: "ok"; data: TraceResponse }
  | { status: "error"; message: string };

// ── Wrappers ───────────────────────────────────────────────────────────────────

/**
 * POST /api/v1/interact
 * Enqueues a user message and returns the trace_id immediately.
 * Throws on network error so the caller can surface a toast.
 */
export async function interact(input: string): Promise<InteractResponse> {
  const res = await fetch(`${BASE_URL}/interact`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`interact ${res.status}: ${text}`);
  }

  return res.json() as Promise<InteractResponse>;
}

/**
 * GET /api/v1/trace/{trace_id}
 * Never throws — returns a discriminated union so the polling loop
 * can handle all states without try/catch at the call site.
 *
 *  "queued"  → 404, event not yet picked up by worker
 *  "ok"      → trace rows present; check .data.completed for done state
 *  "error"   → unexpected HTTP status or network failure
 */
export async function pollTrace(traceId: string): Promise<TraceResult> {
  try {
    const res = await fetch(`${BASE_URL}/trace/${traceId}`);

    if (res.status === 404) return { status: "queued" };

    if (!res.ok) {
      return { status: "error", message: `HTTP ${res.status}` };
    }

    const data = (await res.json()) as TraceResponse;
    return { status: "ok", data };
  } catch (err) {
    return {
      status: "error",
      message: err instanceof Error ? err.message : "Network error",
    };
  }
}
