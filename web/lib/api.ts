export type Label = "in_scope" | "out_of_scope" | "ambiguous";

export type Project = {
  id: string;
  name: string;
  support_domain_description: string;
  allowed_topics: string[];
  disallowed_topics: string[];
  routing_notes: string;
  agent_model: string;
  max_rounds: number;
  target_macro_f1: number;
  target_out_of_scope_precision: number;
  sandbox_profile: string;
  promoted_run_id?: string | null;
  created_at: string;
  updated_at: string;
};

export type Example = {
  id: string;
  text: string;
  label: Label;
  source: string;
  approved: boolean;
  split?: "train" | "eval" | "holdout" | null;
  created_at: string;
};

export type Run = {
  id: string;
  project_id: string;
  status: string;
  stop_reason?: string | null;
  best_round_id?: string | null;
  best_macro_f1?: number | null;
  summary?: string | null;
  workspace_root: string;
  created_at: string;
  updated_at: string;
};

export type RunEvent = {
  id: number;
  run_id: string;
  event_type: string;
  message: string;
  payload: Record<string, unknown>;
  created_at: string;
};

export type Round = {
  id: string;
  round_index: number;
  status: string;
  holdout_file?: string | null;
  holdout_evaluation_file?: string | null;
  metrics: Record<string, unknown>;
  holdout_metrics: Record<string, unknown>;
  note?: string | null;
};

export type RunDetail = Run & {
  rounds: Round[];
  plan_markdown: string;
  review_markdown: string;
  final_summary_markdown: string;
  holdout_summary: Record<string, unknown>;
  events: RunEvent[];
};

export type RunStatusResponse = Run & {
  event_count: number;
  latest_event?: RunEvent | null;
};

export type ProjectDetail = {
  project: Project;
  examples: Example[];
  runs: Run[];
  holdout_counts: Record<string, number>;
  holdout_ready: boolean;
};

export type ClassificationResponse = {
  label: Label;
  confidence: number;
  probabilities: Record<string, number>;
  explanation: string;
};

export type QuickStartResponse = {
  project_id: string;
  run_id: string;
};

export type LuckyPromptResponse = {
  description: string;
};

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

/** False when the app will silently call the visitor's own localhost. */
export const isApiBaseUrlConfigured = Boolean(process.env.NEXT_PUBLIC_API_BASE_URL);

if (typeof window !== "undefined" && process.env.NODE_ENV === "production" && !isApiBaseUrlConfigured) {
  // Warn rather than throw: throwing at module scope breaks the build for
  // anyone building without the variable, including CI.
  console.warn(
    "NEXT_PUBLIC_API_BASE_URL is not set, so the app is calling http://127.0.0.1:8000. " +
      "Set it to the deployed API URL."
  );
}

const SESSION_STORAGE_KEY = "scope-classifier-session-id";

/** Error carrying the HTTP status so callers can branch on it. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string, readonly detail?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** Copy shown to users. Backend detail is logged, never rendered verbatim. */
function friendlyMessage(status: number): string {
  if (status === 404) return "That item could not be found.";
  if (status === 409) return "This project already has a run in progress.";
  if (status === 422) return "That input was not accepted. Check the fields and try again.";
  if (status === 429) return "You have hit the demo limit. Please wait a moment and try again.";
  if (status >= 500) return "The classifier service is unavailable. Please try again shortly.";
  return "Something went wrong. Please try again.";
}

function sanitizeDetail(raw: string): string {
  // A FastAPI or proxy failure can return a full HTML page or a Python
  // traceback. Trim it and strip markup before it is ever logged or shown.
  return raw.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim().slice(0, 300);
}

export async function checkBackendHealth(timeoutMs = 5000): Promise<boolean> {
  if (typeof window === "undefined") return true;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      signal: controller.signal,
      cache: "no-store"
    });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

function createFallbackSessionId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).slice(2, 12);
  return `session-${timestamp}-${random}`;
}

// Fallback for browsers where localStorage throws (Safari private mode,
// sandboxed iframes, blocked storage, quota exceeded). Without this every
// request would fail before it was even sent.
let memorySessionId: string | null = null;

export function getClientSessionId(): string {
  if (typeof window === "undefined") {
    return "server-render";
  }
  try {
    const existing = window.localStorage.getItem(SESSION_STORAGE_KEY);
    if (existing) return existing;
  } catch {
    // Storage unavailable; fall through to the in-memory session.
  }
  const created =
    memorySessionId ??
    (typeof window.crypto?.randomUUID === "function"
      ? window.crypto.randomUUID()
      : createFallbackSessionId());
  memorySessionId = created;
  try {
    window.localStorage.setItem(SESSION_STORAGE_KEY, created);
  } catch {
    // Ignored: the in-memory id keeps this session working.
  }
  return created;
}

type RequestOptions = RequestInit & { timeoutMs?: number };

async function request<T>(path: string, init?: RequestOptions): Promise<T> {
  const { timeoutMs = 30_000, ...fetchInit } = init ?? {};
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...fetchInit,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        "X-Session-Id": getClientSessionId(),
        ...(fetchInit.headers ?? {})
      },
      cache: "no-store"
    });
  } catch (error) {
    clearTimeout(timer);
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError(408, "The request timed out. Please try again.");
    }
    throw new ApiError(0, "Could not reach the classifier service.");
  }
  clearTimeout(timer);

  if (!response.ok) {
    let detail = "";
    try {
      detail = sanitizeDetail(await response.text());
    } catch {
      detail = "";
    }
    if (detail) console.error(`API ${response.status} ${path}: ${detail}`);
    throw new ApiError(response.status, friendlyMessage(response.status), detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }
  const text = await response.text();
  if (!text.trim()) {
    return undefined as T;
  }
  try {
    return JSON.parse(text) as T;
  } catch {
    console.error(`API ${path} returned non-JSON response`);
    throw new ApiError(response.status, "Received an unexpected response from the service.");
  }
}

// Responses are asserted to T and then dereferenced by the pages, so a 200
// with an unexpected shape produced "Cannot read properties of undefined".
// These guards turn that into a real error message.
function asArray<T>(value: unknown, name: string): T[] {
  if (!Array.isArray(value)) throw new ApiError(200, `Unexpected ${name} response.`);
  return value as T[];
}

function parseProjectDetail(payload: unknown): ProjectDetail {
  const project = (payload as ProjectDetail | null)?.project;
  if (!project?.id) throw new ApiError(200, "Unexpected project response.");
  const detail = payload as ProjectDetail;
  return {
    project,
    examples: asArray<Example>(detail.examples ?? [], "examples"),
    runs: asArray<Run>(detail.runs ?? [], "runs"),
    holdout_counts: detail.holdout_counts ?? {},
    holdout_ready: Boolean(detail.holdout_ready)
  };
}

function parseRunDetail(payload: unknown): RunDetail {
  const run = payload as RunDetail | null;
  if (!run?.id) throw new ApiError(200, "Unexpected run response.");
  return { ...run, rounds: asArray<Round>(run.rounds ?? [], "rounds"), events: asArray<RunEvent>(run.events ?? [], "events") };
}

export const api = {
  getProject: async (projectId: string) =>
    parseProjectDetail(await request<unknown>(`/projects/${projectId}`)),
  addExamples: (projectId: string, payload: Array<{ text: string; label: Label }>) =>
    request<Example[]>(`/projects/${projectId}/examples`, { method: "POST", body: JSON.stringify(payload) }),
  startRun: (projectId: string) =>
    request<Run>(`/projects/${projectId}/runs`, { method: "POST", body: JSON.stringify({}) }),
  getRun: async (runId: string) => parseRunDetail(await request<unknown>(`/runs/${runId}`)),
  // Used by the run watchdog to recover a status the stream never delivered.
  getRunStatus: (runId: string) => request<RunStatusResponse>(`/runs/${runId}/status`, { timeoutMs: 15_000 }),
  getRunEvents: async (runId: string) =>
    asArray<RunEvent>(await request<unknown>(`/runs/${runId}/events`), "events"),
  classify: (projectId: string, text: string) =>
    request<ClassificationResponse>(`/projects/${projectId}/classify`, {
      method: "POST",
      body: JSON.stringify({ text }),
      // Cold-start checkpoint loads can exceed the default timeout.
      timeoutMs: 60_000
    }),
  quickStart: (description: string) =>
    request<QuickStartResponse>("/quick-start", {
      method: "POST",
      body: JSON.stringify({ description }),
      timeoutMs: 60_000
    }),
  luckyPrompt: () =>
    request<LuckyPromptResponse>("/quick-start/lucky", {
      method: "POST",
      timeoutMs: 15_000
    })
};
