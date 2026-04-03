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

export type ProjectSummary = Project & {
  seed_counts: Record<string, number>;
  holdout_counts: Record<string, number>;
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
  holdout_summary: Record<string, unknown>;
  events: RunEvent[];
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
const SESSION_STORAGE_KEY = "scope-classifier-session-id";

function createFallbackSessionId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).slice(2, 12);
  return `session-${timestamp}-${random}`;
}

export function getClientSessionId(): string {
  if (typeof window === "undefined") {
    return "server-render";
  }
  const existing = window.localStorage.getItem(SESSION_STORAGE_KEY);
  if (existing) return existing;
  const created =
    typeof window.crypto?.randomUUID === "function"
      ? window.crypto.randomUUID()
      : createFallbackSessionId();
  window.localStorage.setItem(SESSION_STORAGE_KEY, created);
  return created;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const sessionId = getClientSessionId();
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      "X-Session-Id": sessionId,
      ...(init?.headers ?? {})
    },
    cache: "no-store"
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  listProjects: () => request<ProjectSummary[]>("/projects"),
  getProject: (projectId: string) => request<ProjectDetail>(`/projects/${projectId}`),
  createProject: (payload: Omit<Project, "id" | "created_at" | "updated_at" | "promoted_run_id">) =>
    request<Project>("/projects", { method: "POST", body: JSON.stringify(payload) }),
  addExamples: (projectId: string, payload: Array<{ text: string; label: Label }>) =>
    request<Example[]>(`/projects/${projectId}/examples`, { method: "POST", body: JSON.stringify(payload) }),
  startRun: (projectId: string) => request<Run>(`/projects/${projectId}/runs`, { method: "POST", body: JSON.stringify({}) }),
  getRun: (runId: string) => request<RunDetail>(`/runs/${runId}`),
  getRunEvents: (runId: string) => request<RunEvent[]>(`/runs/${runId}/events`),
  classify: (projectId: string, text: string) =>
    request<ClassificationResponse>(`/projects/${projectId}/classify`, {
      method: "POST",
      body: JSON.stringify({ text })
    }),
  quickStart: (description: string) =>
    request<QuickStartResponse>("/quick-start", {
      method: "POST",
      body: JSON.stringify({ description }),
    }),
  luckyPrompt: () =>
    request<LuckyPromptResponse>("/quick-start/lucky", {
      method: "POST",
    }),
  promoteRun: (projectId: string, runId: string) =>
    request<void>(`/projects/${projectId}/promote/${runId}`, { method: "POST" }),
};
