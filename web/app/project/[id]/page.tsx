"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { API_BASE_URL, api, ClassificationResponse, Example, Label, Round, Run, RunDetail, RunEvent, getClientSessionId } from "../../../lib/api";

// ─── Types ───────────────────────────────────────────────────────────────────
type SeedTab = "view" | "add";

const LABEL_COLORS: Record<Label, string> = {
  in_scope:     "badge-green",
  out_of_scope: "badge-red",
  ambiguous:    "badge-amber",
};

const LABEL_DISPLAY: Record<Label, string> = {
  in_scope:     "In Scope",
  out_of_scope: "Out of Scope",
  ambiguous:    "Ambiguous",
};

const LABEL_CSS: Record<Label, string> = {
  in_scope:     "in-scope",
  out_of_scope: "out-scope",
  ambiguous:    "ambiguous",
};

const seedTemplate: Record<Label, string> = {
  in_scope:     "I was charged twice for my subscription.\nI cannot log into my account.\nCan you help me cancel my plan?",
  out_of_scope: "What is the distance from Earth to the sun?\nCan you write me a React component?\nBrainstorm a fantasy novel plot.",
  ambiguous:    "My account is locked and also can you help me write a scraper?\nCan you explain your billing and also recommend restaurants nearby?",
};

const PLAYGROUND_EXAMPLES = [
  "I was charged twice this month",
  "How do I reset my password?",
  "Can you write me a Python script?",
  "I need to cancel my subscription",
  "What year was Python created?",
  "My invoice looks wrong",
];

// ─── Helpers ─────────────────────────────────────────────────────────────────
function fmt(n: number | null | undefined): string {
  if (n == null) return "—";
  return (Math.round(n * 1000) / 1000).toFixed(3);
}

function metricColor(n: number | null | undefined): string {
  if (n == null) return "";
  if (n >= 0.85) return "good";
  if (n >= 0.7)  return "warn";
  return "bad";
}

function fmtTs(iso: string): string {
  return new Date(iso).toLocaleTimeString([], {
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  });
}

function fmtTime(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// ─── Stage tracker ────────────────────────────────────────────────────────────
type StageStatus = "pending" | "active" | "done" | "failed";

function deriveStages(
  events: RunEvent[],
  run: Run,
  maxRounds: number,
): { id: string; label: string; status: StageStatus }[] {
  const evTypes = new Set(events.map((e) => e.event_type));
  const roundsStarted = new Set(
    events
      .filter((e) => e.event_type === "round_started")
      .map((e) => (e.payload as Record<string, number>)?.round_index)
  );
  const roundsDone = new Set(
    events
      .filter((e) => e.event_type === "round_complete")
      .map((e) => (e.payload as Record<string, number>)?.round_index)
  );

  const isFailed = run.status === "failed";
  const isComplete = run.status === "completed";
  const planDone = evTypes.has("plan_ready");
  const planActive = evTypes.has("run_started") && !planDone;

  const stages: { id: string; label: string; status: StageStatus }[] = [];

  // Plan stage
  stages.push({
    id: "plan",
    label: "Plan",
    status: planDone ? "done" : planActive ? "active" : "pending",
  });

  // Round stages
  for (let i = 1; i <= maxRounds; i++) {
    const started = roundsStarted.has(i);
    const done = roundsDone.has(i);
    stages.push({
      id: `r${i}`,
      label: `R${i}`,
      status: done ? "done" : started ? "active" : "pending",
    });
  }

  // Review stage
  const reviewDone = evTypes.has("run_reviewed") || isComplete;
  const reviewActive = evTypes.has("review_started") && !reviewDone;
  stages.push({
    id: "review",
    label: "Review",
    status: reviewDone ? "done" : reviewActive ? "active" : "pending",
  });

  // Done stage
  stages.push({
    id: "done",
    label: "Done",
    status: isComplete ? "done" : isFailed ? "failed" : "pending",
  });

  return stages;
}

function RunStageTracker({ run, events, maxRounds }: { run: Run; events: RunEvent[]; maxRounds: number }) {
  const stages = deriveStages(events, run, maxRounds);
  return (
    <div className="stage-tracker">
      {stages.map((stage, i) => (
        <div key={stage.id} className="stage-track-item">
          <div className={`stage-node ${stage.status}`}>
            <div className="stage-dot">
              {stage.status === "done" ? "✓" : stage.status === "failed" ? "✗" : stage.label}
            </div>
            <span className="stage-node-label">{stage.label}</span>
          </div>
          {i < stages.length - 1 && (
            <div className={`stage-pipe ${stage.status === "done" ? "done" : stage.status === "active" ? "active" : ""}`} />
          )}
        </div>
      ))}
    </div>
  );
}

// ─── Terminal sub-components ──────────────────────────────────────────────────

function TerminalMetricBar({ label, value, color = "auto" }: {
  label: string; value: number | null | undefined; color?: "auto" | "blue" | "green";
}) {
  if (value == null) return null;
  const pct = Math.min(100, value * 100);
  let barClass = "bar-accent";
  if (color === "blue")       barClass = "bar-blue";
  else if (color === "green") barClass = "bar-green";
  else if (value >= 0.85)     barClass = "bar-green";
  else if (value >= 0.7)      barClass = "bar-amber";
  else                        barClass = "bar-red";
  return (
    <div className="term-metric-row">
      <span className="term-metric-label">{label}</span>
      <div className="term-metric-bar-wrap">
        <div className={`term-metric-bar ${barClass}`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`term-metric-val ${color === "blue" ? "metric-blue" : metricColor(value)}`}>
        {fmt(value)}
      </span>
    </div>
  );
}

function TerminalMarkdown({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const lines = text.split("\n");
  const preview = lines.slice(0, 5).join("\n");
  const hasMore = lines.length > 5;
  return (
    <div className="term-markdown-block">
      <pre className="term-markdown-pre">{open ? text : preview + (hasMore ? "\n…" : "")}</pre>
      {hasMore && (
        <button className="term-expand-btn" onClick={() => setOpen(o => !o)} type="button">
          {open ? "▲ collapse" : `▼ show ${lines.length - 5} more lines`}
        </button>
      )}
    </div>
  );
}

function TerminalEvent({ event, runDetail }: { event: RunEvent; runDetail: RunDetail }) {
  const ts = fmtTs(event.created_at);
  const p = (event.payload ?? {}) as Record<string, unknown>;

  switch (event.event_type) {
    case "run_started":
      return (
        <div className="term-block term-run-start-block">
          <div className="term-line">
            <span className="term-ts">{ts}</span>
            <span className="term-icon" style={{ color: "var(--green)" }}>●</span>
            <span style={{ color: "var(--green)", fontWeight: 700, letterSpacing: "0.06em" }}>RUN STARTED</span>
            <span className="term-dim" style={{ marginLeft: 8 }}>{event.run_id.slice(0, 14)}…</span>
          </div>
        </div>
      );

    case "plan_ready":
      return (
        <div className="term-block">
          <div className="term-line">
            <span className="term-ts">{ts}</span>
            <span className="term-icon" style={{ color: "var(--accent-2)" }}>◈</span>
            <span className="term-bold">Strategy planned</span>
          </div>
          {runDetail.plan_markdown && <TerminalMarkdown text={runDetail.plan_markdown} />}
        </div>
      );

    case "round_started": {
      const idx = (p.round_index as number) ?? 1;
      return (
        <div className="term-round-sep">
          <span className="term-ts-sep">{ts}</span>
          <span className="term-round-label">ROUND {idx}</span>
          <span className="term-round-rule" />
        </div>
      );
    }

    case "candidate_generation_started":
      return (
        <div className="term-line term-minor">
          <span className="term-ts">{ts}</span>
          <span className="term-icon term-spin" style={{ color: "var(--accent)" }}>⚙</span>
          <span className="term-dim">Generating synthetic examples…</span>
        </div>
      );

    case "generation_in_progress": {
      const topic = p.topic as string | undefined;
      const count = p.count as number | undefined;
      return (
        <div className="term-line term-minor">
          <span className="term-ts">{ts}</span>
          <span className="term-icon term-dim">·</span>
          <span className="term-dim">
            {count != null ? `Generating ${count}` : "Generating"}
            {topic ? ` · ${topic.slice(0, 45)}${topic.length > 45 ? "…" : ""}` : ""}
          </span>
        </div>
      );
    }

    case "candidates_generated": {
      const total = (p.generated_count ?? p.total) as number | undefined;
      return (
        <div className="term-line">
          <span className="term-ts">{ts}</span>
          <span className="term-icon" style={{ color: "var(--blue)" }}>↗</span>
          <span className="term-bold">{total != null ? `${total} ` : ""}Candidates generated</span>
        </div>
      );
    }

    case "dataset_prep_started":
    case "holdout_generation_started":
    case "holdout_evaluation_started":
    case "evaluation_started":
      return (
        <div className="term-line term-minor">
          <span className="term-ts">{ts}</span>
          <span className="term-icon term-dim">·</span>
          <span className="term-dim">{event.message}</span>
        </div>
      );

    case "dataset_prepared": {
      const train    = (p.train_count  ?? p.train) as number | undefined;
      const ev       = (p.eval_count   ?? p.eval)  as number | undefined;
      const accepted = p.accepted_examples          as number | undefined;
      const labels   = p.label_counts               as Record<string, number> | undefined;
      return (
        <div className="term-block">
          <div className="term-line">
            <span className="term-ts">{ts}</span>
            <span className="term-icon" style={{ color: "var(--blue)" }}>⊡</span>
            <span className="term-bold">Dataset prepared</span>
            {accepted != null && <span className="term-dim" style={{ marginLeft: 8 }}>+{accepted} new</span>}
            {(train != null || ev != null) && (
              <span className="term-dim" style={{ marginLeft: 6 }}>
                ({[train != null && `${train} train`, ev != null && `${ev} eval`].filter(Boolean).join(" / ")})
              </span>
            )}
          </div>
          {labels && (
            <div className="term-indent-grid">
              {Object.entries(labels).map(([lbl, cnt]) => (
                <div key={lbl} className="term-kv">
                  <span className="term-key">{lbl.replace(/_/g, " ")}</span>
                  <span className="term-val">{cnt}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      );
    }

    case "training_started":
      return (
        <div className="term-block">
          <div className="term-line">
            <span className="term-ts">{ts}</span>
            <span className="term-icon" style={{ color: "var(--amber)" }}>⚡</span>
            <span className="term-bold">Fine-tuning DistilBERT</span>
            <span className="term-dim" style={{ marginLeft: 8, fontSize: "0.70rem" }}>training in progress…</span>
          </div>
          <div className="term-train-bar-wrap">
            <div className="term-train-bar-track">
              <div className="term-train-bar-fill" />
            </div>
          </div>
        </div>
      );

    case "training_complete": {
      const trainingLoss = (p.training_loss ?? p.final_loss) as number | undefined;
      const trainCount   = (p.train_count  ?? p.count)       as number | undefined;
      return (
        <div className="term-block">
          <div className="term-line">
            <span className="term-ts">{ts}</span>
            <span className="term-icon" style={{ color: "var(--amber)" }}>⚡</span>
            <span className="term-bold">Training complete</span>
          </div>
          <div className="term-indent-grid">
            {trainingLoss != null && (
              <div className="term-kv">
                <span className="term-key">loss</span>
                <span className="term-val">{trainingLoss.toFixed(4)}</span>
              </div>
            )}
            {trainCount != null && (
              <div className="term-kv">
                <span className="term-key">examples</span>
                <span className="term-val">{trainCount}</span>
              </div>
            )}
          </div>
        </div>
      );
    }

    case "evaluation_complete": {
      const macro_f1 = p.macro_f1                as number | undefined;
      const oos_prec = p.out_of_scope_precision   as number | undefined;
      const perClass = p.per_class                as Record<string, Record<string, number>> | undefined;
      return (
        <div className="term-block">
          <div className="term-line">
            <span className="term-ts">{ts}</span>
            <span className="term-icon" style={{ color: "var(--green)" }}>✓</span>
            <span className="term-bold">Evaluation complete</span>
          </div>
          <div className="term-metrics-block">
            <TerminalMetricBar label="Macro F1   " value={macro_f1} />
            <TerminalMetricBar label="OOS Prec   " value={oos_prec} />
          </div>
          {perClass && (
            <div className="term-perclass-grid">
              {Object.entries(perClass).map(([lbl, m]) => (
                <div key={lbl} className="term-perclass-row">
                  <span className="term-perclass-label">{lbl.replace(/_/g, " ")}</span>
                  <span className={`term-perclass-val ${metricColor(m.f1)}`}>{fmt(m.f1)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      );
    }

    case "holdout_created": {
      const cnt = (p.count ?? p.total) as number | undefined;
      return (
        <div className="term-line">
          <span className="term-ts">{ts}</span>
          <span className="term-icon" style={{ color: "var(--blue)" }}>⊞</span>
          <span>Holdout set created</span>
          {cnt != null && <span className="term-dim" style={{ marginLeft: 8 }}>{cnt} examples</span>}
        </div>
      );
    }

    case "holdout_reused":
      return (
        <div className="term-line term-minor">
          <span className="term-ts">{ts}</span>
          <span className="term-icon" style={{ color: "var(--blue)" }}>⊞</span>
          <span className="term-dim">Holdout set reused from round 1</span>
        </div>
      );

    case "holdout_evaluated": {
      const macro_f1 = p.macro_f1              as number | undefined;
      const oos_prec = p.out_of_scope_precision as number | undefined;
      const perClass = p.per_class              as Record<string, Record<string, number>> | undefined;
      return (
        <div className="term-block">
          <div className="term-line">
            <span className="term-ts">{ts}</span>
            <span className="term-icon" style={{ color: "var(--blue)" }}>⊞</span>
            <span className="term-bold">Holdout evaluated</span>
          </div>
          <div className="term-metrics-block">
            <TerminalMetricBar label="Holdout F1 " value={macro_f1} color="blue" />
            <TerminalMetricBar label="OOS Prec   " value={oos_prec} color="blue" />
          </div>
          {perClass && (
            <div className="term-perclass-grid">
              {Object.entries(perClass).map(([lbl, m]) => (
                <div key={lbl} className="term-perclass-row">
                  <span className="term-perclass-label">{lbl.replace(/_/g, " ")}</span>
                  <span className={`term-perclass-val ${metricColor(m.f1)}`}>{fmt(m.f1)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      );
    }

    case "review_started":
      return (
        <div className="term-line term-minor">
          <span className="term-ts">{ts}</span>
          <span className="term-icon term-pulse" style={{ color: "var(--accent-2)" }}>✎</span>
          <span className="term-dim">Agent reviewing results…</span>
        </div>
      );

    case "review_recorded": {
      const roundIdx = p.round_index as number | undefined;
      return (
        <div className="term-block">
          <div className="term-line">
            <span className="term-ts">{ts}</span>
            <span className="term-icon" style={{ color: "var(--accent-2)" }}>✎</span>
            <span className="term-bold">Review recorded</span>
            {roundIdx != null && <span className="term-dim" style={{ marginLeft: 8 }}>round {roundIdx}</span>}
          </div>
        </div>
      );
    }

    case "run_reviewed":
      return (
        <div className="term-block">
          <div className="term-line">
            <span className="term-ts">{ts}</span>
            <span className="term-icon" style={{ color: "var(--accent-2)" }}>◈</span>
            <span className="term-bold">Final review complete</span>
          </div>
          {runDetail.review_markdown && <TerminalMarkdown text={runDetail.review_markdown} />}
        </div>
      );

    case "checkpoint_promoted": {
      const roundIdx = (p.round_index as number) ?? 1;
      return (
        <div className="term-block term-promoted">
          <div className="term-line">
            <span className="term-ts">{ts}</span>
            <span className="term-icon" style={{ color: "var(--accent-2)" }}>★</span>
            <span style={{ color: "var(--accent-2)", fontWeight: 700, letterSpacing: "0.06em" }}>
              CHECKPOINT PROMOTED
            </span>
            <span className="term-dim" style={{ marginLeft: 8 }}>round {roundIdx}</span>
          </div>
          {(p.note as string | undefined) && (
            <div className="term-indent-block">
              <span className="term-dim">{p.note as string}</span>
            </div>
          )}
        </div>
      );
    }

    case "round_complete":
      return (
        <div className="term-line term-minor">
          <span className="term-ts">{ts}</span>
          <span className="term-icon" style={{ color: "var(--text-3)" }}>◎</span>
          <span className="term-dim">Round complete</span>
        </div>
      );

    case "run_completed": {
      const bestF1    = (p.best_macro_f1 ?? runDetail.best_macro_f1) as number | undefined;
      const holdoutF1 = (runDetail.holdout_summary as Record<string, number> | undefined)?.macro_f1;
      return (
        <div className="run-complete-banner">
          <div className="run-complete-title">
            <span style={{ fontSize: "1.1rem" }}>✓</span>
            RUN COMPLETED
          </div>
          <div className="run-complete-metrics">
            {bestF1 != null && (
              <div className="run-complete-metric">
                <div className={`run-complete-metric-val ${metricColor(bestF1)}`}>{fmt(bestF1)}</div>
                <div className="run-complete-metric-lbl">Best F1</div>
              </div>
            )}
            {holdoutF1 != null && (
              <div className="run-complete-metric">
                <div className={`run-complete-metric-val metric-blue`} style={{ color: "var(--blue)" }}>{fmt(holdoutF1)}</div>
                <div className="run-complete-metric-lbl">Holdout F1</div>
              </div>
            )}
          </div>
          {runDetail.stop_reason && (
            <div className="run-complete-note">{runDetail.stop_reason}</div>
          )}
        </div>
      );
    }

    case "run_failed":
      return (
        <div className="term-block term-run-failed-block">
          <div className="term-line">
            <span className="term-ts">{ts}</span>
            <span className="term-icon" style={{ color: "var(--red)" }}>✗</span>
            <span style={{ color: "var(--red)", fontWeight: 700, letterSpacing: "0.06em" }}>RUN FAILED</span>
          </div>
          {((p.error ?? p.message) as string | undefined) && (
            <div className="term-indent-block">
              <span style={{ color: "var(--red)", fontSize: "0.72rem" }}>
                {(p.error ?? p.message) as string}
              </span>
            </div>
          )}
        </div>
      );

    default:
      return (
        <div className="term-line term-minor">
          <span className="term-ts">{ts}</span>
          <span className="term-icon term-dim">·</span>
          <span className="term-dim">{event.message}</span>
        </div>
      );
  }
}

// ─── Component ───────────────────────────────────────────────────────────────
export default function ProjectPage() {
  const params = useParams<{ id: string }>();
  const projectId = params.id ?? "";
  const paramId = useRef(projectId);

  const [projectName, setProjectName]     = useState("");
  const [maxRounds, setMaxRounds]         = useState(3);
  const [examples, setExamples]           = useState<Example[]>([]);
  const [runs, setRuns]                   = useState<Run[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runDetail, setRunDetail]         = useState<RunDetail | null>(null);
  const [holdoutCounts, setHoldoutCounts] = useState<Record<string, number>>({});
  const [holdoutReady, setHoldoutReady]   = useState(false);

  const [seedTab, setSeedTab]         = useState<SeedTab>("view");
  const [addLabelTab, setAddLabelTab] = useState<Label>("in_scope");
  const [seedInputs, setSeedInputs]   = useState(seedTemplate);
  const [savingSeeds, setSavingSeeds] = useState(false);
  const [seedMsg, setSeedMsg]         = useState("");

  const [message, setMessage]               = useState("");
  const [classifying, setClassifying]       = useState(false);
  const [classification, setClassification] = useState<ClassificationResponse | null>(null);

  const [startingRun, setStartingRun] = useState(false);
  const [error, setError]             = useState("");
  const [loading, setLoading]         = useState(true);
  const [sseActive, setSseActive]     = useState(false);

  const terminalRef       = useRef<HTMLDivElement>(null);
  const terminalBottomRef = useRef<HTMLDivElement>(null);
  const [atBottom, setAtBottom] = useState(true);

  // ── Initial load ──
  useEffect(() => {
    const id = paramId.current;
    api
      .getProject(id)
      .then((payload) => {
        setProjectName(payload.project.name);
        setMaxRounds((payload.project as unknown as { max_rounds?: number }).max_rounds ?? 3);
        setExamples(payload.examples);
        setRuns(payload.runs);
        setHoldoutCounts(payload.holdout_counts);
        setHoldoutReady(payload.holdout_ready);
        const first = payload.runs[0];
        if (first) {
          setSelectedRunId(first.id);
          return api.getRun(first.id).then(setRunDetail);
        }
        return null;
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  // ── Live polling ──
  const isLive = runs.some((r) => r.status === "running" || r.status === "queued");

  useEffect(() => {
    if (!isLive || sseActive) return;
    const id = paramId.current;
    const interval = setInterval(async () => {
      try {
        const payload = await api.getProject(id);
        setRuns(payload.runs);
        setHoldoutCounts(payload.holdout_counts);
        setHoldoutReady(payload.holdout_ready);
        const activeRun =
          payload.runs.find((r) => r.status === "running" || r.status === "queued") ??
          payload.runs[0];
        if (activeRun) {
          const detail = await api.getRun(activeRun.id);
          setRunDetail(detail);
          if (selectedRunId === null || selectedRunId === activeRun.id) {
            setSelectedRunId(activeRun.id);
          }
        }
      } catch { /* silently ignore */ }
    }, 3000);
    return () => clearInterval(interval);
  }, [isLive, sseActive, selectedRunId]);

  // ── SSE stream ──
  useEffect(() => {
    if (!selectedRunId) return;
    const selected = runs.find((r) => r.id === selectedRunId);
    if (!selected || !["running", "queued"].includes(selected.status)) return;

    const source = new EventSource(
      `${API_BASE_URL}/runs/${selectedRunId}/events/stream?session_id=${encodeURIComponent(getClientSessionId())}`
    );
    setSseActive(true);

    const refreshRun = async () => {
      try {
        const detail = await api.getRun(selectedRunId);
        setRunDetail(detail);
        setRuns((cur) => cur.map((r) => (r.id === detail.id ? { ...r, ...detail } : r)));
      } catch { /* ignore */ }
    };

    source.addEventListener("run_event", (ev) => {
      const payload = JSON.parse((ev as MessageEvent).data) as RunEvent;
      setRunDetail((cur) => {
        if (!cur || cur.id !== selectedRunId) return cur;
        if (cur.events.some((item) => item.id === payload.id)) return cur;
        return { ...cur, events: [...cur.events, payload] };
      });
      void refreshRun();
    });

    source.addEventListener("run_done", () => { void refreshRun(); source.close(); setSseActive(false); });
    source.onerror = () => { source.close(); setSseActive(false); };
    return () => { source.close(); setSseActive(false); };
  }, [selectedRunId, runs]);

  // ── Auto-scroll terminal ──
  const eventCount = runDetail?.events?.length ?? 0;
  useEffect(() => {
    if (atBottom) terminalBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [eventCount, atBottom]);

  // ── Select run ──
  const selectRun = useCallback(async (runId: string) => {
    setSelectedRunId(runId);
    setRunDetail(null);
    setAtBottom(true);
    try {
      const detail = await api.getRun(runId);
      setRunDetail(detail);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load run");
    }
  }, []);

  // ── Start run ──
  async function startRun() {
    setStartingRun(true);
    setError("");
    try {
      const run = await api.startRun(paramId.current);
      setRuns((r) => [run, ...r]);
      setSelectedRunId(run.id);
      setRunDetail(null);
      setAtBottom(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to start run");
    } finally {
      setStartingRun(false);
    }
  }

  // ── Save seeds ──
  async function saveSeeds(e: FormEvent) {
    e.preventDefault();
    setSavingSeeds(true);
    setSeedMsg("");
    try {
      const payload = (Object.entries(seedInputs) as Array<[Label, string]>).flatMap(
        ([label, block]) =>
          block.split("\n").map((l) => l.trim()).filter(Boolean).map((text) => ({ text, label }))
      );
      const created = await api.addExamples(paramId.current, payload);
      setExamples((ex) => [...ex, ...created]);
      setSeedMsg(`${created.length} examples saved.`);
      setSeedInputs(seedTemplate);
      setSeedTab("view");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to save seeds");
    } finally {
      setSavingSeeds(false);
    }
  }

  // ── Classify ──
  async function classify(e: FormEvent) {
    e.preventDefault();
    setClassifying(true);
    setClassification(null);
    try {
      const result = await api.classify(paramId.current, message);
      setClassification(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Classification failed");
    } finally {
      setClassifying(false);
    }
  }

  function handleTerminalScroll() {
    const el = terminalRef.current;
    if (!el) return;
    setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 80);
  }

  function scrollToBottom() {
    terminalBottomRef.current?.scrollIntoView({ behavior: "smooth" });
    setAtBottom(true);
  }

  // ── Derived ──
  const humanCounts = examples
    .filter((ex) => ex.source === "human_seed")
    .reduce<Record<string, number>>((acc, ex) => {
      acc[ex.label] = (acc[ex.label] ?? 0) + 1;
      return acc;
    }, {});
  const missingSeedCounts = {
    in_scope: Math.max(0, 5 - (humanCounts.in_scope ?? 0)),
    out_of_scope: Math.max(0, 5 - (humanCounts.out_of_scope ?? 0)),
    ambiguous: Math.max(0, 3 - (humanCounts.ambiguous ?? 0)),
  };
  const seedMinimumMet = Object.values(missingSeedCounts).every((count) => count === 0);

  const selectedRun  = runs.find((r) => r.id === selectedRunId) ?? null;
  const latestRound: Round | null = runDetail?.rounds[runDetail.rounds.length - 1] ?? null;
  const addPreviewCount = seedInputs[addLabelTab].split("\n").map(l => l.trim()).filter(Boolean).length;

  const humanSeeds = examples.filter((ex) => ex.source === "human_seed");

  if (loading) {
    return (
      <>
        <a className="skip-link" href="#project-main">
          Skip to main content
        </a>
        <main className="shell" id="project-main">
          <div className="panel" style={{ padding: 48, textAlign: "center" }}>
            <div className="spinner" style={{ margin: "0 auto 14px", width: 24, height: 24, borderWidth: 3 }} />
            <p style={{ color: "var(--text-2)", fontFamily: "var(--font-display), sans-serif" }}>Loading project…</p>
          </div>
        </main>
      </>
    );
  }

  return (
    <>
      <a className="skip-link" href="#project-main">
        Skip to main content
      </a>
      <main className="shell" id="project-main">
      {/* ── Header ── */}
      <div className="breadcrumb">
        <Link href="/">Projects</Link>
        <span className="breadcrumb-sep">/</span>
        <span style={{ color: "var(--text)" }}>{projectName}</span>
      </div>

      <div className="row-between" style={{ marginBottom: 20 }}>
        <h1 style={{
          fontSize: "1.4rem", fontWeight: 800, letterSpacing: "-0.03em",
          fontFamily: "var(--font-display), sans-serif"
        }}>{projectName}</h1>
        <div className="row">
          {isLive && (
            <span className="live-badge">
              <span className="live-badge-dot" />
              Live
            </span>
          )}
          <button
            className="btn btn-primary"
            onClick={startRun}
            disabled={startingRun || !seedMinimumMet}
            type="button"
          >
            {startingRun ? <><span className="spinner spinner-sm" /> Starting…</> : "▶ Start Run"}
          </button>
        </div>
      </div>

      {error && <p className="inline-error" style={{ marginBottom: 14 }}>{error}</p>}
      {!seedMinimumMet && (
        <p className="inline-error" style={{ marginBottom: 14 }}>
          Add {missingSeedCounts.in_scope} more in-scope, {missingSeedCounts.out_of_scope} more out-of-scope,
          and {missingSeedCounts.ambiguous} more ambiguous seed examples before starting a run.
        </p>
      )}

      {/* ── Three-column layout ── */}
      <div className="project-layout">

        {/* ══ Column 1: Seeds ══ */}
        <aside className="panel seeds-panel">
          {/* Header */}
          <div className="seeds-header">
            <span className="seeds-title">Training Seeds</span>
            <button
              className={`btn btn-sm ${seedTab === "add" ? "btn-ghost" : "btn-primary"}`}
              onClick={() => setSeedTab(seedTab === "add" ? "view" : "add")}
              type="button"
              style={{ fontSize: "0.75rem", padding: "4px 12px" }}
            >
              {seedTab === "add" ? "← View" : "+ Add"}
            </button>
          </div>

          {/* Stats row */}
          <div className="seed-stats">
            <div className="seed-stat">
              <div className="seed-stat-value" style={{ color: "var(--green)" }}>{humanCounts.in_scope ?? 0}</div>
              <div className="seed-stat-label">In-scope</div>
            </div>
            <div className="seed-stat">
              <div className="seed-stat-value" style={{ color: "var(--red)" }}>{humanCounts.out_of_scope ?? 0}</div>
              <div className="seed-stat-label">Out-scope</div>
            </div>
            <div className="seed-stat">
              <div className="seed-stat-value" style={{ color: "var(--amber)" }}>{humanCounts.ambiguous ?? 0}</div>
              <div className="seed-stat-label">Ambiguous</div>
            </div>
          </div>

          {/* Holdout readiness */}
          <div className="seed-holdout-row">
            <span className={`badge ${holdoutReady ? "badge-green" : "badge-neutral"}`} style={{ fontSize: "0.7rem" }}>
              {holdoutReady ? "✓ holdout ready" : "holdout not ready"}
            </span>
            {holdoutReady && (
              <span className="badge badge-neutral" style={{ fontSize: "0.7rem" }}>
                {(holdoutCounts.in_scope ?? 0) + (holdoutCounts.out_of_scope ?? 0) + (holdoutCounts.ambiguous ?? 0)} examples
              </span>
            )}
          </div>

          <div className="divider" style={{ margin: "12px 0" }} />

          {/* ── View mode ── */}
          {seedTab === "view" && (
            <div className="seed-scroll">
              {humanSeeds.length === 0 ? (
                <div className="seed-empty">
                  <div style={{ fontSize: "2rem", marginBottom: 8, opacity: 0.4 }}>🌱</div>
                  <p>No seed examples yet.</p>
                  <button
                    className="btn btn-ghost btn-sm"
                    onClick={() => setSeedTab("add")}
                    type="button"
                    style={{ marginTop: 10 }}
                  >
                    + Add seeds
                  </button>
                </div>
              ) : (
                <div className="seed-groups">
                  {(["in_scope", "out_of_scope", "ambiguous"] as Label[]).map((label) => {
                    const items = humanSeeds.filter((ex) => ex.label === label);
                    if (items.length === 0) return null;
                    return (
                      <div key={label} className="seed-label-section">
                        <div className="seed-section-header">
                          <span className={`badge ${LABEL_COLORS[label]}`} style={{ fontSize: "0.68rem" }}>
                            {LABEL_DISPLAY[label]}
                          </span>
                          <span className="seed-section-count">{items.length}</span>
                        </div>
                        {items.map((ex) => (
                          <div key={ex.id} className={`seed-item ${LABEL_CSS[label]}`}>
                            {ex.text}
                          </div>
                        ))}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* ── Add mode ── */}
          {seedTab === "add" && (
            <form onSubmit={saveSeeds}>
              {/* Label tabs */}
              <div className="seed-add-tabs">
                {(["in_scope", "out_of_scope", "ambiguous"] as Label[]).map((label) => (
                  <button
                    key={label}
                    type="button"
                    className={`seed-add-tab ${addLabelTab === label ? `active ${LABEL_CSS[label]}` : ""}`}
                    onClick={() => setAddLabelTab(label)}
                  >
                    {label === "in_scope" ? "In Scope" : label === "out_of_scope" ? "Out of Scope" : "Ambiguous"}
                  </button>
                ))}
              </div>

              <div className="field">
                <textarea
                  className="textarea"
                  rows={8}
                  value={seedInputs[addLabelTab]}
                  onChange={(e) => setSeedInputs((s) => ({ ...s, [addLabelTab]: e.target.value }))}
                  placeholder="One example per line…"
                  style={{ fontSize: "0.8rem", fontFamily: "var(--font-mono), monospace" }}
                />
                {addPreviewCount > 0 && (
                  <div className="seed-add-preview">
                    <span className="seed-preview-count">{addPreviewCount}</span>
                    <span>{addPreviewCount === 1 ? "example" : "examples"} to add for this label</span>
                  </div>
                )}
              </div>

              {seedMsg && <p className="inline-success" style={{ marginBottom: 8 }}>{seedMsg}</p>}

              <button
                className="btn btn-primary"
                type="submit"
                disabled={savingSeeds}
                style={{ width: "100%", justifyContent: "center", marginTop: 4 }}
              >
                {savingSeeds ? <><span className="spinner spinner-sm" /> Saving…</> : "Save all seeds"}
              </button>
            </form>
          )}
        </aside>

        {/* ══ Column 2: Terminal ══ */}
        <section className="panel terminal-panel">
          {/* Terminal top bar */}
          <div className="terminal-header">
            <div className="terminal-header-left">
              <span className="terminal-title">Agent Terminal</span>
              {isLive && (
                <span className="live-badge">
                  <span className="live-badge-dot" />
                  Streaming
                </span>
              )}
              {selectedRun && (
                <div className="term-run-metrics">
                  {selectedRun.best_macro_f1 != null && (
                    <span className={`badge ${selectedRun.best_macro_f1 >= 0.85 ? "badge-green" : "badge-amber"}`}>
                      F1 {fmt(selectedRun.best_macro_f1)}
                    </span>
                  )}
                  <span className={`badge ${
                    selectedRun.status === "completed" ? "badge-green" :
                    selectedRun.status === "failed"    ? "badge-red"   :
                    selectedRun.status === "running"   ? "badge-amber" : "badge-neutral"
                  }`}>{selectedRun.status}</span>
                </div>
              )}
            </div>

            {runs.length > 0 && (
              <div className="run-tabs">
                {runs.map((run) => (
                  <button
                    key={run.id}
                    className={`run-tab ${selectedRunId === run.id ? "active" : ""}`}
                    onClick={() => selectRun(run.id)}
                    type="button"
                  >
                    <span className={`status-dot ${run.status}`} />
                    {run.id.slice(0, 8)}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Stage tracker */}
          {selectedRun && runDetail && (runDetail.events ?? []).length > 0 && (
            <RunStageTracker run={selectedRun} events={runDetail.events} maxRounds={maxRounds} />
          )}

          {/* Terminal body */}
          <div className="terminal-body" ref={terminalRef} onScroll={handleTerminalScroll}>
            {runs.length === 0 && (
              <div className="term-empty">
                <span className="term-prompt">$</span>
                <span>&nbsp;no runs yet —&nbsp;</span>
                <span className="term-hl">▶ Start Run</span>
                <span>&nbsp;to begin</span>
              </div>
            )}

            {selectedRun && (runDetail?.events ?? []).length === 0 && (
              <div className="term-line">
                <span className="term-ts">{fmtTime(selectedRun.created_at)}</span>
                <span className="term-icon" style={{ color: "var(--blue)" }}>·</span>
                <span className="term-dim">
                  {selectedRun.status === "queued" ? "Queued — waiting to start…" : "Loading events…"}
                </span>
              </div>
            )}

            {runDetail && (runDetail.events ?? []).map((event) => (
              <TerminalEvent key={event.id} event={event} runDetail={runDetail} />
            ))}

            {isLive && (runDetail?.events ?? []).length > 0 && (
              <div className="term-line term-cursor-line">
                <span className="term-ts">&nbsp;</span>
                <span className="term-blink">█</span>
              </div>
            )}

            <div ref={terminalBottomRef} />
          </div>

          {!atBottom && (
            <button className="scroll-fab" onClick={scrollToBottom} type="button">
              ↓ latest
            </button>
          )}
        </section>

        {/* ══ Column 3: Playground ══ */}
        <aside className="panel playground-panel">
          <div className="playground-header">
            <span className="playground-title">Live Playground</span>
            {latestRound?.status === "completed" && (
              <span className="badge badge-green" style={{ fontSize: "0.68rem" }}>Model ready</span>
            )}
          </div>

          {/* Model not ready state */}
          {!latestRound || latestRound.status !== "completed" ? (
            <div className="playground-empty-state">
              <div className="playground-empty-icon">🧠</div>
              <p className="playground-empty-text">
                Train a model first.<br />
                Start an agent run to fine-tune your classifier.
              </p>
            </div>
          ) : null}

          {/* Example prompt chips */}
          {latestRound?.status === "completed" && !message && (
            <>
              <p className="playground-examples-label">Try an example</p>
              <div className="playground-chips">
                {PLAYGROUND_EXAMPLES.map((eg) => (
                  <button
                    key={eg}
                    className="eg-chip"
                    onClick={() => setMessage(eg)}
                    type="button"
                  >
                    {eg}
                  </button>
                ))}
              </div>
            </>
          )}

          <form className="stack-sm" onSubmit={classify} style={{ marginTop: 8 }}>
            <div className="field">
              <textarea
                className="textarea"
                rows={4}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Type a support message to classify…"
                style={{ fontSize: "0.85rem" }}
              />
            </div>
            <button
              className="btn btn-primary"
              type="submit"
              disabled={classifying || !message.trim()}
              style={{ width: "100%", justifyContent: "center" }}
            >
              {classifying
                ? <><span className="spinner spinner-sm" /> Classifying…</>
                : "Classify →"
              }
            </button>
          </form>

          {classification && (
            <>
              <div className="divider" />
              <div className="classify-result">
                <div className="classify-label-row">
                  <span className={`badge ${LABEL_COLORS[classification.label]} classify-label-badge`}>
                    {LABEL_DISPLAY[classification.label]}
                  </span>
                  <span style={{ fontSize: "0.82rem", color: "var(--text-2)", fontWeight: 700 }}>
                    {(classification.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <p className="classify-explanation">{classification.explanation}</p>
                <div className="confidence-row">
                  {Object.entries(classification.probabilities)
                    .sort(([, a], [, b]) => b - a)
                    .map(([label, prob]) => (
                      <div className="confidence-item" key={label}>
                        <div className="confidence-label-row">
                          <span className="confidence-label">{LABEL_DISPLAY[label as Label] ?? label}</span>
                          <span className="confidence-pct">{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="progress-bar-wrap">
                          <div
                            className={`progress-bar-fill ${
                              label === "in_scope"     ? "green" :
                              label === "out_of_scope" ? "accent" : "amber"
                            }`}
                            style={{ width: `${Math.min(100, prob * 100)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </>
          )}

          {/* Active model stats */}
          {latestRound?.status === "completed" && (
            <>
              <div className="divider" />
              <p className="panel-title" style={{ marginBottom: 10 }}>Active model</p>
              <div className="model-stats-grid">
                {(() => {
                  const m = latestRound.metrics as Record<string, number>;
                  return [
                    { label: "Macro F1",  value: m?.macro_f1 },
                    { label: "OOS Prec",  value: m?.out_of_scope_precision },
                    { label: "Recall",    value: m?.macro_recall },
                  ].map(({ label, value }) => (
                    <div key={label} className="model-stat-item">
                      <span className={`model-stat-val ${metricColor(value)}`}>{fmt(value)}</span>
                      <span className="model-stat-lbl">{label}</span>
                    </div>
                  ));
                })()}
              </div>
            </>
          )}
        </aside>

      </div>
      </main>
    </>
  );
}
