"use client";

import {
  KeyboardEvent,
  RefObject,
  startTransition,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import Link from "next/link";
import styles from "./page.module.css";
import {
  API_BASE_URL,
  api,
  Example,
  RunDetail,
  RunEvent,
  getClientSessionId,
} from "../lib/api";

type ViewState = "idle" | "processing" | "complete" | "error";
type LaunchMode = "manual" | "lucky";
type WorkflowStageId =
  | "starting"
  | "analyzing"
  | "generating"
  | "training"
  | "refining"
  | "finalizing";
type ActivityTone = "neutral" | "accent" | "success" | "warn";
type DemoLabel = "in_scope" | "out_of_scope" | "ambiguous";
type IdleStep = "welcome" | "compose";

interface ActivityItem {
  id: string;
  message: string;
  detail?: string;
  tone: ActivityTone;
  createdAt: string;
}

interface RoundSummary {
  index: number;
  trainF1: number | null;
  holdoutF1: number | null;
}

interface TestResult {
  text: string;
  label: string;
  confidence: number;
  explanation: string;
  probabilities: Record<string, number>;
}

const LABEL_ORDER: DemoLabel[] = ["in_scope", "out_of_scope", "ambiguous"];
const SEED_PAGE_SIZE = 2;
const MANUAL_STAGE_TIMING = {
  analyzeAtMs: 1600,
  generateAtMs: 3600,
  minTrainingAtMs: 5600,
};
const LUCKY_STAGE_TIMING = {
  analyzeAtMs: 900,
  generateAtMs: 2600,
  minTrainingAtMs: 5000,
};

const EMPTY_SEED_PREVIEW: Record<DemoLabel, string[]> = {
  in_scope: [],
  out_of_scope: [],
  ambiguous: [],
};

const EMPTY_SEED_COUNTS: Record<DemoLabel, number> = {
  in_scope: 0,
  out_of_scope: 0,
  ambiguous: 0,
};

function truncate(text: string, max: number): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max).trimEnd()}…`;
}

function labelDisplay(label: string): string {
  if (label === "in_scope") return "In Scope";
  if (label === "out_of_scope") return "Out of Scope";
  return "Ambiguous";
}

function formatSeedPreview(examples: Example[]): Record<DemoLabel, string[]> {
  const liveExamples = examples.filter((example) => example.approved && example.split !== "holdout");
  return LABEL_ORDER.reduce<Record<DemoLabel, string[]>>((acc, label) => {
    const matching = liveExamples.filter((example) => example.label === label);
    const preferred = matching.filter((example) => example.source !== "human_seed");
    const fallback = matching.filter((example) => example.source === "human_seed");
    acc[label] = [...preferred, ...fallback]
      .slice(0, 8)
      .map((example) => example.text);
    return acc;
  }, { ...EMPTY_SEED_PREVIEW });
}

function countSeedExamples(examples: Example[]): Record<DemoLabel, number> {
  return LABEL_ORDER.reduce<Record<DemoLabel, number>>((acc, label) => {
    acc[label] = examples.filter(
      (example) => example.source === "human_seed" && example.label === label,
    ).length;
    return acc;
  }, { ...EMPTY_SEED_COUNTS });
}

function countLiveExamples(examples: Example[]): number {
  return examples.filter((example) => example.approved && example.split !== "holdout").length;
}

function formatPercent(value: number | null): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function formatMetric(value: number | null): string {
  if (value == null) return "—";
  return value.toFixed(3);
}

function formatDuration(totalSeconds: number): string {
  if (totalSeconds <= 0) return "0m 00s";
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}m ${seconds.toString().padStart(2, "0")}s`;
}

function compactDuration(totalSeconds: number): string {
  if (totalSeconds < 60) return `${totalSeconds}s`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}m ${seconds.toString().padStart(2, "0")}s`;
}

function formatClock(iso: string): string {
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(iso));
}

function formatEventTypeLabel(eventType: string): string {
  return eventType
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatPayload(payload: Record<string, unknown> | null | undefined): string | null {
  if (!payload) return null;
  if (Object.keys(payload).length === 0) return null;
  return JSON.stringify(payload, null, 2);
}

function extractEvaluationMetric(payload: Record<string, unknown>, key: string): number | null {
  const direct = payload[key];
  if (typeof direct === "number") return direct;
  const evaluation = payload.evaluation;
  if (evaluation && typeof evaluation === "object") {
    const record = evaluation as Record<string, unknown>;
    if (typeof record[key] === "number") {
      return record[key] as number;
    }
  }
  return null;
}

function upsertRoundSummary(
  previous: RoundSummary[],
  nextSummary: RoundSummary,
): RoundSummary[] {
  const next = previous.filter((round) => round.index !== nextSummary.index);
  next.push(nextSummary);
  next.sort((a, b) => a.index - b.index);
  return next;
}

function labelToneClass(label: string): string {
  if (label === "in_scope") return styles.toneInScope;
  if (label === "out_of_scope") return styles.toneOutOfScope;
  return styles.toneAmbiguous;
}
function BrandMark() {
  return (
    <span className={styles.brandMark} aria-hidden="true">
      <span className={styles.brandMarkCore} />
      <span className={styles.brandMarkOrbit} />
    </span>
  );
}

function ProcessingShell(props: {
  description: string;
  currentStage: WorkflowStageId;
  launchMode: LaunchMode;
  latestEvent: RunEvent | null;
  exampleCount: number;
  roundBudget: number;
  rounds: RoundSummary[];
  bestRoundIndex: number | null;
  seedPreview: Record<DemoLabel, string[]>;
  seedCounts: Record<DemoLabel, number>;
  seedPage: Record<DemoLabel, number>;
  onChangeSeedPage: (label: DemoLabel, direction: -1 | 1) => void;
  activityFeed: ActivityItem[];
  liveEvents: RunEvent[];
  elapsedSeconds: number;
  onOpenDetails: () => void;
  agentModel: string;
}) {
  const terminalRef = useRef<HTMLDivElement>(null);
  const terminalShouldFollowRef = useRef(true);
  const logEvents = props.liveEvents.filter((event) => event.message);

  useLayoutEffect(() => {
    const terminal = terminalRef.current;
    if (!terminal) return;

    const frame = window.requestAnimationFrame(() => {
      if (!terminalShouldFollowRef.current) return;
      if (terminal.scrollHeight <= terminal.clientHeight) {
        terminal.scrollTop = 0;
        return;
      }
      terminal.scrollTop = terminal.scrollHeight - terminal.clientHeight;
    });

    return () => window.cancelAnimationFrame(frame);
  }, [logEvents.length]);

  function handleTerminalScroll() {
    const terminal = terminalRef.current;
    if (!terminal) return;
    terminalShouldFollowRef.current =
      terminal.scrollHeight - terminal.scrollTop - terminal.clientHeight < 24;
  }

  return (
    <section className={styles.takeoverBody} aria-label="Processing experience">
      <div className={`${styles.composerShell} ${styles.processingShell}`}>
        <section className={`${styles.composerCard} ${styles.processingCard}`} style={{ animation: "demoScaleIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) both" }}>
          <header className={styles.composerHeader}>
            <div className={styles.productLockup}>
              <BrandMark />
              <div>
                <p className={styles.productEyebrow}>Agent Boundary</p>
                <h1 className={styles.productName}>Live run</h1>
              </div>
            </div>
            <button className={styles.ghostButton} onClick={props.onOpenDetails} type="button">
              Open trace
            </button>
          </header>

          <div className={styles.processingTerminalWrap}>
            <div className={`terminal-panel ${styles.processingTerminal}`}>
              <div className={`terminal-header ${styles.processingTerminalHeader}`}>
                <div className="terminal-header-left">
                  <span className={styles.terminalDot} />
                  <span className="terminal-title">Agent Terminal</span>
                </div>
                <div className={styles.processingTerminalMeta}>
                  <span className={styles.processingTerminalBadge}>{compactDuration(props.elapsedSeconds)}</span>
                </div>
              </div>

              <div
                className={`terminal-body ${styles.processingTerminalBody}`}
                ref={terminalRef}
                onScroll={handleTerminalScroll}
              >
                {logEvents.length === 0 ? (
                  <div className="term-line">
                    <span className="term-ts">{compactDuration(props.elapsedSeconds)}</span>
                    <span className={styles.processingTerminalInfo}>·</span>
                    <span className="term-dim">Waiting for first event…</span>
                  </div>
                ) : (
                  logEvents.map((event) => (
                    <div key={event.id} className="term-line">
                      <span className="term-ts">{formatClock(event.created_at)}</span>
                      <span
                        className={
                          event.event_type === "run_completed" || event.event_type === "checkpoint_promoted"
                            ? styles.processingTerminalDone
                            : event.event_type.includes("failed")
                              ? styles.processingTerminalWarn
                              : event.event_type === "review_started" || event.event_type === "run_reviewed"
                                ? styles.processingTerminalInfo
                                : styles.processingTerminalActive
                        }
                      >
                        {event.event_type === "checkpoint_promoted" ? "★" : event.event_type.includes("failed") ? "!" : "•"}
                      </span>
                      <span className={styles.processingTerminalEvent}>{event.message}</span>
                    </div>
                  ))
                )}

                <div className="term-line term-cursor-line">
                  <span className="term-ts">&nbsp;</span>
                  <span className="term-blink">█</span>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </section>
  );
}

function ResultShell(props: {
  description: string;
  exampleCount: number;
  bestF1: number | null;
  bestHoldoutF1: number | null;
  bestRoundIndex: number | null;
  roundBudget: number;
  rounds: RoundSummary[];
  durationSeconds: number;
  testInput: string;
  onTestInputChange: (value: string) => void;
  onTestKeyDown: (event: KeyboardEvent<HTMLInputElement>) => void;
  onClassify: () => void;
  classifying: boolean;
  latestResult: TestResult | null;
  olderResults: TestResult[];
  suggestionChips: Array<{ text: string; label: DemoLabel }>;
  onSuggestionClick: (text: string) => void;
  probabilityEntries: Array<{ label: DemoLabel; value: number }>;
  onStartAnother: () => void;
  onOpenDetails: () => void;
  testError: string;
  testInputRef: RefObject<HTMLInputElement | null>;
}) {
  return (
    <section className={styles.takeoverBody} aria-label="Completed experience">
      <div className={styles.composerShell}>
        <section className={`${styles.composerCard} ${styles.resultCard}`} style={{ animation: "demoScaleIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) both" }}>
          <header className={styles.composerHeader}>
            <div className={styles.productLockup}>
              <BrandMark />
              <div>
                <p className={styles.productEyebrow}>Agent Boundary</p>
                <h1 className={styles.productName}>Your classifier is live</h1>
              </div>
            </div>
            <button className={styles.ghostButton} onClick={props.onOpenDetails} type="button">
              View trace
            </button>
          </header>

          <div className={styles.composerCopy}>
            <p className={styles.stageCaption} style={{ color: "#10b981" }}>✓ Ready for production</p>
            <div className={styles.resultPromptCard}>
              <span className={styles.resultPromptLabel}>Original prompt</span>
              <p className={styles.resultPromptText}>
                {props.description || "No original prompt captured."}
              </p>
            </div>
          </div>

          <div className={styles.resultMiniMetrics} style={{ background: "rgba(0,0,0,0.03)", padding: "12px 16px", borderRadius: "14px" }}>
            <span><strong>F1</strong> {formatMetric(props.bestF1)}</span>
            <span style={{ opacity: 0.3 }}>|</span>
            <span><strong>Holdout F1</strong> {formatMetric(props.bestHoldoutF1)}</span>
            <span style={{ opacity: 0.3 }}>|</span>
            <span><strong>Dataset</strong> {props.exampleCount} examples</span>
            <span style={{ opacity: 0.3 }}>|</span>
            <span><strong>Runtime</strong> {formatDuration(props.durationSeconds)}</span>
            <span style={{ opacity: 0.3 }}>|</span>
            <span><strong>Promotion</strong> {props.bestRoundIndex != null ? `Round ${props.bestRoundIndex} after 3 rounds` : "Pending"}</span>
          </div>

          <div className={styles.resultComposer}>
            <div className={styles.composerRow}>
              <input
                id="classifier-test-input"
                className={styles.composerInput}
                ref={props.testInputRef}
                type="text"
                autoComplete="off"
                inputMode="text"
                placeholder="Type a ticket or support message…"
                value={props.testInput}
                onChange={(event) => props.onTestInputChange(event.target.value)}
                onKeyDown={props.onTestKeyDown}
                style={{ borderRadius: "16px" }}
              />
              <button
                className={styles.primaryButton}
                onClick={props.onClassify}
                disabled={props.classifying || !props.testInput.trim()}
                type="button"
                style={{ minHeight: "58px" }}
              >
                <span className={styles.buttonLabel}>Run test</span>
                {props.classifying && <span className={styles.buttonSpinner} aria-hidden="true" />}
              </button>
            </div>
            {props.testError && (
              <p className={styles.inlineError} role="alert">
                {props.testError}
              </p>
            )}
          </div>

          <div className={styles.resultSurface}>
            {props.latestResult ? (
              <div className={styles.simpleResult} style={{ animation: "demoFadeUp 0.4s ease both" }}>
                <div className={styles.featuredTop}>
                  <span className={`${styles.seedBadge} ${labelToneClass(props.latestResult.label)}`}>
                    {labelDisplay(props.latestResult.label)}
                  </span>
                  <span className={styles.featuredConfidence}>
                    Confidence: <strong>{formatPercent(props.latestResult.confidence)}</strong>
                  </span>
                </div>
                <p className={styles.featuredText}>"{props.latestResult.text}"</p>
                <p className={styles.featuredExplanation}>{props.latestResult.explanation}</p>
              </div>
            ) : (
              <div className={styles.emptyState}>
                <p className={styles.emptyTitle}>Try an example</p>
                <div className={styles.simpleSuggestionList}>
                  {props.suggestionChips.slice(0, 3).map((item) => (
                    <button
                      key={`${item.label}-${item.text}`}
                      className={styles.simpleSuggestion}
                      onClick={() => props.onSuggestionClick(item.text)}
                      type="button"
                      style={{ transition: "all 0.2s ease" }}
                    >
                      <span className={`${styles.seedBadge} ${labelToneClass(item.label)}`}>
                        {labelDisplay(item.label)}
                      </span>
                      <span style={{ fontSize: "14px" }}>{truncate(item.text, 80)}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className={styles.resultActionRow}>
            <button
              className={`${styles.secondaryButton} ${styles.resultActionButton} ${styles.resultActionLeft}`}
              onClick={props.onStartAnother}
              type="button"
            >
              Start another run
            </button>
            <Link
              className={`${styles.secondaryButton} ${styles.resultActionButton} ${styles.resultActionCenter}`}
              href="/read-more"
            >
              Read more
            </Link>
            <div className={styles.resultFooterLinks}>
              <a
                className={styles.resultFooterLink}
                href="https://kyle-mirich.vercel.app"
                rel="noreferrer"
                target="_blank"
              >
                Portfolio
              </a>
              <a
                className={styles.resultFooterLink}
                href="https://www.linkedin.com/in/kyle-mirich/"
                rel="noreferrer"
                target="_blank"
              >
                LinkedIn
              </a>
              <a
                className={styles.resultFooterLink}
                href="https://github.com/kyle-mirich"
                rel="noreferrer"
                target="_blank"
              >
                GitHub
              </a>
            </div>
          </div>
        </section>
      </div>
    </section>
  );
}

function ErrorShell(props: {
  description: string;
  error: string;
  onRetry: () => void;
  onEditBrief: () => void;
  logLines: string[];
}) {
  return (
    <section className={styles.errorShell} aria-label="Run failed">
      <div className={styles.errorCard}>
        <span className={`${styles.statusPill} ${styles.statusPillWarn}`}>Run interrupted</span>
        <h2 className={styles.errorTitle}>The workflow stopped before the classifier was ready.</h2>
        <p className={styles.errorBody}>
          {props.error || "Something unexpected happened while the backend was processing this run."}
        </p>
        <div className={styles.errorBrief}>
          <span className={styles.panelLabel}>Original brief</span>
          <p className={styles.briefText}>{props.description || "No brief captured."}</p>
        </div>
        {props.logLines.length > 0 && (
          <pre className={styles.errorLog}>{props.logLines.slice(-8).join("\n")}</pre>
        )}
        <div className={styles.errorActions}>
          <button className={styles.primaryButton} onClick={props.onRetry} type="button">
            <span className={styles.buttonLabel}>Retry run</span>
          </button>
          <button className={styles.secondaryButton} onClick={props.onEditBrief} type="button">
            Return to editor
          </button>
        </div>
      </div>
    </section>
  );
}

function LuckyPromptModal(props: { prompt: string; loading: boolean; closing: boolean }) {
  const [displayed, setDisplayed] = useState("");
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!props.prompt) {
      setDisplayed("");
      return;
    }
    setDisplayed("");
    let i = 0;
    function tick() {
      i++;
      setDisplayed(props.prompt.slice(0, i));
      if (i < props.prompt.length) {
        timerRef.current = setTimeout(tick, 18 + Math.random() * 14);
      }
    }
    timerRef.current = setTimeout(tick, 18);
    return () => {
      if (timerRef.current !== null) clearTimeout(timerRef.current);
    };
  }, [props.prompt]);

  const typing = displayed.length < props.prompt.length;

  return (
    <>
      <div className={styles.drawerBackdrop} aria-hidden="true" />
      <div
        className={`${styles.luckyModal} ${props.closing ? styles.luckyModalClosing : ""}`}
        aria-label="Feeling Lucky"
        role="dialog"
        aria-modal="true"
      >
        <p className={styles.luckyModalEyebrow}>Feeling Lucky</p>
        <h2 className={styles.luckyModalTitle}>
          {props.loading ? "Choosing a scope" : "Selected scope"}
        </h2>
        <div className={styles.luckyModalContent}>
          {props.loading ? (
            <div className={styles.luckyModalLoading}>
              <span className={styles.luckyModalSpinner} aria-hidden="true" />
              <p className={styles.luckyModalBody}>
                Finding a realistic chatbot boundary for the demo…
              </p>
            </div>
          ) : (
            <p className={styles.luckyModalBody}>
              <span aria-hidden="true" style={{ visibility: "hidden" }}>{props.prompt}</span>
              <span className={styles.luckyModalTyped} aria-live="polite">
                {displayed}
                {typing && <span className={styles.luckyModalCursor} aria-hidden="true" />}
              </span>
            </p>
          )}
        </div>
      </div>
    </>
  );
}

function TraceDrawer(props: {
  open: boolean;
  onClose: () => void;
  description: string;
  rounds: RoundSummary[];
  bestRoundIndex: number | null;
  bestF1: number | null;
  exampleCount: number;
  logLines: string[];
  liveEvents: RunEvent[];
  runDetail: RunDetail | null;
}) {
  if (!props.open) return null;

  const traceEvents = props.runDetail?.events?.length
    ? props.runDetail.events
    : props.liveEvents;
  const holdoutSummary = formatPayload(props.runDetail?.holdout_summary);
  const roundCount = props.rounds.length;
  const eventCount = traceEvents.length;

  return (
    <>
      <button
        aria-label="Close training trace"
        className={styles.drawerBackdrop}
        onClick={props.onClose}
        type="button"
      />
      <aside className={styles.drawer} aria-label="Training trace" aria-modal="true" role="dialog">
        <div className={styles.drawerShell}>
          <header className={styles.drawerHeader}>
            <div className={styles.drawerHeaderCopy}>
              <p className={styles.panelLabel}>Training trace</p>
              <h2 className={styles.drawerTitle}>Review the run end to end</h2>
              <p className={styles.drawerIntro}>
                Step back through the original prompt, generated plan, event timeline, and model
                results without leaving the modal.
              </p>
            </div>
            <button className={styles.drawerClose} onClick={props.onClose} type="button">
              Close
            </button>
          </header>

          <section className={styles.traceHero} aria-label="Original prompt summary">
            <div className={styles.traceHeroCopy}>
              <span className={styles.traceHeroLabel}>Original prompt</span>
              <p className={styles.traceHeroPrompt}>{props.description || "No brief captured."}</p>
            </div>
            <div className={styles.traceHeroStats}>
              <div className={styles.traceHeroStat}>
                <span className={styles.traceHeroStatLabel}>Best F1</span>
                <strong className={styles.traceHeroStatValue}>{formatMetric(props.bestF1)}</strong>
              </div>
              <div className={styles.traceHeroStat}>
                <span className={styles.traceHeroStatLabel}>Examples</span>
                <strong className={styles.traceHeroStatValue}>{props.exampleCount}</strong>
              </div>
              <div className={styles.traceHeroStat}>
                <span className={styles.traceHeroStatLabel}>Rounds</span>
                <strong className={styles.traceHeroStatValue}>{roundCount}</strong>
              </div>
              <div className={styles.traceHeroStat}>
                <span className={styles.traceHeroStatLabel}>Events</span>
                <strong className={styles.traceHeroStatValue}>{eventCount}</strong>
              </div>
            </div>
          </section>

          <div className={styles.drawerBody}>
            <details className={styles.traceDisclosure} open>
              <summary className={styles.traceSummary}>
                <span className={styles.traceHeading}>Overview</span>
                <span className={styles.traceSummaryMeta}>Key metrics</span>
              </summary>
              <div className={styles.tracePanelBody}>
                <div className={styles.summaryCard}>
                  <div className={styles.summaryRow}>
                    <span>Best F1</span>
                    <strong>{formatMetric(props.bestF1)}</strong>
                  </div>
                  <div className={styles.summaryRow}>
                    <span>Examples generated</span>
                    <strong>{props.exampleCount}</strong>
                  </div>
                  <div className={styles.summaryRow}>
                    <span>Best round</span>
                    <strong>
                      {props.bestRoundIndex != null ? `Round ${props.bestRoundIndex}` : "—"}
                    </strong>
                  </div>
                </div>
                {holdoutSummary && <pre className={styles.traceCode}>{holdoutSummary}</pre>}
              </div>
            </details>

            {props.rounds.length > 0 && (
              <details className={styles.traceDisclosure} open>
                <summary className={styles.traceSummary}>
                  <span className={styles.traceHeading}>Round metrics</span>
                  <span className={styles.traceSummaryMeta}>{props.rounds.length} rounds</span>
                </summary>
                <div className={styles.tracePanelBody}>
                  <table className={styles.traceTable}>
                    <thead>
                      <tr>
                        <th>Round</th>
                        <th>Train F1</th>
                        <th>Holdout F1</th>
                      </tr>
                    </thead>
                    <tbody>
                      {props.rounds.map((round) => (
                        <tr key={round.index}>
                          <td>
                            Round {round.index}
                            {props.bestRoundIndex === round.index && (
                              <span className={styles.bestTag}>Best</span>
                            )}
                          </td>
                          <td>{formatMetric(round.trainF1)}</td>
                          <td>{formatMetric(round.holdoutF1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            )}

            {traceEvents.length > 0 && (
              <details className={styles.traceDisclosure} open>
                <summary className={styles.traceSummary}>
                  <span className={styles.traceHeading}>Event timeline</span>
                  <span className={styles.traceSummaryMeta}>{traceEvents.length} events</span>
                </summary>
                <div className={styles.tracePanelBody}>
                  <div className={styles.traceEventList}>
                    {traceEvents.map((event) => {
                      const payload = formatPayload(event.payload);
                      return (
                        <details className={styles.traceEventItem} key={event.id}>
                          <summary className={styles.traceEventSummary}>
                            <div className={styles.traceEventTop}>
                              <span className={styles.traceEventType}>
                                {formatEventTypeLabel(event.event_type)}
                              </span>
                              <span className={styles.traceEventTime}>
                                {formatClock(event.created_at)}
                              </span>
                            </div>
                            <p className={styles.traceEventMessage}>
                              {event.message || "No event message captured."}
                            </p>
                          </summary>
                          {payload && <pre className={styles.traceCode}>{payload}</pre>}
                        </details>
                      );
                    })}
                  </div>
                </div>
              </details>
            )}

            {props.runDetail?.plan_markdown && (
              <details className={styles.traceDisclosure}>
                <summary className={styles.traceSummary}>
                  <span className={styles.traceHeading}>Run plan</span>
                  <span className={styles.traceSummaryMeta}>Prompt + structure</span>
                </summary>
                <div className={styles.tracePanelBody}>
                  <pre className={styles.traceCode}>{props.runDetail.plan_markdown}</pre>
                </div>
              </details>
            )}

            {props.runDetail?.review_markdown && (
              <details className={styles.traceDisclosure}>
                <summary className={styles.traceSummary}>
                  <span className={styles.traceHeading}>Final review</span>
                  <span className={styles.traceSummaryMeta}>Model selection</span>
                </summary>
                <div className={styles.tracePanelBody}>
                  <pre className={styles.traceCode}>{props.runDetail.review_markdown}</pre>
                </div>
              </details>
            )}

            <details className={styles.traceDisclosure}>
              <summary className={styles.traceSummary}>
                <span className={styles.traceHeading}>System log</span>
                <span className={styles.traceSummaryMeta}>
                  {props.logLines.length > 0 ? `${props.logLines.length} lines` : "No logs"}
                </span>
              </summary>
              <div className={styles.tracePanelBody}>
                <pre className={styles.traceCode}>
                  {props.logLines.length > 0 ? props.logLines.join("\n") : "No log lines captured."}
                </pre>
              </div>
            </details>
          </div>
        </div>
      </aside>
    </>
  );
}

export default function HomePage() {
  const descriptionRef = useRef<HTMLTextAreaElement>(null);
  const testInputRef = useRef<HTMLInputElement>(null);
  const setupTimersRef = useRef<number[]>([]);
  const idleTransitionTimerRef = useRef<number | null>(null);
  const activityIndexRef = useRef(0);
  const startedAtRef = useRef<number | null>(null);
  const stageGateUntilRef = useRef<number>(0);

  const [view, setView] = useState<ViewState>("idle");
  const [idleStep, setIdleStep] = useState<IdleStep>("welcome");
  const [renderedIdleStep, setRenderedIdleStep] = useState<IdleStep>("welcome");
  const [idleStepPhase, setIdleStepPhase] = useState<"entered" | "exiting" | "entering">("entered");
  const [launchMode, setLaunchMode] = useState<LaunchMode>("manual");
  const [currentStage, setCurrentStage] = useState<WorkflowStageId>("starting");
  const [description, setDescription] = useState("");
  const [error, setError] = useState("");
  const [testError, setTestError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [luckyLoading, setLuckyLoading] = useState(false);
  const [projectId, setProjectId] = useState("");
  const [runId, setRunId] = useState("");
  const [roundBudget, setRoundBudget] = useState(3);
  const [agentModel, setAgentModel] = useState("gpt-5.4-mini");
  const [seedPreview, setSeedPreview] =
    useState<Record<DemoLabel, string[]>>(EMPTY_SEED_PREVIEW);
  const [seedCounts, setSeedCounts] =
    useState<Record<DemoLabel, number>>(EMPTY_SEED_COUNTS);
  const [seedPage, setSeedPage] = useState<Record<DemoLabel, number>>({
    in_scope: 0,
    out_of_scope: 0,
    ambiguous: 0,
  });
  const [activityFeed, setActivityFeed] = useState<ActivityItem[]>([]);
  const [liveEvents, setLiveEvents] = useState<RunEvent[]>([]);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [rounds, setRounds] = useState<RoundSummary[]>([]);
  const [bestF1, setBestF1] = useState<number | null>(null);
  const [bestRoundIndex, setBestRoundIndex] = useState<number | null>(null);
  const [exampleCount, setExampleCount] = useState(0);
  const [durationSeconds, setDurationSeconds] = useState(0);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null);

  const [testInput, setTestInput] = useState("");
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [classifying, setClassifying] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [luckyPreviewPrompt, setLuckyPreviewPrompt] = useState("");
  const [luckyModalClosing, setLuckyModalClosing] = useState(false);

  function clearSetupTimers() {
    setupTimersRef.current.forEach((timer) => window.clearTimeout(timer));
    setupTimersRef.current = [];
    stageGateUntilRef.current = 0;
  }

  function shouldHoldIntroStages(): boolean {
    return Date.now() < stageGateUntilRef.current;
  }

  function pushActivity(
    message: string,
    detail = "",
    tone: ActivityTone = "neutral",
    createdAt = new Date().toISOString(),
  ) {
    const nextItem: ActivityItem = {
      id: `activity-${activityIndexRef.current++}`,
      message,
      detail,
      tone,
      createdAt,
    };

    startTransition(() => {
      setActivityFeed((previous) => [nextItem, ...previous].slice(0, 18));
    });
  }

  function resetRuntimeState(nextDescription = "") {
    clearSetupTimers();
    startedAtRef.current = null;
    setDescription(nextDescription);
    setError("");
    setTestError("");
    setSubmitting(false);
    setLuckyLoading(false);
    setProjectId("");
    setRunId("");
    setRoundBudget(3);
    setAgentModel("gpt-5.4-mini");
    setCurrentStage("starting");
    setSeedPreview(EMPTY_SEED_PREVIEW);
    setSeedCounts(EMPTY_SEED_COUNTS);
    setSeedPage({ in_scope: 0, out_of_scope: 0, ambiguous: 0 });
    setActivityFeed([]);
    setLiveEvents([]);
    setLogLines([]);
    setRounds([]);
    setBestF1(null);
    setBestRoundIndex(null);
    setExampleCount(0);
    setDurationSeconds(0);
    setDrawerOpen(false);
    setRunDetail(null);
    setLuckyPreviewPrompt("");
    setLuckyModalClosing(false);
    setTestInput("");
    setTestResults([]);
    setClassifying(false);
  }

  function beginProcessing(mode: LaunchMode, nextDescription: string) {
    resetRuntimeState(nextDescription);
    setView("processing");
    setLaunchMode(mode);
    setSubmitting(true);
    startedAtRef.current = Date.now();
    if (mode === "lucky") {
      setCurrentStage("starting");
      setLuckyLoading(true);
      pushActivity("Finding a strong demo brief", "Generating a polished starting point.", "accent");
    } else {
      setCurrentStage("starting");
      pushActivity("Booting the classifier workspace", "Preparing the premium workflow shell.", "accent");
    }
  }

  async function fetchRunDetail(runIdentifier: string) {
    const detail = await api.getRun(runIdentifier).catch(() => null);
    if (!detail) return null;
    setRunDetail(detail);
    setRounds(
      detail.rounds.map((round) => ({
        index: round.round_index,
        trainF1:
          typeof round.metrics?.macro_f1 === "number"
            ? (round.metrics.macro_f1 as number)
            : null,
        holdoutF1:
          typeof round.holdout_metrics?.macro_f1 === "number"
            ? (round.holdout_metrics.macro_f1 as number)
            : null,
      })),
    );
    setBestF1(detail.best_macro_f1 ?? null);
    const bestRound = detail.rounds.find((round) => round.id === detail.best_round_id);
    if (bestRound) {
      setBestRoundIndex(bestRound.round_index);
    }
    return detail;
  }

  async function refreshProjectExamples(projectIdentifier: string) {
    const detail = await api.getProject(projectIdentifier);
    setSeedPreview(formatSeedPreview(detail.examples));
    setSeedCounts(countSeedExamples(detail.examples));
    setExampleCount(countLiveExamples(detail.examples));
    return detail;
  }

  async function startQuickStart(
    rawDescription: string,
    options: { skipAnalyzeStep?: boolean } = {},
  ) {
    const trimmed = rawDescription.trim();
    if (!trimmed) return;
    const setupStartedAt = Date.now();
    const stageTiming = options.skipAnalyzeStep ? LUCKY_STAGE_TIMING : MANUAL_STAGE_TIMING;
    stageGateUntilRef.current = setupStartedAt + stageTiming.minTrainingAtMs;

    setDescription(trimmed);
    setError("");

    clearSetupTimers();
    stageGateUntilRef.current = setupStartedAt + stageTiming.minTrainingAtMs;
    if (!options.skipAnalyzeStep) {
      setupTimersRef.current.push(
        window.setTimeout(() => {
          pushActivity("Understanding the request", "Mapping the support boundary and plan.", "accent");
        }, stageTiming.analyzeAtMs),
      );
    }
    setupTimersRef.current.push(
      window.setTimeout(() => {
        pushActivity("Generating the seed dataset", "Creating balanced labeled examples.", "accent");
      }, stageTiming.generateAtMs),
    );

    try {
      const result = await api.quickStart(trimmed);
      setProjectId(result.project_id);
      setRunId(result.run_id);

      const projectDetail = await refreshProjectExamples(result.project_id);
      const remainingDelay = Math.max(
        0,
        stageTiming.minTrainingAtMs - (Date.now() - setupStartedAt),
      );
      if (remainingDelay > 0) {
        await new Promise((resolve) => window.setTimeout(resolve, remainingDelay));
      }

      setRoundBudget(projectDetail.project.max_rounds);
      setAgentModel(projectDetail.project.agent_model);
      stageGateUntilRef.current = 0;
      pushActivity(
        "Seed set ready",
        `${countLiveExamples(projectDetail.examples)} examples prepared for training.`,
        "success",
      );
    } catch (err) {
      setSubmitting(false);
      setLuckyLoading(false);
      setError(err instanceof Error ? err.message : "Something went wrong. Try again.");
      setView("error");
    }
  }

  async function handleSubmit() {
    const trimmed = description.trim();
    if (!trimmed || submitting || luckyLoading) {
      if (!trimmed) {
        setError("Add a classifier brief to start the workflow.");
        descriptionRef.current?.focus();
      }
      return;
    }

    beginProcessing("manual", trimmed);
    await startQuickStart(trimmed);
  }

  async function handleFeelingLucky() {
    if (submitting || luckyLoading) return;

    setLuckyLoading(true);
    setLuckyPreviewPrompt("");
    setLuckyModalClosing(false);
    setError("");
    try {
      await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));
      const result = await api.luckyPrompt();
      setDescription(result.description);
      setLuckyPreviewPrompt(result.description);
      await new Promise((resolve) => window.setTimeout(resolve, 5000));
      setLuckyModalClosing(true);
      await new Promise((resolve) => window.setTimeout(resolve, 420));
      beginProcessing("lucky", result.description);
      setLuckyLoading(false);
      pushActivity("Selected a strong starting brief", truncate(result.description, 120), "success");
      await startQuickStart(result.description, { skipAnalyzeStep: true });
    } catch (err) {
      setLuckyPreviewPrompt("");
      setLuckyModalClosing(false);
      setLuckyLoading(false);
      setError(err instanceof Error ? err.message : "Something went wrong. Try again.");
    }
  }

  function changeSeedPage(label: DemoLabel, direction: -1 | 1) {
    setSeedPage((previous) => {
      const totalPages = Math.max(1, Math.ceil(seedPreview[label].length / SEED_PAGE_SIZE));
      return {
        ...previous,
        [label]: Math.min(totalPages - 1, Math.max(0, previous[label] + direction)),
      };
    });
  }

  async function handleRetry() {
    if (!description.trim()) {
      setView("idle");
      requestAnimationFrame(() => descriptionRef.current?.focus());
      return;
    }

    beginProcessing("manual", description.trim());
    await startQuickStart(description.trim());
  }

  function handleEditBrief() {
    const preservedDescription = description;
    resetRuntimeState(preservedDescription);
    setView("idle");
    setIdleStep("compose");
    requestAnimationFrame(() => descriptionRef.current?.focus());
  }

  function handleStartAnother() {
    resetRuntimeState("");
    setView("idle");
    setIdleStep("compose");
    requestAnimationFrame(() => descriptionRef.current?.focus());
  }

  async function handleClassify(inputText?: string) {
    const text = (inputText ?? testInput).trim();
    if (!text || classifying) return;

    setClassifying(true);
    setTestError("");

    try {
      const result = await api.classify(projectId, text);
      setTestResults((previous) =>
        [
          {
            text,
            label: result.label,
            confidence: result.confidence,
            explanation: result.explanation,
            probabilities: result.probabilities,
          },
          ...previous,
        ].slice(0, 6),
      );
      if (!inputText) setTestInput("");
      pushActivity("Ran a live classification", truncate(text, 84), "success");
    } catch (err) {
      setTestError(err instanceof Error ? err.message : "Unable to classify that message.");
    } finally {
      setClassifying(false);
    }
  }

  function handlePromptKey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      void handleSubmit();
    }
  }

  function handleTestKey(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter") {
      event.preventDefault();
      void handleClassify();
    }
  }

  function recordRunEvent(event: RunEvent, appendLive: boolean) {
    if (appendLive) {
      startTransition(() => {
        setLiveEvents((previous) => {
          if (previous.some((item) => item.id === event.id)) return previous;
          return [...previous, event].slice(-120);
        });
        if (event.message) {
          setLogLines((previous) => {
            if (previous.includes(event.message)) return previous;
            return [...previous, event.message].slice(-240);
          });
        }
      });
    }

    if (event.message) {
      const tone: ActivityTone =
        event.event_type === "run_completed" || event.event_type === "checkpoint_promoted"
          ? "success"
          : event.event_type.includes("failed")
            ? "warn"
            : event.event_type === "review_started" || event.event_type === "run_reviewed"
              ? "accent"
              : "neutral";
      pushActivity(event.message, event.event_type.replaceAll("_", " "), tone, event.created_at);
    }

    const payload = event.payload as Record<string, unknown>;
    const roundIndex = typeof payload.round_index === "number" ? payload.round_index : null;
    const trainF1 = extractEvaluationMetric(payload, "macro_f1");
    const holdoutF1 =
      typeof payload.holdout_macro_f1 === "number" ? (payload.holdout_macro_f1 as number) : null;

    if (event.event_type === "seed_generation_completed") {
      const counts = payload.counts as Record<string, number> | undefined;
      if (counts) {
        setSeedCounts({
          in_scope: counts.in_scope ?? 0,
          out_of_scope: counts.out_of_scope ?? 0,
          ambiguous: counts.ambiguous ?? 0,
        });
        setExampleCount((counts.in_scope ?? 0) + (counts.out_of_scope ?? 0) + (counts.ambiguous ?? 0));
      }
      if (projectId) {
        void refreshProjectExamples(projectId);
      }
      setCurrentStage("generating");
    }

    if (event.event_type === "plan_ready" && !shouldHoldIntroStages()) {
      setCurrentStage("training");
    }

    if (
      event.event_type === "training_started" ||
      event.event_type === "training_complete" ||
      event.event_type === "evaluation_started" ||
      event.event_type === "evaluation_complete" ||
      event.event_type === "run_started"
    ) {
      if (!shouldHoldIntroStages()) {
        setCurrentStage("training");
      }
    }

    if (event.event_type === "round_complete" && roundIndex != null) {
      if (!shouldHoldIntroStages()) {
        setCurrentStage("training");
      }
      setRounds((previous) =>
        upsertRoundSummary(previous, {
          index: roundIndex,
          trainF1,
          holdoutF1,
        }),
      );
    }

    if (
      event.event_type === "review_started" ||
      event.event_type === "review_recorded" ||
      event.event_type === "run_reviewed"
    ) {
      if (!shouldHoldIntroStages()) {
        setCurrentStage("refining");
      }
      if (typeof payload.best_round_index === "number") {
        setBestRoundIndex(payload.best_round_index);
      }
      if (typeof payload.macro_f1 === "number") {
        setBestF1(payload.macro_f1);
      }
    }

    if (
      event.event_type === "checkpoint_promoted" ||
      event.event_type === "run_completed"
    ) {
      stageGateUntilRef.current = 0;
      setCurrentStage("finalizing");
      if (typeof payload.best_round_index === "number") {
        setBestRoundIndex(payload.best_round_index);
      }
      if (typeof payload.best_macro_f1 === "number") {
        setBestF1(payload.best_macro_f1);
      }
    }
  }

  async function handleRunDone(raw: { status: string; best_macro_f1: number | null }) {
    if (raw.status === "failed") {
      setSubmitting(false);
      setLuckyLoading(false);
      setError("The backend reported that the training run failed.");
      setView("error");
      pushActivity("Training failed", "The run ended before the classifier was ready.", "warn");
      return;
    }

    setCurrentStage("finalizing");
    pushActivity("Finalizing the live classifier", "Comparing all 3 rounds and confirming the production checkpoint.", "accent");

    if (startedAtRef.current) {
      setDurationSeconds(Math.max(1, Math.round((Date.now() - startedAtRef.current) / 1000)));
    }
    if (raw.best_macro_f1 != null) {
      setBestF1(raw.best_macro_f1);
    }

    await fetchRunDetail(runId);
    if (projectId) {
      await refreshProjectExamples(projectId);
    }

    setSubmitting(false);
    setLuckyLoading(false);
    window.setTimeout(() => {
      setView("complete");
      pushActivity("Classifier ready", "The live testing surface is now available.", "success");
    }, 700);
  }

  function handleStreamFailure() {
    setSubmitting(false);
    setLuckyLoading(false);
    setError("The live event stream disconnected before the run completed.");
    setView("error");
    pushActivity("Streaming failed", "The live backend connection was interrupted.", "warn");
  }

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    function onKeyDown(event: globalThis.KeyboardEvent) {
      if (event.key === "Escape") {
        setDrawerOpen(false);
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setDrawerOpen(params.get("details") === "1");
  }, []);

  useEffect(() => {
    const url = new URL(window.location.href);
    if (drawerOpen) {
      url.searchParams.set("details", "1");
    } else {
      url.searchParams.delete("details");
    }
    window.history.replaceState({}, "", url);
  }, [drawerOpen]);

  useEffect(() => {
    if (!drawerOpen) return;

    const bodyOverflow = document.body.style.overflow;
    const bodyOverscroll = document.body.style.overscrollBehavior;
    const htmlOverflow = document.documentElement.style.overflow;

    document.body.style.overflow = "hidden";
    document.body.style.overscrollBehavior = "none";
    document.documentElement.style.overflow = "hidden";

    return () => {
      document.body.style.overflow = bodyOverflow;
      document.body.style.overscrollBehavior = bodyOverscroll;
      document.documentElement.style.overflow = htmlOverflow;
    };
  }, [drawerOpen]);

  useEffect(() => {
    if (view === "processing") {
      const interval = window.setInterval(() => {
        if (!startedAtRef.current) return;
        setDurationSeconds(Math.max(1, Math.round((Date.now() - startedAtRef.current) / 1000)));
      }, 1000);
      return () => window.clearInterval(interval);
    }
  }, [view]);

  useEffect(() => {
    if (view === "complete") {
      testInputRef.current?.focus();
    }
  }, [view]);

  useEffect(() => {
    if (view === "idle" && renderedIdleStep === "compose" && idleStepPhase === "entered") {
      requestAnimationFrame(() => descriptionRef.current?.focus());
    }
  }, [view, renderedIdleStep, idleStepPhase]);

  useEffect(() => {
    if (idleTransitionTimerRef.current) {
      window.clearTimeout(idleTransitionTimerRef.current);
      idleTransitionTimerRef.current = null;
    }

    if (idleStep === renderedIdleStep) {
      if (idleStepPhase !== "entered") {
        setIdleStepPhase("entered");
      }
      return;
    }

    setIdleStepPhase("exiting");
    idleTransitionTimerRef.current = window.setTimeout(() => {
      setRenderedIdleStep(idleStep);
      setIdleStepPhase("entering");
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          setIdleStepPhase("entered");
        });
      });
    }, 180);

    return () => {
      if (idleTransitionTimerRef.current) {
        window.clearTimeout(idleTransitionTimerRef.current);
        idleTransitionTimerRef.current = null;
      }
    };
  }, [idleStep, renderedIdleStep, idleStepPhase]);

  useLayoutEffect(() => {
    document.documentElement.style.overflow =
      view === "idle" && !drawerOpen && !luckyLoading && !luckyPreviewPrompt ? "" : "hidden";
    return () => {
      document.documentElement.style.overflow = "";
    };
  }, [view, drawerOpen, luckyLoading, luckyPreviewPrompt]);

  useEffect(() => {
    if (view !== "processing" || !runId) return;

    const sessionId = getClientSessionId();
    const url = `${API_BASE_URL}/runs/${runId}/events/stream?session_id=${encodeURIComponent(sessionId)}`;
    let source: EventSource | null = null;

    async function connect() {
      const history = await api.getRunEvents(runId).catch(() => []);
      startTransition(() => {
        setLiveEvents(history.slice(-120));
        setLogLines(history.map((event) => event.message).slice(-240));
      });
      history.forEach((event) => recordRunEvent(event, false));

      source = new EventSource(url);
      source.addEventListener("run_event", (event: MessageEvent) => {
        const parsed = JSON.parse(event.data) as RunEvent;
        recordRunEvent(parsed, true);
      });
      source.addEventListener("run_done", (event: MessageEvent) => {
        source?.close();
        void handleRunDone(JSON.parse(event.data) as { status: string; best_macro_f1: number | null });
      });
      source.onerror = () => {
        source?.close();
        handleStreamFailure();
      };
    }

    void connect();
    return () => source?.close();
  }, [view, runId]);

  const latestEvent = liveEvents[liveEvents.length - 1] ?? null;
  const latestResult = testResults[0] ?? null;
  const olderResults = testResults.slice(1);
  const bestHoldoutF1 =
    rounds.find((round) => round.index === bestRoundIndex)?.holdoutF1 ?? null;
  const suggestionChips = LABEL_ORDER.flatMap((label) =>
    seedPreview[label].slice(0, 2).map((text) => ({ text, label })),
  ).filter((item) => item.text);
  const probabilityEntries = latestResult
    ? LABEL_ORDER.map((label) => ({
      label,
      value: latestResult.probabilities[label] ?? 0,
    }))
    : [];

  const idleContent =
    renderedIdleStep === "welcome" ? (
      <section className={styles.welcomeShell}>
        <div className={styles.welcomeHero}>
          <div className={styles.landingNav}>
            <span className={styles.landingNavLabel}>Kyle Mirich</span>
            <div className={styles.landingNavLinks}>
              <a
                className={styles.landingNavLink}
                href="https://kyle-mirich.vercel.app"
                rel="noreferrer"
                target="_blank"
              >
                Portfolio
              </a>
              <a
                className={styles.landingNavLink}
                href="https://github.com/kyle-mirich"
                rel="noreferrer"
                target="_blank"
              >
                GitHub
              </a>
              <a
                className={styles.landingNavLink}
                href="https://www.linkedin.com/in/kyle-mirich/"
                rel="noreferrer"
                target="_blank"
              >
                LinkedIn
              </a>
            </div>
          </div>

          <div className={styles.welcomeMain}>
            <div className={styles.welcomeLead}>
              <div className={styles.welcomeBrand}>
                <BrandMark />
                <div className={styles.welcomeBrandCopy}>
                  <p className={styles.welcomeKicker}>Kyle Mirich · AI Engineer</p>
                  <p className={styles.welcomeBrandName}>Agent Boundary</p>
                  <p className={styles.welcomeSubhead}>
                    A scope classifier that screens out off-topic chats before they reach a
                    customer-facing bot or human support agent.
                  </p>
                </div>
              </div>

              <div className={styles.welcomeCopy}>
                <p className={styles.welcomeEyebrow}>Agent scope boundary</p>
                <h1 className={styles.welcomeTitle}>Keep agents on topic.</h1>
                <p className={styles.welcomeBody}>
                  Agent Boundary runs a self-improving loop that generates examples, trains
                  across multiple rounds, compares checkpoints, and deploys the strongest
                  classifier as the boundary layer in front of support.
                </p>
                <ul className={styles.welcomeProofList}>
                  <li className={styles.welcomeProofItem}>Generates labeled data from a scope definition</li>
                  <li className={styles.welcomeProofItem}>Fine-tunes a BERT classifier across multiple rounds and compares checkpoints</li>
                  <li className={styles.welcomeProofItem}>Serves the best classifier for production screening</li>
                </ul>
              </div>
            </div>

            <div className={styles.welcomeStepsSection}>
              <div className={styles.welcomeStepsTimeline}>
                <div
                  className={styles.welcomeStep}
                  role="note"
                  tabIndex={0}
                  aria-label="Define scope step details"
                >
                  <div className={styles.stepNumber}>1</div>
                  <div className={styles.stepLabel}>Define scope</div>
                  <div className={styles.hoverTooltip} role="tooltip">
                    Write the in-scope behavior you want the classifier to learn. This prompt
                    becomes the source for synthetic examples and labeling rules.
                  </div>
                </div>
                <div className={styles.stepConnector}></div>
                <div
                  className={styles.welcomeStep}
                  role="note"
                  tabIndex={0}
                  aria-label="Generate examples step details"
                >
                  <div className={styles.stepNumber}>2</div>
                  <div className={styles.stepLabel}>Generate examples</div>
                  <div className={styles.hoverTooltip} role="tooltip">
                    The pipeline expands the scope definition into labeled on-topic and
                    off-topic examples so the model has training data before any fine-tuning
                    run starts.
                  </div>
                </div>
                <div className={styles.stepConnector}></div>
                <div
                  className={styles.welcomeStep}
                  role="note"
                  tabIndex={0}
                  aria-label="Train classifier step details"
                >
                  <div className={styles.stepNumber}>3</div>
                  <div className={styles.stepLabel}>Train classifier</div>
                  <div className={styles.hoverTooltip} role="tooltip">
                    Each round fine-tunes the classifier, evaluates checkpoints, and keeps the
                    strongest model. The score shown in the example comes back from this model.
                  </div>
                </div>
                <div className={styles.stepConnector}></div>
                <div
                  className={styles.welcomeStep}
                  role="note"
                  tabIndex={0}
                  aria-label="Deploy boundary step details"
                >
                  <div className={styles.stepNumber}>4</div>
                  <div className={styles.stepLabel}>Deploy boundary</div>
                  <div className={styles.hoverTooltip} role="tooltip">
                    Deployment here means serving the trained classifier as a boundary service
                    other products can call before they handle the request.
                  </div>
                </div>
              </div>
            </div>

            <div className={styles.welcomeSide}>
              <div className={styles.welcomeExample}>
                <div className={styles.welcomeExampleHeader}>
                  <p className={styles.welcomeExampleLabel}>Use Case</p>
                </div>

                <div className={styles.chatContainer}>
                  <div className={styles.userLine}>
                    <div
                      className={`${styles.chatMessage} ${styles.userMessage} ${styles.annotatedMessage}`}
                      role="note"
                      tabIndex={0}
                      aria-label="User message classification details"
                    >
                      Can you help me build a Python script to automate my workflow?
                      <span className={styles.classificationBadge}>
                        <span className={styles.classificationValue}>73% Off Topic</span>
                        <div
                          className={`${styles.hoverTooltip} ${styles.hoverTooltipWide}`}
                          role="tooltip"
                        >
                          User prompt gets sent to the classifier (what we are building).
                        </div>
                      </span>
                    </div>
                  </div>

                  <div
                    className={`${styles.chatMessage} ${styles.botMessage} ${styles.annotatedMessage}`}
                    role="note"
                    tabIndex={0}
                    aria-label="Agent response details"
                  >
                    I'm here to only help with questions about our product.
                    <div className={styles.hoverTooltip} role="tooltip">
                      Since the user message was labeled off topic, the agent responds with a
                      hardcoded response.
                    </div>
                  </div>
                </div>
              </div>

              <div className={styles.welcomeActions}>
                <button
                  className={`${styles.primaryButton} ${styles.welcomePrimaryButton}`}
                  onClick={() => setIdleStep("compose")}
                  type="button"
                >
                  <span className={styles.buttonLabel}>Start Live Training</span>
                </button>
              </div>

              <div className={styles.welcomeStackRow}>
                <span className={styles.welcomeStackLabel}>Built with</span>
                <div className={styles.welcomeStackList}>
                  <span className={styles.welcomeStackItem}>Next.js</span>
                  <span className={styles.welcomeStackItem}>FastAPI</span>
                  <span className={styles.welcomeStackItem}>LangChain</span>
                  <span className={styles.welcomeStackItem}>LangGraph</span>
                  <span className={styles.welcomeStackItem}>PyTorch</span>
                  <span className={styles.welcomeStackItem}>Agent orchestration</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    ) : (
      <section className={styles.composerShell}>
        <div className={styles.composerCard}>
          <header className={styles.composerHeader}>
            <button
              className={styles.ghostButton}
              onClick={() => setIdleStep("welcome")}
              type="button"
            >
              Back
            </button>
          </header>

          <div className={styles.composerCopy}>
            <p className={styles.composerTitle}>Describe what belongs in scope.</p>
            <p className={styles.composerBody}>
              One sentence is enough. The guided workflow will generate examples, train the
              model, and reveal the final classifier.
            </p>
          </div>

          <div className={styles.launchPanel}>
            <div className={styles.launchHeader}>
              <div>
                <h3 className={styles.cardTitle}>What topics should the chatbot cover?</h3>
              </div>
            </div>

            <label className="sr-only" htmlFor="classifier-description">
              Classifier description
            </label>
            <textarea
              id="classifier-description"
              className={styles.textarea}
              ref={descriptionRef}
              rows={5}
              autoComplete="off"
              name="classifier_description"
              placeholder="Questions about Star Wars lore, characters, timelines, and canon debates…"
              value={description}
              onChange={(event) => {
                setDescription(event.target.value);
                if (error) setError("");
              }}
              onKeyDown={handlePromptKey}
            />
            {error && (
              <p className={styles.inlineError} role="alert">
                {error}
              </p>
            )}

            <div className={styles.launchActions}>
              <button
                className={styles.primaryButton}
                disabled={submitting || luckyLoading || !description.trim()}
                onClick={() => void handleSubmit()}
                type="button"
              >
                <span className={styles.buttonLabel}>Build classifier</span>
                {submitting && <span className={styles.buttonSpinner} aria-hidden="true" />}
              </button>

              <button
                className={styles.luckyButton}
                disabled={submitting || luckyLoading}
                onClick={() => void handleFeelingLucky()}
                type="button"
              >
                <span className={styles.luckySpark} aria-hidden="true" />
                <span className={styles.buttonLabel}>
                  {luckyLoading ? "Picking a scope…" : "Feeling Lucky"}
                </span>
              </button>
            </div>
          </div>
        </div>
      </section>
    );

  return (
    <>
      <a className="skip-link" href="#app-main">
        Skip to main content
      </a>
      <main className={styles.page} id="app-main">
        {!mounted ? null : view === "idle" ? (
          <div
            className={`${styles.idleStage} ${
              idleStepPhase === "exiting"
                ? styles.idleStepExit
                : idleStepPhase === "entering"
                  ? styles.idleStepEnter
                  : styles.idleStepEntered
            }`}
          >
            {idleContent}
          </div>
        ) : view === "processing" ? (
          <div className={styles.takeoverStage}>
            <ProcessingShell
              activityFeed={activityFeed}
              agentModel={agentModel}
              currentStage={currentStage}
              description={description}
              elapsedSeconds={durationSeconds}
              exampleCount={exampleCount}
              liveEvents={liveEvents}
              latestEvent={latestEvent}
              launchMode={launchMode}
              bestRoundIndex={bestRoundIndex}
              onChangeSeedPage={changeSeedPage}
              onOpenDetails={() => setDrawerOpen(true)}
              roundBudget={roundBudget}
              rounds={rounds}
              seedCounts={seedCounts}
              seedPage={seedPage}
              seedPreview={seedPreview}
            />
          </div>
        ) : view === "complete" ? (
          <div className={`${styles.takeoverStage} ${styles.completeStage}`}>
            <ResultShell
              bestF1={bestF1}
              bestHoldoutF1={bestHoldoutF1}
              bestRoundIndex={bestRoundIndex}
              classifying={classifying}
              description={description}
              durationSeconds={durationSeconds}
              exampleCount={exampleCount}
              latestResult={latestResult}
              olderResults={olderResults}
              onClassify={() => void handleClassify()}
              onOpenDetails={() => setDrawerOpen(true)}
              onStartAnother={handleStartAnother}
              onSuggestionClick={(text) => void handleClassify(text)}
              onTestInputChange={setTestInput}
              onTestKeyDown={handleTestKey}
              probabilityEntries={probabilityEntries}
              roundBudget={roundBudget}
              rounds={rounds}
              suggestionChips={suggestionChips}
              testError={testError}
              testInput={testInput}
              testInputRef={testInputRef}
            />
          </div>
        ) : (
          <div className={styles.takeoverStage}>
            <ErrorShell
              description={description}
              error={error}
              logLines={logLines}
              onEditBrief={handleEditBrief}
              onRetry={() => void handleRetry()}
            />
          </div>
        )}

        <TraceDrawer
          bestF1={bestF1}
          bestRoundIndex={bestRoundIndex}
          exampleCount={exampleCount}
          description={description}
          liveEvents={liveEvents}
          logLines={logLines}
          onClose={() => setDrawerOpen(false)}
          open={drawerOpen}
          rounds={rounds}
          runDetail={runDetail}
        />
        {view === "idle" && (luckyLoading || luckyPreviewPrompt) && (
          <LuckyPromptModal
            loading={!luckyPreviewPrompt}
            prompt={luckyPreviewPrompt}
            closing={luckyModalClosing}
          />
        )}
      </main>
    </>
  );
}
