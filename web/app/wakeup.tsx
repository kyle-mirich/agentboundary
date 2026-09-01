"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { checkBackendHealth, isApiBaseUrlConfigured } from "../lib/api";
import styles from "./wakeup.module.css";

type WakeStatus = "checking" | "waking" | "ready" | "unreachable";

const POLL_INTERVAL_MS = 4000;
const MAX_POLL_INTERVAL_MS = 15_000;
const MAX_ATTEMPTS = 20;
const EXPECTED_BOOT_S = 30;
const PATIENCE_HINT_AFTER_S = 60;

function LeafIcon() {
  return (
    <svg
      className={styles.badgeIcon}
      viewBox="0 0 16 16"
      fill="none"
      aria-hidden="true"
      focusable="false"
    >
      <path
        d="M13.5 2.5c-6 0-10 2.5-10 7.5 0 1.2.3 2.3.8 3.2.3.5.9.6 1.3.2.4-.3.4-.9.2-1.3-.1-.2-.2-.4-.2-.6 3.2.3 7.9-.7 7.9-8.5a.5.5 0 0 0-.5-.5ZM4.6 9.4C5.5 6.6 8 5.2 11.9 5c-.5 5-3.6 5.6-6.3 5.4l-1-1Z"
        fill="currentColor"
      />
    </svg>
  );
}

export default function WakeupGate({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<WakeStatus>("checking");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [attempts, setAttempts] = useState(0);
  const [attemptRun, setAttemptRun] = useState(0);

  useEffect(() => {
    // A per-effect `cancelled` flag and a stored timer handle. The previous
    // version shared a `mounted` ref that was reset to `true` on remount, so
    // under Strict Mode the first chain resumed alongside the second and every
    // unhealthy response doubled the number of in-flight poll chains.
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let attempt = 0;

    async function poll() {
      const healthy = await checkBackendHealth();
      if (cancelled) return;

      if (healthy) {
        setStatus("ready");
        return;
      }

      attempt += 1;
      setAttempts(attempt);
      if (attempt >= MAX_ATTEMPTS) {
        setStatus("unreachable");
        return;
      }

      setStatus("waking");
      // Back off so a sleeping server is not hammered by every open tab.
      const delay = Math.min(POLL_INTERVAL_MS * 2 ** Math.floor(attempt / 5), MAX_POLL_INTERVAL_MS);
      timer = setTimeout(poll, delay);
    }

    void poll();
    return () => {
      cancelled = true;
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [attemptRun]);

  const retry = useCallback(() => {
    setAttempts(0);
    setElapsedSeconds(0);
    setStatus("checking");
    setAttemptRun((run) => run + 1);
  }, []);

  useEffect(() => {
    if (status !== "waking") return;
    const startedAt = Date.now();
    const timer = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, [status]);

  // Children always render so the route content stays in the
  // server-rendered HTML. Previously the gate replaced the page until a
  // client-side health check passed, so crawlers and link previews only
  // ever saw "Connecting…".
  const gated = status !== "ready";
  const waking = status === "waking";
  const progress = Math.min(elapsedSeconds / EXPECTED_BOOT_S, 1);

  return (
    <>
      <div aria-hidden={gated} inert={gated}>
        {children}
      </div>
      {gated && (
        <main className={styles.screen} aria-live="polite" role="status">
          <div className={styles.card}>
            <p className={styles.badge}>
              <LeafIcon />
              Sleeping when idle to save energy and costs
            </p>
            {status !== "unreachable" && <div className={styles.spinner} aria-hidden="true" />}
            <h1 className={styles.title}>
              {waking ? "Waking the server…" : status === "unreachable" ? "Server unavailable" : "Connecting…"}
            </h1>
            {status === "unreachable" ? (
              <>
                <p className={styles.message}>
                  The demo server did not respond after {attempts} attempts.{" "}
                  {isApiBaseUrlConfigured
                    ? "It may be restarting."
                    : "NEXT_PUBLIC_API_BASE_URL is not set, so the app is calling localhost instead of the deployed API."}
                </p>
                <button type="button" className="btn btn-primary" onClick={retry}>
                  Try again
                </button>
              </>
            ) : waking ? (
              <>
                <p className={styles.message}>
                  To cut energy use and hosting costs, the demo powers down
                  between visits and spins back up on demand. This usually takes
                  about half a minute.
                </p>
                <div
                  className={styles.progress}
                  role="progressbar"
                  aria-label="Server wake-up progress"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(progress * 100)}
                >
                  <div
                    className={styles.progressFill}
                    style={{ width: `${Math.max(progress * 100, 4)}%` }}
                  />
                </div>
                <p className={styles.counter} role="status">
                  Awake for {elapsedSeconds}
                  &nbsp;second{elapsedSeconds === 1 ? "" : "s"}
                </p>
                {elapsedSeconds >= PATIENCE_HINT_AFTER_S && (
                  <p className={styles.hint}>
                    Still waking — thanks for your patience. The page will continue
                    on its own.
                  </p>
                )}
              </>
            ) : (
              <p className={styles.message}>
                Checking whether the demo server is awake…
              </p>
            )}
          </div>
          </main>
      )}
    </>
  );
}
