"use client";

import { useEffect, useRef, useState } from "react";
import { checkBackendHealth } from "../lib/api";
import styles from "./wakeup.module.css";

type WakeStatus = "checking" | "waking" | "ready";

const POLL_INTERVAL_MS = 4000;
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
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;

    async function poll() {
      const healthy = await checkBackendHealth();
      if (!mounted.current) return;
      if (healthy) {
        setStatus("ready");
      } else {
        setStatus("waking");
        setTimeout(poll, POLL_INTERVAL_MS);
      }
    }

    poll();
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    if (status !== "waking") return;
    const startedAt = Date.now();
    const timer = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, [status]);

  if (status === "ready") {
    return <>{children}</>;
  }

  const waking = status === "waking";
  const progress = Math.min(elapsedSeconds / EXPECTED_BOOT_S, 1);

  return (
    <main className={styles.screen} aria-live="polite">
      <div className={styles.card}>
        <p className={styles.badge}>
          <LeafIcon />
          Sleeping when idle to save energy and costs
        </p>
        <div className={styles.spinner} aria-hidden="true" />
        <h1 className={styles.title}>
          {waking ? "Waking the server…" : "Connecting…"}
        </h1>
        {waking ? (
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
  );
}
