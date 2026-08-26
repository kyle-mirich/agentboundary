"use client";

import { useEffect, useRef, useState } from "react";
import { checkBackendHealth } from "../lib/api";
import styles from "./wakeup.module.css";

type WakeStatus = "checking" | "waking" | "ready";

const POLL_INTERVAL_MS = 4000;
const PATIENCE_HINT_AFTER_S = 60;

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

  return (
    <main className={styles.screen} aria-live="polite">
      <div className={styles.card}>
        <div className={styles.spinner} aria-hidden="true" />
        <h1 className={styles.title}>
          {waking ? "Waking the server…" : "Connecting…"}
        </h1>
        {waking && (
          <>
            <p className={styles.message}>
              The demo server sleeps overnight to save compute, and it&rsquo;s
              spinning up right now. This usually takes about half a minute.
            </p>
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
        )}
      </div>
    </main>
  );
}
