import Link from "next/link";
import BrandMark from "../components/BrandMark";
import type { Metadata } from "next";
import styles from "./page.module.css";

export const metadata: Metadata = {
  title: "Read more | Agent Boundary",
  description:
    "Why Agent Boundary matters in production: classifier guardrails for chatbots, support agents, and scope-based routing.",
};

export default function ReadMorePage() {
  return (
    <>
      <a className="skip-link" href="#read-more-main">
        Skip to main content
      </a>
      <main className={styles.page} id="read-more-main">
        <div className={styles.shell}>
        <header className={styles.topBar}>
          <Link className={styles.brandLink} href="/">
            <BrandMark />
            <div>
              <p className={styles.brandEyebrow}>Agent Boundary</p>
              <p className={styles.brandName}>Read more</p>
            </div>
          </Link>
          <Link className={styles.backLink} href="/">
            Back to demo
          </Link>
        </header>

        <section className={styles.hero}>
          <p className={styles.kicker}>Project explainer</p>
          <div className={styles.heroGrid}>
            <div className={styles.heroCopy}>
              <h1 className={styles.title}>
                A classifier is not just a toy demo. It is a guardrail layer for chatbots and agents.
              </h1>
              <p className={styles.lead}>
                Agent Boundary shows how you can take a user message, score it against scope, and
                decide whether a downstream agent should answer, escalate, or send a prebuilt fallback.
                That makes the project useful anywhere you need a controlled conversation, not just a
                neat model benchmark.
              </p>
              <div className={styles.ctaRow}>
                <Link className={styles.primaryCta} href="/">
                  Back to demo
                </Link>
              </div>
            </div>

            <aside className={styles.aside}>
              <p className={styles.asideLabel}>Routing pattern</p>
              <div className={styles.asideFlow} aria-label="Classifier routing flow">
                <div className={styles.flowStep}>
                  <span>1</span>
                  <p>User question arrives</p>
                </div>
                <div className={styles.flowArrow} />
                <div className={styles.flowStep}>
                  <span>2</span>
                  <p>Classifier checks scope and confidence</p>
                </div>
                <div className={styles.flowArrow} />
                <div className={styles.flowStep}>
                  <span>3</span>
                  <p>Pass to agent, or block and reply with a preset message</p>
                </div>
              </div>
            </aside>
          </div>
        </section>

        <article className={styles.article}>
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Why this matters</h2>
            <p className={styles.sectionBody}>
              The value is control. If you are shipping a customer support assistant that should only
              answer questions about a product, policy, or workflow, a classifier can act as a
              prescreener before the agent ever sees the message. Inputs that clear the threshold get
              routed into the chatbot. Inputs that do not can be redirected to a deterministic fallback
              response, a handoff, or a human review path.
            </p>
          </section>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>What Agent Boundary demonstrates</h2>
            <div className={styles.featureGrid}>
              <div className={styles.feature}>
                <h3>Scoped data collection</h3>
                <p>
                  It starts from the brief and grows a labeled dataset around the exact behavior you
                  want to allow or reject.
                </p>
              </div>
              <div className={styles.feature}>
                <h3>Round-by-round improvement</h3>
                <p>
                  The training loop evaluates F1 and holdout performance so the model is promoted only
                  when it is actually ready.
                </p>
              </div>
              <div className={styles.feature}>
                <h3>Live decisioning</h3>
                <p>
                  A support message can be classified in real time and pushed to the right experience:
                  answer, escalate, or stop.
                </p>
              </div>
              <div className={styles.feature}>
                <h3>Production-friendly guardrails</h3>
                <p>
                  The same pattern works for website chatbots, internal agents, moderation flows, lead
                  triage, and any system that needs a confidence gate.
                </p>
              </div>
            </div>
          </section>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Production architecture</h2>
            <div className={styles.featureGrid}>
              <div className={styles.feature}>
                <h3>Vercel frontend</h3>
                <p>
                  The public demo runs as a Next.js app with browser-scoped sessions, server-sent
                  event streaming, and a trace drawer for reviewing the run after completion.
                </p>
              </div>
              <div className={styles.feature}>
                <h3>Railway API</h3>
                <p>
                  FastAPI owns project state, seed generation, Deep Agents orchestration, training,
                  promotion, health checks, and live classification.
                </p>
              </div>
              <div className={styles.feature}>
                <h3>Supabase Postgres</h3>
                <p>
                  Projects, examples, runs, rounds, and event logs are durable, which lets the UI
                  replay progress and recover from stream reconnects.
                </p>
              </div>
              <div className={styles.feature}>
                <h3>OpenAI + classifier loop</h3>
                <p>
                  The agent plans the experiment, while deterministic backend tools generate data,
                  train the model, evaluate checkpoints, and promote the best round.
                </p>
              </div>
            </div>
          </section>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Operational details</h2>
            <ul className={styles.useCases}>
              <li>
                `GET /health` is used for platform uptime checks; `GET /health/details` reports safe
                database and model diagnostics without exposing secrets.
              </li>
              <li>
                Public quick-start runs are capped per browser session so the OpenAI-backed workflow
                is harder to abuse.
              </li>
              <li>
                Every run writes persisted events, so the terminal view is a replayable trace rather
                than a temporary loading animation.
              </li>
              <li>
                Local development defaults to SQLite, while production uses PostgreSQL through the
                same repository layer.
              </li>
            </ul>
          </section>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Where it fits in the real world</h2>
            <ul className={styles.useCases}>
              <li>
                Customer support bots that should only answer on a defined product surface.
              </li>
              <li>
                Website assistants that need to reject out-of-scope questions before they reach the
                main agent.
              </li>
              <li>
                Internal copilots that must distinguish between approved workflows and everything else.
              </li>
              <li>
                Safety or moderation layers that decide whether content should continue downstream.
              </li>
            </ul>
          </section>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Why this helps your resume</h2>
            <p className={styles.sectionBody}>
              This project shows more than model training. It shows product thinking, evaluation, and
              system design: turning a classifier into infrastructure that controls user experience,
              reduces bad answers, and makes an agent safer to ship. That is the kind of work teams want
              when they are building conversational products that need both quality and restraint.
            </p>
          </section>

          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Next engineering step</h2>
            <p className={styles.sectionBody}>
              The current system persists runs and statuses, but the next production upgrade would be
              moving execution into a dedicated worker with queue-backed retries and object storage for
              checkpoints. That would separate request handling from long-running model work while
              keeping the same product flow.
            </p>
          </section>
        </article>
        </div>
      </main>
    </>
  );
}
