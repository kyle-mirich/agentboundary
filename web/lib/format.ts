import type { Label } from "./api";

/** Alias kept so page-local types read the same as the API label union. */
export type DemoLabel = Label;

export const LABEL_ORDER: DemoLabel[] = ["in_scope", "out_of_scope", "ambiguous"];

export const LABEL_DISPLAY: Record<DemoLabel, string> = {
  in_scope: "In Scope",
  out_of_scope: "Out of Scope",
  ambiguous: "Ambiguous",
};

export function labelDisplay(label: string): string {
  return LABEL_DISPLAY[label as DemoLabel] ?? label;
}

// Hoisted rather than constructed per call: the processing view re-renders
// every second and maps over up to 120 events, so building a formatter per
// event allocated ~120 of them per second on the main thread.
const CLOCK_FORMAT = new Intl.DateTimeFormat(undefined, {
  hour: "numeric",
  minute: "2-digit",
});

const TIMESTAMP_FORMAT = new Intl.DateTimeFormat(undefined, {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false,
});

export function formatClock(iso: string): string {
  return CLOCK_FORMAT.format(new Date(iso));
}

export function formatTimestamp(iso: string): string {
  return TIMESTAMP_FORMAT.format(new Date(iso));
}

export function formatMetric(value: number | null | undefined): string {
  if (value == null) return "—";
  return value.toFixed(3);
}

export function formatPercent(value: number | null): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export function metricColor(value: number | null | undefined): string {
  if (value == null) return "";
  if (value >= 0.85) return "good";
  if (value >= 0.7) return "warn";
  return "bad";
}

export function truncate(text: string, max: number): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max).trimEnd()}…`;
}
