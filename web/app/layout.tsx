import "./globals.css";
import "./demo-v2.css";
import type { Metadata, Viewport } from "next";
import WakeupGate from "./wakeup";

export const metadata: Metadata = {
  title: "Agent Boundary",
  description:
    "Portfolio project for designing, training, and evaluating the Agent Boundary scope classifier with Deep Agents, FastAPI, and Next.js.",
  applicationName: "Agent Boundary",
  keywords: [
    "portfolio project",
    "customer support classifier",
    "Deep Agents",
    "FastAPI",
    "Next.js",
    "PyTorch",
    "LLM evaluation",
  ],
  openGraph: {
    title: "Agent Boundary",
    description:
      "A polished portfolio demo for building and reviewing Agent Boundary end to end.",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Agent Boundary",
    description:
      "A polished portfolio demo for building and reviewing Agent Boundary end to end.",
  },
};

export const viewport: Viewport = {
  themeColor: "#ece7dc",
  colorScheme: "light",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      style={
        {
          "--font-display": '"Avenir Next", "Segoe UI", sans-serif',
          "--font-body": '"Avenir Next", "Segoe UI", sans-serif',
          "--font-mono": '"SFMono-Regular", "SF Mono", "JetBrains Mono", monospace',
        } as React.CSSProperties
      }
    >
      <body>
        <WakeupGate>{children}</WakeupGate>
      </body>
    </html>
  );
}
