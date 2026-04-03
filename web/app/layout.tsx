import "./globals.css";
import "./demo-v2.css";
import { Syne, DM_Sans, JetBrains_Mono } from "next/font/google";
import type { Metadata, Viewport } from "next";

const syne = Syne({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["400", "500", "600", "700", "800"],
  display: "swap",
});

const dmSans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  weight: ["400", "500", "700"],
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: new URL("https://example.com"),
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
    <html lang="en" className={`${syne.variable} ${dmSans.variable} ${jetbrainsMono.variable}`}>
      <body>
        {children}
      </body>
    </html>
  );
}
