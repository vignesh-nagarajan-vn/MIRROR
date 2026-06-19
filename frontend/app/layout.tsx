import type { Metadata } from "next";
import "../styles/globals.css";

export const metadata: Metadata = {
  title: "MIRROR — Radiology Reasoning & Observation Reporter",
  description:
    "Explainable multimodal radiology analysis: predictions, saliency evidence, and draft reports. Research prototype.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        {/* Reading-room type: a technical mono for data, a humanist sans for prose. */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
