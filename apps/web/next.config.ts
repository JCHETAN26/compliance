import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Self-contained server bundle for the Docker image.
  output: "standalone",
  poweredByHeader: false,
  reactStrictMode: true,
  typedRoutes: true,
  // Do not write AGENTS.md into the app directory during `next dev`.
  agentRules: false,
  turbopack: {
    resolveAlias: {
      // pdf.js 6's modern build needs ES2025 built-ins (Uint8Array#toHex,
      // Map#getOrInsertComputed, Math.sumPrecise) missing from current stable browsers
      // such as Chrome 139. The legacy build bundles polyfills for them.
      "pdfjs-dist": "pdfjs-dist/legacy/build/pdf.mjs",
    },
  },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        ],
      },
    ];
  },
};

export default nextConfig;
