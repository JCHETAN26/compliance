// Copy the pdf.js worker that matches the installed pdfjs-dist into /public so the
// viewer can load it same-origin (no CDN, no version skew).
import { copyFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const reactPdfDir = dirname(require.resolve("react-pdf/package.json"));
const pdfjsPackage = createRequire(join(reactPdfDir, "package.json")).resolve("pdfjs-dist/package.json");
// Legacy build: matches the aliased main-thread build (see next.config.ts).
const source = join(dirname(pdfjsPackage), "legacy", "build", "pdf.worker.min.mjs");
const publicDir = join(dirname(fileURLToPath(import.meta.url)), "..", "public");
mkdirSync(publicDir, { recursive: true });
copyFileSync(source, join(publicDir, "pdf.worker.min.mjs"));
console.log(`copied ${source} -> public/pdf.worker.min.mjs`);
