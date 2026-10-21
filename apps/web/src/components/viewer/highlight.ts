import type { BBox, ChunkInfo } from "@/lib/api/client";

const HTML_ESCAPES: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

/** Escape untrusted document text before it is inserted as HTML (pdf.js text layer). */
export function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (char) => HTML_ESCAPES[char] ?? char);
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * HTML for one pdf.js text item with case-insensitive matches of ``query`` wrapped in
 * <mark>. All document text is escaped; only our <mark> tags are markup.
 */
export function highlightText(text: string, query: string): string {
  const needle = query.trim();
  if (!needle) return escapeHtml(text);
  const pattern = new RegExp(escapeRegExp(needle), "gi");
  let html = "";
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    const index = match.index ?? 0;
    html += escapeHtml(text.slice(last, index));
    html += `<mark class="search-hit">${escapeHtml(match[0])}</mark>`;
    last = index + match[0].length;
  }
  return html + escapeHtml(text.slice(last));
}

/** Count case-insensitive occurrences of ``query`` in ``text``. */
export function countMatches(text: string, query: string): number {
  const needle = query.trim().toLowerCase();
  if (!needle) return 0;
  let count = 0;
  let from = text.toLowerCase().indexOf(needle);
  const haystack = text.toLowerCase();
  while (from !== -1) {
    count += 1;
    from = haystack.indexOf(needle, from + needle.length);
  }
  return count;
}

export interface PixelRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** Convert a PDF-point box into CSS pixels for a page rendered ``renderedWidth`` px wide. */
export function toPixels(box: BBox, pageWidthPts: number, renderedWidth: number): PixelRect {
  const scale = renderedWidth / pageWidthPts;
  return {
    left: box.x0 * scale,
    top: box.y0 * scale,
    width: Math.max((box.x1 - box.x0) * scale, 1),
    height: Math.max((box.y1 - box.y0) * scale, 1),
  };
}

export interface Segment {
  text: string;
  highlighted: boolean;
}

/**
 * Split the block ``[blockStart, blockStart + blockText.length)`` into plain and
 * highlighted segments for the document range ``[rangeStart, rangeEnd)``.
 */
export function segmentBlock(
  blockText: string,
  blockStart: number,
  rangeStart: number | null,
  rangeEnd: number | null,
): Segment[] {
  if (rangeStart === null || rangeEnd === null) return [{ text: blockText, highlighted: false }];
  const from = Math.max(rangeStart - blockStart, 0);
  const to = Math.min(rangeEnd - blockStart, blockText.length);
  if (from >= to) return [{ text: blockText, highlighted: false }];
  return [
    { text: blockText.slice(0, from), highlighted: false },
    { text: blockText.slice(from, to), highlighted: true },
    { text: blockText.slice(to), highlighted: false },
  ].filter((segment) => segment.text.length > 0);
}

/** Chunks whose text contains ``query`` (case-insensitive), in document order. */
export function chunksMatching(chunks: ChunkInfo[], query: string): ChunkInfo[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return chunks;
  return chunks.filter((chunk) => chunk.text.toLowerCase().includes(needle));
}
