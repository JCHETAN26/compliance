import type { SearchHit, SearchMode, SearchRequest } from "@/lib/api/client";

const MODES: SearchMode[] = ["hybrid", "vector", "keyword"];
const DOC_TYPES = ["contract", "policy", "spreadsheet", "other"] as const;
export const DEFAULT_TOP_K = 8;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

type Params = { get(name: string): string | null };

/** Read the search request from the URL (shareable, survives back navigation). Invalid
 * values fall back to defaults instead of being sent to the API. */
export function parseSearchParams(params: Params): SearchRequest {
  const mode = params.get("mode");
  const k = Number(params.get("k"));
  const doc = params.get("doc");
  const type = params.get("type");
  return {
    query: (params.get("q") ?? "").trim().slice(0, 1000),
    mode: MODES.includes(mode as SearchMode) ? (mode as SearchMode) : "hybrid",
    top_k: Number.isInteger(k) && k >= 1 && k <= 50 ? k : DEFAULT_TOP_K,
    document_ids: doc && UUID.test(doc) ? [doc] : [],
    doc_types: DOC_TYPES.includes(type as (typeof DOC_TYPES)[number]) ? [type as (typeof DOC_TYPES)[number]] : [],
    rerank: params.get("rerank") === "1",
  };
}

export function toSearchParams(request: SearchRequest): URLSearchParams {
  const params = new URLSearchParams();
  if (request.query) params.set("q", request.query);
  if (request.mode !== "hybrid") params.set("mode", request.mode);
  if (request.top_k !== DEFAULT_TOP_K) params.set("k", String(request.top_k));
  if (request.document_ids?.[0]) params.set("doc", request.document_ids[0]);
  if (request.doc_types?.[0]) params.set("type", request.doc_types[0]);
  if (request.rerank) params.set("rerank", "1");
  return params;
}

/** Human-readable source location: spreadsheet row, PDF page(s) or section. */
export function locationLabel(hit: SearchHit): string {
  if (hit.sheet_name !== null && hit.row_number !== null) {
    return `Sheet “${hit.sheet_name}”, row ${hit.row_number}`;
  }
  if (hit.page_start !== null) {
    return hit.page_end !== null && hit.page_end !== hit.page_start
      ? `pages ${hit.page_start}–${hit.page_end}`
      : `page ${hit.page_start}`;
  }
  return hit.section_path ?? `characters ${hit.char_start}–${hit.char_end}`;
}

const STOP_WORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "has", "have",
  "how", "in", "is", "it", "its", "may", "of", "on", "or", "that", "the", "their", "this", "to", "was",
  "what", "when", "where", "which", "who", "will", "with",
]);

/** Distinct, lower-cased query words worth marking in results (display only). */
export function queryTerms(query: string): string[] {
  const words = query.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
  return [...new Set(words.filter((w) => w.length >= 3 && !STOP_WORDS.has(w)))];
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export interface TextPart {
  text: string;
  match: boolean;
}

/** Split stored text into parts, marking word-prefix matches of ``terms`` (case-insensitive).
 * Concatenating the parts always reproduces ``text`` exactly. */
export function splitByTerms(text: string, terms: string[]): TextPart[] {
  if (terms.length === 0 || text.length === 0) return [{ text, match: false }];
  const pattern = new RegExp(
    `(?<![\\p{L}\\p{N}])(?:${[...terms].sort((a, b) => b.length - a.length).map(escapeRegExp).join("|")})[\\p{L}\\p{N}]*`,
    "giu",
  );
  const parts: TextPart[] = [];
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    const index = match.index ?? 0;
    if (index > last) parts.push({ text: text.slice(last, index), match: false });
    parts.push({ text: match[0], match: true });
    last = index + match[0].length;
  }
  if (last < text.length) parts.push({ text: text.slice(last), match: false });
  return parts.length ? parts : [{ text, match: false }];
}
