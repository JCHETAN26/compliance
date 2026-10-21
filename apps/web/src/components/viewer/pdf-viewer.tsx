"use client";

import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

import { AlertTriangle, Loader2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { BBox, PageInfo } from "@/lib/api/client";

import { highlightText, toPixels } from "./highlight";

// Served from /public (copied from the installed pdfjs-dist by scripts/copy-pdf-worker.mjs,
// so the worker always matches the library version). Must be set in this module.
pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";

const MAX_WIDTH = 900;

export interface PdfViewerProps {
  fileUrl: string;
  pages: PageInfo[];
  highlight: BBox[];
  /** Incremented whenever the viewer should scroll to the highlight (e.g. chunk click). */
  focusToken: number;
  searchQuery: string;
  /** Page to scroll to for search navigation (1-based), with its own token. */
  scrollToPage: { page: number; token: number } | null;
}

export function PdfViewer({ fileUrl, pages, highlight, focusToken, searchQuery, scrollToPage }: PdfViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(MAX_WIDTH);
  const [numPages, setNumPages] = useState<number | null>(null);
  const [visible, setVisible] = useState<Set<number>>(() => new Set([1, 2]));
  const pageInfo = useMemo(() => new Map(pages.map((p) => [p.page_number, p])), [pages]);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(Math.min(MAX_WIDTH, Math.floor(entry.contentRect.width)));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  // Render only pages near the viewport (large documents stay responsive).
  useEffect(() => {
    if (!numPages) return;
    const observer = new IntersectionObserver(
      (entries) => {
        setVisible((current) => {
          const next = new Set(current);
          for (const entry of entries) {
            const page = Number((entry.target as HTMLElement).dataset.page);
            if (entry.isIntersecting) next.add(page);
          }
          return next.size === current.size ? current : next;
        });
      },
      { root: containerRef.current, rootMargin: "800px 0px" },
    );
    const elements = containerRef.current?.querySelectorAll<HTMLElement>("[data-page]") ?? [];
    for (const element of elements) observer.observe(element);
    return () => observer.disconnect();
  }, [numPages]);

  const scrollTo = useCallback(
    (page: number, yPoints: number) => {
      const container = containerRef.current;
      const element = container?.querySelector<HTMLElement>(`[data-page="${page}"]`);
      const info = pageInfo.get(page);
      if (!container || !element) return false;
      const scale = info ? width / info.width : 1;
      setVisible((current) => (current.has(page) ? current : new Set([...current, page])));
      // Offset of the page inside the scroll container (independent of offsetParent).
      const pageTop =
        element.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop;
      container.scrollTo({ top: Math.max(pageTop + yPoints * scale - 96, 0), behavior: "smooth" });
      return true;
    },
    [pageInfo, width],
  );

  // Focus waits until the PDF has loaded (numPages) so a fresh ?chunk= deep link to a
  // later page still scrolls once page placeholders with real geometry exist.
  useEffect(() => {
    const first = highlight[0];
    if (numPages && focusToken > 0 && first) scrollTo(first.page, first.y0);
  }, [numPages, focusToken, highlight, scrollTo]);

  useEffect(() => {
    if (numPages && scrollToPage) scrollTo(scrollToPage.page, 0);
  }, [numPages, scrollToPage, scrollTo]);

  const textRenderer = useCallback(
    ({ str }: { str: string }) => highlightText(str, searchQuery),
    [searchQuery],
  );

  return (
    <div ref={containerRef} className="h-full overflow-auto bg-muted/40 p-4" data-testid="pdf-scroll">
      <Document
        file={fileUrl}
        onLoadSuccess={(pdf) => setNumPages(pdf.numPages)}
        loading={
          <div className="flex items-center gap-2 p-6 text-sm text-muted-foreground" role="status">
            <Loader2 className="size-4 animate-spin" aria-hidden />
            Loading PDF…
          </div>
        }
        error={
          <Alert variant="destructive">
            <AlertTriangle aria-hidden />
            <AlertTitle>The PDF could not be displayed</AlertTitle>
            <AlertDescription>The original file failed to load in the viewer.</AlertDescription>
          </Alert>
        }
      >
        {numPages
          ? Array.from({ length: numPages }, (_, i) => i + 1).map((pageNumber) => {
              const info = pageInfo.get(pageNumber);
              const height = info ? (info.height * width) / info.width : width * 1.294;
              const rects = info ? highlight.filter((b) => b.page === pageNumber) : [];
              return (
                <div
                  key={pageNumber}
                  data-page={pageNumber}
                  className="relative mx-auto mb-4 bg-white shadow-sm"
                  style={{ width, minHeight: height }}
                  aria-label={`Page ${pageNumber}${info?.is_ocr ? " (scanned, OCR text)" : ""}`}
                  role="region"
                >
                  {visible.has(pageNumber) ? (
                    <Page
                      pageNumber={pageNumber}
                      width={width}
                      renderAnnotationLayer={false}
                      customTextRenderer={searchQuery.trim() ? textRenderer : undefined}
                    />
                  ) : null}
                  {info
                    ? rects.map((box, index) => {
                        const px = toPixels(box, info.width, width);
                        return (
                          <div
                            key={`${box.x0}-${box.y0}-${index}`}
                            data-testid="chunk-highlight"
                            className="pointer-events-none absolute z-10 rounded-sm bg-amber-300/40 ring-1 ring-amber-500"
                            style={{ left: px.left, top: px.top, width: px.width, height: px.height }}
                          />
                        );
                      })
                    : null}
                </div>
              );
            })
          : null}
      </Document>
    </div>
  );
}
