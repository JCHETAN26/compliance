import { chunksMatching, countMatches, escapeHtml, highlightText, segmentBlock, toPixels } from "./highlight";

describe("highlightText", () => {
  it("escapes document text so a PDF cannot inject markup into the text layer", () => {
    const html = highlightText(`<img src=x onerror="alert(1)"> & 'quotes'`, "");
    expect(html).toBe("&lt;img src=x onerror=&quot;alert(1)&quot;&gt; &amp; &#39;quotes&#39;");
    expect(html).not.toContain("<img");
  });

  it("wraps case-insensitive matches in mark tags and still escapes around them", () => {
    expect(highlightText("Return <b>ALL</b> copies, return them", "return")).toBe(
      '<mark class="search-hit">Return</mark> &lt;b&gt;ALL&lt;/b&gt; copies, <mark class="search-hit">return</mark> them',
    );
  });

  it("treats regex metacharacters in the query literally", () => {
    expect(highlightText("Section 1.2 (a)", "1.2 (a)")).toBe('Section <mark class="search-hit">1.2 (a)</mark>');
    expect(highlightText("abc", ".*")).toBe("abc");
  });

  it("escapes the matched text itself", () => {
    expect(highlightText("a<b", "<")).toBe('a<mark class="search-hit">&lt;</mark>b');
    expect(escapeHtml("&")).toBe("&amp;");
  });
});

describe("geometry and ranges", () => {
  it("scales PDF points to rendered pixels", () => {
    const px = toPixels({ page: 1, x0: 72, y0: 100, x1: 144, y1: 112 }, 612, 918);
    expect(px.left).toBeCloseTo(108);
    expect(px.top).toBeCloseTo(150);
    expect(px.width).toBeCloseTo(108);
    expect(px.height).toBeCloseTo(18);
  });

  it("segments a block exactly at document offsets", () => {
    // block "Hello brave new world" starts at document offset 100
    expect(segmentBlock("Hello brave new world", 100, 106, 115)).toEqual([
      { text: "Hello ", highlighted: false },
      { text: "brave new", highlighted: true },
      { text: " world", highlighted: false },
    ]);
    expect(segmentBlock("abc", 100, 50, 60)).toEqual([{ text: "abc", highlighted: false }]);
    expect(segmentBlock("abc", 100, 50, 200)).toEqual([{ text: "abc", highlighted: true }]);
    expect(segmentBlock("abc", 100, null, null)).toEqual([{ text: "abc", highlighted: false }]);
  });

  it("counts matches and filters chunks", () => {
    expect(countMatches("Confidential and CONFIDENTIAL", "confidential")).toBe(2);
    expect(countMatches("x", " ")).toBe(0);
    const chunk = (id: string, text: string) => ({
      id,
      text,
      chunk_index: 0,
      char_start: 0,
      char_end: text.length,
      page_start: 1,
      page_end: 1,
      bboxes: [],
      section_path: null,
      token_count: 1,
      embedding_windows: 1,
      table_row_id: null,
    });
    const chunks = [chunk("a", "Return copies"), chunk("b", "Governing law")];
    expect(chunksMatching(chunks, "LAW").map((c) => c.id)).toEqual(["b"]);
    expect(chunksMatching(chunks, "")).toHaveLength(2);
  });
});
