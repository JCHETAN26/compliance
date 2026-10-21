# Dataset licenses

Licenses are checked and recorded here **before** a dataset is downloaded or used.
Raw datasets are never committed to git. They are fetched by scripts into a
git-ignored cache (`.cache/datasets/` on the host, or the `/cache/datasets` volume in
containers).

## ContractNLI (primary)

- **Dataset:** ContractNLI — A Dataset for Document-level Natural Language Inference for
  Contracts (Koreeda & Manning, Findings of EMNLP 2021). Published by Hitachi America,
  Ltd.; maintained by stanfordnlp.
- **Source:** https://stanfordnlp.github.io/contract-nli/
- **Archive:** https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip
  (65,362,913 bytes; `Last-Modified: Fri, 11 Feb 2022 23:16:51 GMT`, ETag
  `"6206ee63-3e55be1"`, observed 2026-10-21). The SHA-256 pinned by the fetch script
  is recorded in `seed/manifest.yaml`.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/ . The project page states: "Our dataset
  is released under CC BY 4.0."
- **Additional terms:** the download page carries Hitachi America, Ltd. "Terms and
  Conditions of Use". Downloading means agreeing to them. They grant use under CC BY 4.0
  and add two things: a disclaimer about third-party links contained in the dataset, and
  a DMCA (17 U.S.C. §512(c)(3)) notice procedure for copyright claims. They add no
  non-commercial or no-derivatives restriction.
- **Checked:** 2026-10-21, by reading the project page text (license section and the
  full Terms and Conditions) before the archive was downloaded.
- **How this project uses it:**
  - Phase 1 `make seed` renders a small number of **test-split** NDAs into demo
    PDF/DOCX files, including one image-only "scanned" PDF.
  - Phase 2/3 will use the official train/dev/test splits unchanged for retrieval
    evaluation and model training.
  - Rendered demo documents are generated at seed time and never committed.
- **Attribution (required by CC BY 4.0):** given in `seed/README.md`, the README and
  the generated vendor register's "Sources" sheet. Each generated document carries its
  provenance in its own metadata: the PDF Subject/Author fields or the DOCX core
  properties name the ContractNLI test document id, the original file name, the
  authors and the license. Changes made:
  - original PDFs are re-saved with that metadata, content unchanged;
  - two documents are re-typeset from the dataset text as DOCX;
  - one PDF is rasterised (with slight skew and noise) to simulate a scan.
- **Citation:**
  ```
  @inproceedings{koreeda-manning-2021-contractnli-dataset,
    title = "{C}ontract{NLI}: A Dataset for Document-level Natural Language Inference for Contracts",
    author = "Koreeda, Yuta and Manning, Christopher",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.164",
  }
  ```

## CUAD (secondary)

Not checked or downloaded yet. It is needed only from Phase 3, and its license will be
recorded here before first use.

## Model weights (not datasets, recorded for completeness)

- `BAAI/bge-small-en-v1.5` (embeddings, Phase 1): MIT license per the Hugging Face model
  card. It is downloaded at a pinned revision into the git-ignored model cache and never
  committed.
