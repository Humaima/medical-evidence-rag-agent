"""
Convert a real PubMed export into the corpus JSON schema used by ingest.py.

Supports three common export formats:

1. "Abstract (text)" .txt export from the PubMed website
   (Search -> Save -> Format: "Abstract (text)" -> Create File).
   This is the recommended route if you exported from pubmed.ncbi.nlm.nih.gov
   directly, since PubMed's CSV export does NOT include abstract text --
   only the "Abstract (text)" and XML routes do.

2. CSV export with at least these columns: PMID, Title, Abstract
   (e.g. from a citation manager or a source that already flattened
   PubMed data into CSV with abstracts included).
   optional columns: Journal, URL

3. Native PubMed XML (PubmedArticleSet), from NCBI E-utilities efetch,
   or older PubMed website XML export.

Usage:
    # Abstract (text) -- most common if exported from the PubMed website
    python src/convert_pubmed.py --input my_pubmed_export.txt --format text --out data/my_corpus.json

    # CSV
    python src/convert_pubmed.py --input my_pubmed_export.csv --format csv --out data/my_corpus.json

    # XML
    python src/convert_pubmed.py --input my_pubmed_export.xml --format xml --out data/my_corpus.json

Then point ingest.py at the resulting file:
    python src/ingest.py --input data/my_corpus.json --out index/faiss_medical
"""

from __future__ import annotations
import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_text(path: str) -> list[dict]:
    """
    Parse PubMed's "Abstract (text)" export format.

    Each record looks roughly like (paragraphs separated by blank lines;
    individual paragraphs may themselves wrap across several physical
    lines -- PubMed hard-wraps at ~80 chars, which is NOT a paragraph break):

        1. Journal Abbrev. 2023 Jan 1;12(3):456-789. doi: 10.xxxx/xxxx.

        Title of the article goes here, possibly wrapped
        across multiple lines.

        Author One, Author Two, Author Three.

        Author information:
        (1)Affiliation one. (2)Affiliation two.

        Full abstract text, possibly with ALL-CAPS section
        headers (INTRODUCTION:, METHODS:, ...) and possibly
        spanning more than one paragraph.

        © 2023. The Author(s).

        DOI: 10.xxxx/xxxx
        PMID: 12345678 [Indexed for MEDLINE]

        Conflict of interest statement: ...

    Newer exports (2024+) drop the literal "Abstract" header line that used
    to precede the abstract body, so this parser locates the abstract by
    position (after the title/author/"Author information" paragraphs, before
    the copyright/DOI/PMID metadata paragraph) rather than by that header.

    This parser is heuristic (PubMed doesn't publish a formal spec for this
    export), so spot-check a few converted entries in the output JSON after
    running this -- if PubMed tweaks the export layout, only this function
    needs updating.
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()

    # Records are numbered "1. ", "2. ", etc. at the start of a line.
    blocks = re.split(r"\n(?=\d+\.\s)", raw.strip())
    docs = []

    metadata_prefixes = (
        "DOI:", "PMID:", "PMCID:", "©", "Comment in", "Comment on",
        "Erratum", "Retraction", "Update", "Update of",
    )

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        pmid_match = re.search(r"PMID:\s*(\d+)", block)
        pmid = pmid_match.group(1) if pmid_match else "N/A"

        # Split into logical paragraphs (blank-line separated), normalizing
        # each paragraph's internal line-wrapping to single spaces.
        paragraphs = [
            " ".join(p.split())
            for p in re.split(r"\n\s*\n", block)
            if p.strip()
        ]

        title = paragraphs[1].rstrip(".") if len(paragraphs) > 1 else "Untitled"

        # Abstract starts right after the "Author information:" paragraph if
        # present, else right after citation/title/author-list (index 3).
        start_idx = 3
        for i, p in enumerate(paragraphs):
            if p.startswith("Author information:"):
                start_idx = i + 1
                break

        # Abstract ends at the first metadata/footer paragraph.
        end_idx = len(paragraphs)
        for i in range(start_idx, len(paragraphs)):
            if paragraphs[i].startswith(metadata_prefixes):
                end_idx = i
                break

        abstract = " ".join(paragraphs[start_idx:end_idx]).strip()

        if not abstract:
            # Skip records with no parseable abstract rather than polluting
            # the corpus with empty entries.
            continue

        docs.append(
            {
                "id": f"pmid_{pmid}" if pmid != "N/A" else f"doc_{len(docs):04d}",
                "pmid": pmid,
                "title": title,
                "source": "PubMed",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid != "N/A" else "N/A",
                "text": abstract,
            }
        )

    return docs


def convert_csv(path: str) -> list[dict]:
    docs = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            pmid = row.get("PMID", "").strip()
            title = row.get("Title", "").strip()
            abstract = row.get("Abstract", "").strip()
            if not title or not abstract:
                continue
            docs.append(
                {
                    "id": f"pmid_{pmid}" if pmid else f"doc_{i:04d}",
                    "pmid": pmid or "N/A",
                    "title": title,
                    "source": row.get("Journal", "PubMed"),
                    "url": row.get("URL") or (
                        f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "N/A"
                    ),
                    "text": abstract,
                }
            )
    return docs


def convert_xml(path: str) -> list[dict]:
    tree = ET.parse(path)
    root = tree.getroot()
    docs = []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else "N/A"

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else "Untitled"

        abstract_parts = article.findall(".//Abstract/AbstractText")
        abstract = " ".join(
            "".join(part.itertext()).strip() for part in abstract_parts if part is not None
        ).strip()

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text.strip() if journal_el is not None and journal_el.text else "PubMed"

        if not title or not abstract:
            continue

        docs.append(
            {
                "id": f"pmid_{pmid}",
                "pmid": pmid,
                "title": title,
                "source": journal,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid != "N/A" else "N/A",
                "text": abstract,
            }
        )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a PubMed export to corpus JSON.")
    parser.add_argument("--input", required=True, help="Path to the PubMed export file.")
    parser.add_argument("--format", choices=["text", "csv", "xml"], required=True)
    parser.add_argument("--out", required=True, help="Output path for corpus JSON.")
    args = parser.parse_args()

    if args.format == "text":
        docs = convert_text(args.input)
    elif args.format == "csv":
        docs = convert_csv(args.input)
    else:
        docs = convert_xml(args.input)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(docs)} documents -> {args.out}")


if __name__ == "__main__":
    main()
