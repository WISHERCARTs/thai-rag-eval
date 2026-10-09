"""Download plain-text Thai Wikipedia articles listed in data/titles.txt.

Writes data/corpus.jsonl with one article per line, pinned to the revision
that was fetched so the evaluation set stays reproducible.

Usage: python scripts/fetch_corpus.py [--titles data/titles.txt] [--out data/corpus.jsonl]
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://th.wikipedia.org/w/api.php"
USER_AGENT = "thai-rag-eval/0.1 (https://github.com/WISHERCARTs)"


def fetch(title: str) -> dict | None:
    params = {
        "action": "query",
        "prop": "extracts|revisions",
        "rvprop": "ids",
        "explaintext": "1",
        "redirects": "1",
        "format": "json",
        "formatversion": "2",
        "titles": title,
    }
    request = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        page = json.load(response)["query"]["pages"][0]
    if page.get("missing") or not page.get("extract"):
        return None
    return {
        "doc_id": str(page["pageid"]),
        "title": page["title"],
        "revision": page["revisions"][0]["revid"],
        "url": f"https://th.wikipedia.org/w/index.php?oldid={page['revisions'][0]['revid']}",
        "text": page["extract"],
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--titles", type=Path, default=Path("data/titles.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/corpus.jsonl"))
    args = parser.parse_args()

    titles = [t.strip() for t in args.titles.read_text(encoding="utf-8").splitlines() if t.strip() and not t.startswith("#")]
    docs, missing, seen = [], [], set()
    for title in titles:
        doc = fetch(title)
        time.sleep(0.5)  # be polite to the Wikipedia API
        if doc is None:
            missing.append(title)
            continue
        if doc["doc_id"] in seen:  # two titles redirecting to one article
            continue
        seen.add(doc["doc_id"])
        docs.append(doc)
        print(f"ok  {doc['title']}  ({len(doc['text']):,} chars)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"\n{len(docs)} articles written to {args.out}")
    if missing:
        print("Missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
