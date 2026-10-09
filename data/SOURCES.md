# Corpus sources

`corpus.jsonl` contains the plain text of 40 articles from the Thai Wikipedia
(https://th.wikipedia.org), fetched with `scripts/fetch_corpus.py` from the
titles in `titles.txt`. Each line records the article's page id, title and the
exact revision used (`url` links to that revision).

The text is licensed under the
[Creative Commons Attribution-ShareAlike 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/)
by the Wikipedia contributors. The revision links give attribution to the
authors of each article. The evaluation questions in `eval/questions.jsonl`
are derived from this text and are shared under the same licence.
