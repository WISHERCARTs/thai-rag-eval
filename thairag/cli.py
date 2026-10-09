"""Command-line entry point: python -m thairag <command>."""
import argparse
import json
import logging
import sys
from pathlib import Path

from thairag.chunking import Chunk
from thairag.embeddings import QueryCache
from thairag.retrievers import BM25Retriever, HybridRetriever, Retriever, VectorRetriever

DEFAULT_INDEX = Path("index")
DEFAULT_QUESTIONS = Path("eval/questions.jsonl")


def build_retrievers(chunks: list[Chunk], matrix, embedder) -> dict[str, Retriever]:
    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever([c.id for c in chunks], matrix, embedder)
    return {"bm25": bm25, "vector": vector, "hybrid": HybridRetriever([bm25, vector])}


def _gemini():
    from thairag.gemini import GeminiEmbedder, GeminiLLM, make_client

    client = make_client()
    return GeminiEmbedder(client), GeminiLLM(client)


def cmd_ingest(args) -> None:
    from thairag.index import build_index, chunk_corpus, load_corpus

    embedder, _ = _gemini()
    docs = load_corpus(args.corpus)
    chunks = chunk_corpus(docs, max_chars=args.max_chars)
    print(f"{len(docs)} documents -> {len(chunks)} chunks; embedding with {embedder.model}...")
    build_index(chunks, embedder, args.index, meta={"embed_model": embedder.model, "max_chars": args.max_chars})
    print(f"Index written to {args.index}/")


def _load(args):
    from thairag.index import load_index

    embedder, llm = _gemini()
    chunks, matrix, meta = load_index(args.index)
    cached = QueryCache(embedder)
    return chunks, build_retrievers(chunks, matrix, cached), cached, llm


def cmd_ask(args) -> None:
    from thairag.rag import RagPipeline

    chunks, retrievers, _, llm = _load(args)
    answer = RagPipeline(retrievers[args.retriever], chunks, llm, k=args.k).ask(args.question)
    print(answer.text)
    print("\nแหล่งที่มา:")
    for i, chunk in enumerate(answer.sources, start=1):
        print(f"[{i}] {chunk.title} — {chunk.section or '(บทนำ)'}  ({chunk.id})")


def cmd_make_questions(args) -> None:
    from thairag.evaluate import save_questions
    from thairag.index import load_index
    from thairag.questions import make_questions

    _, llm = _gemini()
    chunks, _, _ = load_index(args.index)
    questions = make_questions(chunks, llm, n=args.n, seed=args.seed)
    save_questions(questions, args.questions)
    print(f"Wrote {len(questions)} questions to {args.questions}")


def cmd_eval_retrieval(args) -> None:
    from thairag.evaluate import evaluate_retrieval, load_questions, retrieval_table

    chunks, retrievers, cached, _ = _load(args)
    questions = load_questions(args.questions)
    cached.prefill([q.question for q in questions])
    results = evaluate_retrieval(retrievers, questions, {c.id: c.doc_id for c in chunks})
    table = retrieval_table(results)
    print(f"{len(questions)} questions, {len(chunks)} chunks\n\n{table}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "retrieval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.out / "retrieval.md").write_text(table + "\n", encoding="utf-8")


def cmd_eval_answers(args) -> None:
    from thairag.evaluate import evaluate_answers, load_questions
    from thairag.rag import RagPipeline

    chunks, retrievers, cached, llm = _load(args)
    questions = load_questions(args.questions)[: args.limit]
    cached.prefill([q.question for q in questions])
    pipeline = RagPipeline(retrievers[args.retriever], chunks, llm, k=args.k)
    summary, rows = evaluate_answers(pipeline, llm, questions)
    print(json.dumps(summary, indent=2))
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"answers_{args.retriever}"
    (args.out / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(args.out / f"{name}_rows.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_eval_abstention(args) -> None:
    from thairag.evaluate import evaluate_abstention
    from thairag.rag import RagPipeline

    chunks, retrievers, cached, llm = _load(args)
    questions = [q.strip() for q in args.file.read_text(encoding="utf-8").splitlines() if q.strip() and not q.startswith("#")]
    cached.prefill(questions)
    pipeline = RagPipeline(retrievers[args.retriever], chunks, llm, k=args.k)
    summary, rows = evaluate_abstention(pipeline, questions)
    print(json.dumps(summary, indent=2))
    for row in rows:
        if not row["abstained"]:
            print(f"ANSWERED: {row['question']} -> {row['answer'][:200]}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"abstention_{args.retriever}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main(argv=None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(prog="thairag", description="Thai RAG over Wikipedia with evaluation")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="chunk the corpus and build the embedding index")
    p.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"))
    p.add_argument("--max-chars", type=int, default=700)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="answer one question")
    p.add_argument("question")
    p.add_argument("--retriever", choices=["bm25", "vector", "hybrid"], default="hybrid")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("make-questions", help="generate the evaluation question set")
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("-n", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_make_questions)

    p = sub.add_parser("eval-retrieval", help="compare bm25, vector and hybrid retrieval")
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--out", type=Path, default=Path("eval/results"))
    p.set_defaults(func=cmd_eval_retrieval)

    p = sub.add_parser("eval-answers", help="judge generated answers for correctness and faithfulness")
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--retriever", choices=["bm25", "vector", "hybrid"], default="hybrid")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--out", type=Path, default=Path("eval/results"))
    p.set_defaults(func=cmd_eval_answers)

    p = sub.add_parser("eval-abstention", help="check the system refuses questions the corpus cannot answer")
    p.add_argument("--file", type=Path, default=Path("eval/unanswerable.txt"))
    p.add_argument("--retriever", choices=["bm25", "vector", "hybrid"], default="hybrid")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("eval/results"))
    p.set_defaults(func=cmd_eval_abstention)

    args = parser.parse_args(argv)
    args.func(args)
