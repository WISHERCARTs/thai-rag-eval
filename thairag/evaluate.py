"""Measure retrieval quality per retriever, and answer quality with an LLM judge."""
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from thairag.metrics import dedupe, hit_at_k, mean, reciprocal_rank
from thairag.rag import LLM, NOT_FOUND, RagPipeline
from thairag.retrievers import Retriever


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer: str
    gold_chunk_id: str
    doc_id: str


def load_questions(path: Path) -> list[Question]:
    with open(path, encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]


def save_questions(questions: Sequence[Question], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")


# ---- retrieval -----------------------------------------------------------

def evaluate_retrieval(
    retrievers: Mapping[str, Retriever],
    questions: Sequence[Question],
    chunk_doc: Mapping[str, str],
    ks: Sequence[int] = (1, 3, 5),
    depth: int = 10,
) -> dict[str, dict[str, float]]:
    """Chunk-level hit@k and MRR (the exact source passage), plus doc-level hit@k
    (any passage from the right article)."""
    results: dict[str, dict[str, float]] = {}
    for name, retriever in retrievers.items():
        scores: dict[str, list[float]] = {}
        for q in questions:
            ranked = [chunk_id for chunk_id, _ in retriever.search(q.question, depth)]
            ranked_docs = dedupe(chunk_doc[c] for c in ranked)
            for k in ks:
                scores.setdefault(f"chunk_hit@{k}", []).append(hit_at_k(ranked, {q.gold_chunk_id}, k))
                scores.setdefault(f"doc_hit@{k}", []).append(hit_at_k(ranked_docs, {q.doc_id}, k))
            scores.setdefault("chunk_mrr", []).append(reciprocal_rank(ranked, {q.gold_chunk_id}))
        results[name] = {metric: mean(values) for metric, values in scores.items()}
    return results


def retrieval_table(results: Mapping[str, Mapping[str, float]]) -> str:
    metrics = list(next(iter(results.values())).keys())
    lines = ["| retriever | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
    for name, row in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{row[m]:.3f}" for m in metrics) + " |")
    return "\n".join(lines)


# ---- answers -------------------------------------------------------------

JUDGE_SYSTEM = "คุณเป็นผู้ตรวจคำตอบที่เข้มงวดและเป็นกลาง ตอบเป็น JSON เท่านั้น"


def faithfulness_prompt(context: str, answer: str) -> str:
    return (
        "ตรวจว่าทุกข้อความในคำตอบมีข้อมูลรองรับในเอกสารหรือไม่ "
        "ถ้ามีข้อความใดที่เอกสารไม่ได้กล่าวไว้ ให้ถือว่าไม่ faithful\n"
        'ตอบเป็น JSON: {"faithful": true หรือ false, "reason": "เหตุผลสั้นๆ"}\n\n'
        f"เอกสาร:\n{context}\n\nคำตอบ:\n{answer}"
    )


def correctness_prompt(question: str, reference: str, answer: str) -> str:
    return (
        "เทียบคำตอบของระบบกับคำตอบอ้างอิง ถือว่าถูกเมื่อคำตอบของระบบมีข้อเท็จจริงหลักเดียวกับคำตอบอ้างอิง "
        "(ใช้ถ้อยคำต่างกันได้ มีรายละเอียดเพิ่มได้ถ้าไม่ขัดแย้ง)\n"
        'ตอบเป็น JSON: {"correct": true หรือ false, "reason": "เหตุผลสั้นๆ"}\n\n'
        f"คำถาม: {question}\nคำตอบอ้างอิง: {reference}\nคำตอบของระบบ: {answer}"
    )


def parse_judgement(text: str, key: str) -> bool | None:
    """Read a boolean field from a judge reply. None if the reply is unusable."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        value = json.loads(cleaned).get(key)
    except (json.JSONDecodeError, AttributeError):
        return None
    return value if isinstance(value, bool) else None


CITATION = re.compile(r"\[\d+\]")


def evaluate_answers(pipeline: RagPipeline, judge: LLM, questions: Sequence[Question]) -> tuple[dict, list[dict]]:
    rows = []
    for q in questions:
        answer = pipeline.ask(q.question)
        abstained = answer.text.strip().startswith(NOT_FOUND)
        faithful = None
        if not abstained:
            faithful = parse_judgement(
                judge.generate(faithfulness_prompt(answer.context, answer.text), JUDGE_SYSTEM, json_mode=True),
                "faithful",
            )
        correct = parse_judgement(
            judge.generate(correctness_prompt(q.question, q.answer, answer.text), JUDGE_SYSTEM, json_mode=True),
            "correct",
        )
        rows.append(
            {
                "id": q.id,
                "question": q.question,
                "reference": q.answer,
                "answer": answer.text,
                "sources": [c.id for c in answer.sources],
                "gold_in_context": q.gold_chunk_id in {c.id for c in answer.sources},
                "abstained": abstained,
                "cited": bool(CITATION.search(answer.text)),
                "faithful": faithful,
                "correct": correct,
            }
        )
    return summarize_answers(rows), rows


def summarize_answers(rows: Sequence[dict]) -> dict:
    answered = [r for r in rows if not r["abstained"]]
    faith_judged = [r for r in answered if r["faithful"] is not None]
    correct_judged = [r for r in rows if r["correct"] is not None]
    return {
        "questions": len(rows),
        "correct": mean([float(r["correct"]) for r in correct_judged]),
        "faithful": mean([float(r["faithful"]) for r in faith_judged]),
        "abstained": mean([float(r["abstained"]) for r in rows]),
        "cited_when_answering": mean([float(r["cited"]) for r in answered]),
        "gold_in_context": mean([float(r["gold_in_context"]) for r in rows]),
        "unjudged": len(rows) - len(correct_judged) + len(answered) - len(faith_judged),
    }
