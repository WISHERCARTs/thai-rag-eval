import json

import pytest

from tests.fakes import FakeLLM
from thairag.chunking import Chunk
from thairag.evaluate import Question, evaluate_answers, evaluate_retrieval, parse_judgement, summarize_answers
from thairag.metrics import dedupe, hit_at_k, reciprocal_rank
from thairag.questions import make_questions, sample_chunks
from thairag.rag import NOT_FOUND, RagPipeline


class StaticRetriever:
    def __init__(self, ids):
        self.ids = ids

    def search(self, query, k):
        return [(i, 1.0) for i in self.ids[:k]]


# ---- metrics --------------------------------------------------------------

def test_hit_at_k_and_reciprocal_rank():
    ranked = ["a", "b", "c"]
    assert hit_at_k(ranked, {"b"}, 1) == 0.0
    assert hit_at_k(ranked, {"b"}, 2) == 1.0
    assert reciprocal_rank(ranked, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"z"}) == 0.0


def test_dedupe_keeps_first_occurrence_order():
    assert dedupe(["d1", "d2", "d1", "d3"]) == ["d1", "d2", "d3"]


def test_evaluate_retrieval_counts_chunk_and_doc_hits():
    questions = [
        Question("q1", "?", "!", "d1#0", "d1"),
        Question("q2", "?", "!", "d2#0", "d2"),
    ]
    chunk_doc = {"d1#0": "d1", "d1#1": "d1", "d2#0": "d2", "d2#1": "d2"}
    retriever = StaticRetriever(["d1#1", "d1#0", "d2#1"])

    result = evaluate_retrieval({"static": retriever}, questions, chunk_doc, ks=(1, 3))["static"]

    assert result["chunk_hit@1"] == 0.0
    assert result["chunk_hit@3"] == 0.5  # q1's gold is at rank 2; q2's gold never appears
    assert result["chunk_mrr"] == pytest.approx((1 / 2 + 0) / 2)
    assert result["doc_hit@1"] == 0.5  # q1's article is first; q2's article is second
    assert result["doc_hit@3"] == 1.0


# ---- generation -----------------------------------------------------------

def test_pipeline_numbers_sources_and_uses_the_system_prompt(chunks):
    llm = FakeLLM("น่านมีวัดภูมินทร์ [1]")
    pipeline = RagPipeline(StaticRetriever(["nan#0", "bkk#0"]), chunks, llm, k=2)

    answer = pipeline.ask("วัดดังของน่านคืออะไร")

    prompt = llm.calls[0]["prompt"]
    assert "[1] จังหวัดน่าน — ประวัติศาสตร์" in prompt
    assert "[2] กรุงเทพมหานคร — การคมนาคม" in prompt
    assert prompt.endswith("คำถาม: วัดดังของน่านคืออะไร")
    assert "ห้ามใช้ความรู้ภายนอก" in llm.calls[0]["system"]
    assert [c.id for c in answer.sources] == ["nan#0", "bkk#0"]


def test_pipeline_abstains_without_calling_the_llm_when_nothing_is_found(chunks):
    llm = FakeLLM("should not be used")
    answer = RagPipeline(StaticRetriever([]), chunks, llm).ask("อะไรก็ได้")

    assert answer.text == NOT_FOUND
    assert llm.calls == []


# ---- judging --------------------------------------------------------------

@pytest.mark.parametrize(
    "reply, expected",
    [
        ('{"faithful": true, "reason": "ok"}', True),
        ('```json\n{"faithful": false}\n```', False),
        ('{"faithful": "yes"}', None),
        ("not json", None),
        ("[]", None),
    ],
)
def test_parse_judgement(reply, expected):
    assert parse_judgement(reply, "faithful") is expected


def judge_reply(prompt: str) -> str:
    if "faithful" in prompt:
        return json.dumps({"faithful": "ภูเก็ต" not in prompt.split("คำตอบ:")[-1]})
    return json.dumps({"correct": "ภูมินทร์" in prompt.split("คำตอบของระบบ:")[-1]})


def test_evaluate_answers_scores_faithfulness_correctness_and_abstention(chunks):
    questions = [
        Question("q1", "วัดดังของน่าน", "วัดภูมินทร์", "nan#0", "nan"),
        Question("q2", "วัดดังของน่าน (ตอบผิด)", "วัดภูมินทร์", "nan#0", "nan"),
        Question("q3", "คำถามที่ไม่มีคำตอบ", "-", "bkk#0", "bkk"),
    ]
    answers = {
        "วัดดังของน่าน": "วัดภูมินทร์ [1]",
        "วัดดังของน่าน (ตอบผิด)": "หาดป่าตองในภูเก็ต",
        "คำถามที่ไม่มีคำตอบ": NOT_FOUND,
    }
    rag_llm = FakeLLM(lambda prompt: answers[prompt.split("คำถาม: ")[-1]])
    pipeline = RagPipeline(StaticRetriever(["nan#0"]), chunks, rag_llm, k=1)

    summary, rows = evaluate_answers(pipeline, FakeLLM(judge_reply), questions)

    assert [r["faithful"] for r in rows] == [True, False, None]
    assert [r["correct"] for r in rows] == [True, False, False]
    assert [r["cited"] for r in rows] == [True, False, False]
    assert summary["faithful"] == pytest.approx(0.5)
    assert summary["correct"] == pytest.approx(1 / 3)
    assert summary["abstained"] == pytest.approx(1 / 3)
    assert summary["gold_in_context"] == pytest.approx(2 / 3)
    assert summary["unjudged"] == 0


def test_summary_counts_unusable_judge_replies():
    rows = [
        {"abstained": False, "faithful": None, "correct": None, "cited": True, "gold_in_context": True},
        {"abstained": False, "faithful": True, "correct": True, "cited": True, "gold_in_context": True},
    ]
    summary = summarize_answers(rows)
    assert summary["unjudged"] == 2
    assert summary["faithful"] == 1.0  # averaged over judged rows only


# ---- question generation --------------------------------------------------

def test_sample_chunks_respects_length_and_per_document_cap():
    chunks = [Chunk(f"d{d}#{i}", f"d{d}", "t", "", "ก" * 300) for d in range(3) for i in range(5)]
    chunks.append(Chunk("short#0", "short", "t", "", "สั้น"))

    picked = sample_chunks(chunks, n=100, per_doc=2)

    assert len(picked) == 6
    assert all(len(c.text) >= 250 for c in picked)
    assert sample_chunks(chunks, n=4, seed=1) == sample_chunks(chunks, n=4, seed=1)


def test_make_questions_skips_bad_llm_output():
    chunks = [Chunk(f"d{i}#0", f"d{i}", "t", "", "ก" * 300) for i in range(3)]
    replies = iter(['{"question": "ถามอะไร", "answer": "ตอบ"}', "not json", '{"question": "ถามอีก"}'])
    questions = make_questions(chunks, FakeLLM(lambda _p: next(replies)), n=3)

    assert len(questions) == 1
    assert questions[0].id == "q001"
    assert questions[0].gold_chunk_id == questions[0].doc_id + "#0"
