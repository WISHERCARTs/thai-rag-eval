"""Generate an evaluation set: one question per sampled chunk, answerable from that chunk."""
import json
import random
from typing import Sequence

from thairag.chunking import Chunk
from thairag.evaluate import Question
from thairag.rag import LLM


def question_prompt(chunk: Chunk) -> str:
    return (
        "จากข้อความด้านล่าง จงเขียนคำถามภาษาไทย 1 ข้อที่ตอบได้จากข้อความนี้ พร้อมคำตอบสั้น\n"
        "ข้อกำหนด:\n"
        '- คำถามต้องเข้าใจได้โดยไม่ต้องเห็นข้อความ ระบุชื่อสถานที่หรือหัวข้อให้ชัด (เช่น "จังหวัดน่าน..." ไม่ใช่ "จังหวัดนี้...")\n'
        "- ถามข้อเท็จจริงที่เจาะจง เช่น ตัวเลข ปี ชื่อ หรือสถานที่\n"
        "- อย่าคัดลอกวลียาวจากข้อความ ให้เรียบเรียงด้วยคำของตัวเอง\n"
        "- คำตอบไม่เกิน 1 ประโยค\n"
        'ตอบเป็น JSON: {"question": "...", "answer": "..."}\n\n'
        f"ชื่อบทความ: {chunk.title}\nหัวข้อ: {chunk.section or '-'}\nข้อความ:\n{chunk.text}"
    )


def sample_chunks(chunks: Sequence[Chunk], n: int, seed: int = 42, min_chars: int = 250, per_doc: int = 3) -> list[Chunk]:
    """Pick up to n substantial chunks, at most per_doc from any one article."""
    pool = [c for c in chunks if len(c.text) >= min_chars]
    random.Random(seed).shuffle(pool)
    picked: list[Chunk] = []
    per_doc_count: dict[str, int] = {}
    for chunk in pool:
        if per_doc_count.get(chunk.doc_id, 0) >= per_doc:
            continue
        picked.append(chunk)
        per_doc_count[chunk.doc_id] = per_doc_count.get(chunk.doc_id, 0) + 1
        if len(picked) == n:
            break
    return picked


def make_questions(chunks: Sequence[Chunk], llm: LLM, n: int, seed: int = 42) -> list[Question]:
    questions: list[Question] = []
    for chunk in sample_chunks(chunks, n, seed):
        try:
            data = json.loads(llm.generate(question_prompt(chunk), json_mode=True))
            question, answer = data["question"].strip(), data["answer"].strip()
        except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
            continue
        if question and answer:
            questions.append(Question(f"q{len(questions) + 1:03d}", question, answer, chunk.id, chunk.doc_id))
    return questions
