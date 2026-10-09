"""Answer questions in Thai from retrieved chunks, citing them by number."""
from dataclasses import dataclass
from typing import Protocol, Sequence

from thairag.chunking import Chunk
from thairag.retrievers import Retriever

NOT_FOUND = "ไม่พบข้อมูลในเอกสาร"

SYSTEM_PROMPT = f"""คุณเป็นผู้ช่วยตอบคำถามภาษาไทยจากเอกสารที่ให้มาเท่านั้น
กฎ:
1. ตอบโดยใช้ข้อมูลในส่วน "เอกสาร" เท่านั้น ห้ามใช้ความรู้ภายนอก
2. ใส่เลขอ้างอิงในวงเล็บเหลี่ยม เช่น [1] หรือ [2][3] ท้ายประโยคที่ใช้ข้อมูลจากเอกสารนั้น
3. ถ้าเอกสารไม่มีคำตอบ ให้ตอบว่า "{NOT_FOUND}" เท่านั้น
4. ตอบสั้น กระชับ ตรงคำถาม"""


class LLM(Protocol):
    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str: ...


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Chunk]

    @property
    def context(self) -> str:
        return format_context(self.sources)


def format_context(chunks: Sequence[Chunk]) -> str:
    return "\n\n".join(f"[{i}] {chunk.index_text}" for i, chunk in enumerate(chunks, start=1))


class RagPipeline:
    def __init__(self, retriever: Retriever, chunks: Sequence[Chunk], llm: LLM, k: int = 5):
        self.retriever = retriever
        self.by_id = {c.id: c for c in chunks}
        self.llm = llm
        self.k = k

    def retrieve(self, question: str) -> list[Chunk]:
        return [self.by_id[chunk_id] for chunk_id, _ in self.retriever.search(question, self.k)]

    def ask(self, question: str) -> Answer:
        sources = self.retrieve(question)
        if not sources:
            return Answer(question, NOT_FOUND, [])
        prompt = f"เอกสาร:\n{format_context(sources)}\n\nคำถาม: {question}"
        return Answer(question, self.llm.generate(prompt, system=SYSTEM_PROMPT).strip(), sources)
