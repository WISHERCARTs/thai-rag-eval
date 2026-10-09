"""Split plain-text Wikipedia articles into retrieval chunks."""
import re
from dataclasses import asdict, dataclass

HEADING = re.compile(r"^(=+)\s*(.+?)\s*\1$")

# Sections that hold links and citations rather than content
SKIP_SECTIONS = frozenset(
    {"ดูเพิ่ม", "อ้างอิง", "แหล่งข้อมูลอื่น", "เชิงอรรถ", "บรรณานุกรม", "หนังสืออ่านเพิ่ม", "หมายเหตุ"}
)


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    section: str
    text: str

    @property
    def index_text(self) -> str:
        """Text used for embedding and BM25: the passage plus where it comes from."""
        header = self.title if not self.section else f"{self.title} — {self.section}"
        return f"{header}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def split_long(text: str, max_chars: int) -> list[str]:
    """Split text longer than max_chars at the last space before the limit.

    Thai separates sentences and clauses with spaces, so a space is the safest
    break point. Text with no space in range is cut hard.
    """
    pieces = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut <= max_chars // 3:
            cut = max_chars
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return pieces


def chunk_document(
    doc_id: str, title: str, text: str, max_chars: int = 700, min_chars: int = 40
) -> list[Chunk]:
    """Group paragraphs into chunks of at most max_chars, never crossing a section."""
    chunks: list[Chunk] = []
    section = ""
    skip_top = False  # inside a skipped level-2 section
    skip_sub = False  # inside a skipped deeper section
    buffer: list[str] = []

    def flush():
        body = "\n".join(buffer).strip()
        buffer.clear()
        if len(body) >= min_chars:
            chunks.append(Chunk(f"{doc_id}#{len(chunks)}", doc_id, title, section, body))

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        heading = HEADING.match(line)
        if heading:
            flush()
            section = heading.group(2)
            if len(heading.group(1)) <= 2:
                skip_top = section in SKIP_SECTIONS
                skip_sub = False
            else:
                skip_sub = section in SKIP_SECTIONS
            continue
        if skip_top or skip_sub:
            continue
        for piece in split_long(line, max_chars):
            if buffer and sum(len(p) + 1 for p in buffer) + len(piece) > max_chars:
                flush()
            buffer.append(piece)
    flush()
    return chunks
