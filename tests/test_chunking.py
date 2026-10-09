from thairag.chunking import chunk_document, split_long

ARTICLE = """น่านเป็นจังหวัดในภาคเหนือของประเทศไทย มีประวัติศาสตร์ยาวนานกว่าเจ็ดร้อยปี

== ประวัติศาสตร์ ==
เมืองน่านเคยเป็นนครรัฐอิสระก่อนรวมเข้ากับอาณาจักรล้านนา และต่อมาเป็นส่วนหนึ่งของสยาม
=== สมัยใหม่ ===
ในปี พ.ศ. 2474 น่านเปลี่ยนจากเมืองประเทศราชมาเป็นจังหวัดหนึ่งของไทยอย่างเป็นทางการ

== อ้างอิง ==
หนังสือประวัติเมืองน่าน ฉบับพิมพ์ครั้งที่สอง สำนักพิมพ์ตัวอย่าง กรุงเทพมหานคร
=== เว็บไซต์ ===
ลิงก์ไปยังเว็บไซต์ของจังหวัดน่านและหน่วยงานท่องเที่ยวต่างๆ ที่เกี่ยวข้อง

== การท่องเที่ยว ==
วัดภูมินทร์เป็นวัดที่มีชื่อเสียงที่สุดของจังหวัดน่าน มีภาพจิตรกรรมฝาผนังปู่ม่านย่าม่าน
"""


def test_sections_are_tracked_and_reference_sections_skipped():
    chunks = chunk_document("6471", "จังหวัดน่าน", ARTICLE, max_chars=120)
    sections = [c.section for c in chunks]

    assert sections == ["", "ประวัติศาสตร์", "สมัยใหม่", "การท่องเที่ยว"]
    assert not any("หนังสือประวัติ" in c.text or "ลิงก์ไปยัง" in c.text for c in chunks)


def test_chunk_ids_are_sequential_per_document():
    chunks = chunk_document("6471", "จังหวัดน่าน", ARTICLE, max_chars=120)
    assert [c.id for c in chunks] == [f"6471#{i}" for i in range(len(chunks))]
    assert all(c.doc_id == "6471" and c.title == "จังหวัดน่าน" for c in chunks)


def test_paragraphs_in_one_section_are_merged_up_to_the_limit():
    text = "== ก ==\n" + "\n".join(["ย่อหน้าที่มีความยาวพอสมควร " * 3] * 4)
    chunks = chunk_document("1", "t", text, max_chars=200)

    assert all(len(c.text) <= 200 for c in chunks)
    assert len(chunks) < 4  # several paragraphs share a chunk


def test_tiny_fragments_are_dropped():
    chunks = chunk_document("1", "t", "== ก ==\nสั้น\n== ข ==\n" + "ข้อความยาวพอที่จะเก็บไว้เป็นหนึ่ง chunk ได้ " * 2)
    assert [c.section for c in chunks] == ["ข"]


def test_index_text_includes_title_and_section():
    chunk = chunk_document("1", "จังหวัดน่าน", "== ประวัติ ==\n" + "เนื้อหา " * 20)[0]
    assert chunk.index_text.startswith("จังหวัดน่าน — ประวัติ\n")


def test_split_long_prefers_spaces_and_respects_limit():
    text = "คำ" * 30 + " " + "ประโยค" * 30 + " " + "จบ" * 10
    pieces = split_long(text, 100)

    assert all(len(p) <= 100 for p in pieces)
    assert "".join(pieces).replace(" ", "") == text.replace(" ", "")
    assert pieces[0] == "คำ" * 30


def test_split_long_hard_cuts_text_without_spaces():
    pieces = split_long("ก" * 250, 100)
    assert [len(p) for p in pieces] == [100, 100, 50]
