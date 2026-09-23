"""多格式文档解析的离线测试。

**不需要 API Key、不联网**，但 PDF / Word / Excel 三项需要对应的可选依赖
（`pypdf` / `python-docx` / `openpyxl`，见 requirements.txt 的「可选优化依赖」）。
依赖缺失时该格式会标记为 SKIP 而不是 FAIL，避免误报红。

    python tests/test_loaders.py

覆盖点：
  1. 纯文本类的编码兜底（UTF-8 / GBK 都能读）
  2. Word：段落 + **表格**都要抽出来（只读段落会整片丢表格内容）
  3. Excel：逐工作表转文本
  4. PDF：可正常读取页数；**扫描件（无文本层）要给出 OCR 提示而不是静默入库**
  5. 不支持的后缀 / 空内容 / 损坏文件，都要抛明确异常
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders import (  # noqa: E402
    ParseError,
    SUPPORTED_EXTENSIONS,
    UnsupportedFileType,
    load_document,
)

_results = []
_skipped = []


def check(label, cond, extra=""):
    _results.append(bool(cond))
    print(("  [PASS] " if cond else "  [FAIL] ") + label + (f"  {extra}" if extra else ""))


def skip(label, reason):
    _skipped.append(label)
    print(f"  [SKIP] {label}  （{reason}）")


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def make_docx() -> bytes:
    import docx

    d = docx.Document()
    d.add_paragraph("这是正文段落一。")
    d.add_paragraph("这是正文段落二。")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text = "尺码"
    t.cell(0, 1).text = "身高范围"
    t.cell(1, 0).text = "XL"
    t.cell(1, 1).text = "175-185cm"
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def make_xlsx() -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "尺码表"
    ws.append(["尺码", "身高", "体重"])
    ws.append(["XL", "175-185", "140-160"])
    ws2 = wb.create_sheet("洗涤说明")
    ws2.append(["材质", "水温"])
    ws2.append(["纯棉", "≤30℃"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def make_blank_pdf() -> bytes:
    """一个只有空白页、没有文本层的 PDF —— 等价于扫描件。"""
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=595, height=842)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def test_text_encodings():
    section("测试 1：文本类文件的编码兜底")
    text, meta = load_document("a.txt", "身高180的建议尺码是XL。".encode("utf-8"))
    check("UTF-8 正常解析", "建议尺码是XL" in text)
    check("返回字符数元数据", meta["chars"] == len(text), f"chars={meta['chars']}")
    check("记录了原始文件名", meta["original_name"] == "a.txt")

    gbk = "身高180的建议尺码是XL。".encode("gbk")
    text_gbk, _ = load_document("gbk.txt", gbk)
    check("GBK 文件也能读（不会直接崩）", text_gbk == text, f"实际 {text_gbk!r}")

    text_md, meta_md = load_document("readme.md", "# 标题\n\n正文内容".encode("utf-8"))
    check("Markdown 按文本读取", "正文内容" in text_md and meta_md["source_type"] == "md")

    text_csv, meta_csv = load_document("t.csv", "尺码,XL\n身高,180".encode("utf-8"))
    check("CSV 按文本读取", "尺码,XL" in text_csv and meta_csv["source_type"] == "csv")


def test_docx():
    section("测试 2：Word —— 段落与表格都要抽出来")
    try:
        data = make_docx()
    except ImportError as e:
        skip("Word 解析", f"缺少 python-docx：{e}")
        return

    text, meta = load_document("规格.docx", data)
    check("段落被抽出", "这是正文段落一" in text and "这是正文段落二" in text)
    check("表格内容被抽出", "175-185cm" in text,
          "只读段落会丢掉表格 —— 这是最常见的隐性解析劣化")
    check("统计了表格数", meta["tables"] == 1, f"实际 {meta.get('tables')}")
    check("source_type 正确", meta["source_type"] == "docx")


def test_xlsx():
    section("测试 3：Excel —— 逐工作表转文本")
    try:
        data = make_xlsx()
    except ImportError as e:
        skip("Excel 解析", f"缺少 openpyxl：{e}")
        return

    text, meta = load_document("表.xlsx", data)
    check("第一个工作表被抽出", "175-185" in text)
    check("第二个工作表也被抽出", "纯棉" in text, "只看首个 sheet 会漏数据")
    check("带上了工作表名", "工作表：洗涤说明" in text)
    check("sheet 数与行数被统计", meta["sheets"] == 2 and meta["rows"] == 4, f"实际 {meta}")


def test_pdf():
    section("测试 4：PDF —— 读取页数，扫描件要给出 OCR 提示")
    try:
        data = make_blank_pdf()
    except ImportError as e:
        skip("PDF 解析", f"缺少 pypdf：{e}")
        return

    try:
        load_document("扫描件.pdf", data)
        check("无文本层 PDF 应抛 ParseError", False, "实际未抛异常")
    except ParseError as e:
        check("无文本层 PDF 抛 ParseError", True)
        check("错误信息提示需要 OCR", "OCR" in str(e), str(e)[:80])
    except Exception as e:
        check("无文本层 PDF 抛 ParseError", False, f"抛了 {type(e).__name__}：{e}")


def test_rejections():
    section("测试 5：不支持格式 / 空内容 / 损坏文件")
    try:
        load_document("a.xls", b"whatever")
        check("旧版 .xls 应被拒绝", False)
    except UnsupportedFileType as e:
        check("旧版 .xls 被明确拒绝", True)
        check("错误信息列出支持的格式", ".xlsx" in str(e) or "xlsx" in str(e), str(e)[:70])

    try:
        load_document("noext", b"abc")
        check("无后缀文件应被拒绝", False)
    except UnsupportedFileType:
        check("无后缀文件被拒绝", True)

    try:
        load_document("empty.txt", b"   \n  ")
        check("空内容应抛 ParseError", False)
    except ParseError as e:
        check("空内容抛 ParseError", True, str(e)[:60])

    try:
        load_document("broken.docx", b"\x00\x01\x02not-a-zip")
        check("损坏的 docx 应抛 ParseError", False)
    except ParseError:
        check("损坏的 docx 抛 ParseError（而不是裸抛 zipfile 异常）", True)
    except ImportError:
        skip("损坏的 docx", "缺少 python-docx")
    except Exception as e:
        check("损坏的 docx 抛 ParseError", False, f"抛了 {type(e).__name__}")

    check("支持格式清单包含六种",
          set(SUPPORTED_EXTENSIONS) == {".txt", ".md", ".csv", ".pdf", ".docx", ".xlsx"},
          f"实际 {SUPPORTED_EXTENSIONS}")


def main():
    test_text_encodings()
    test_docx()
    test_xlsx()
    test_pdf()
    test_rejections()

    passed, total = sum(_results), len(_results)
    print()
    print("=" * 72)
    print(f"通过 {passed}/{total}" + (f"　跳过 {len(_skipped)} 项" if _skipped else ""))
    if passed == total:
        print("全部通过 ✅")
    else:
        print("存在失败项 ❌")
    print("=" * 72)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
