"""
基于Streamlit完成WEB网页上传服务

Streamlit ： 当WEB页面元素变化，则代码重新执行一遍，无法维护状态
"""
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))  # 读取项目根目录 .env 中的 DASHSCOPE_API_KEY

from knowledge_base import KnowledgeBaseService
from loaders import ParseError, SUPPORTED_EXTENSIONS, UnsupportedFileType, load_document

# 添加网页标题
st.title("知识库更新服务")
st.caption("支持格式：" + "、".join(e.lstrip('.') for e in SUPPORTED_EXTENSIONS)
           + "　｜　文本类直接入库，PDF / Word / Excel 会先抽取正文再切分")

# file_uploader
uploader_file=st.file_uploader(
    "请上传文件",
    type=[e.lstrip('.') for e in SUPPORTED_EXTENSIONS],
    accept_multiple_files=False,   # False表示仅接受一个文件的上传
)

if "service" not in st.session_state:          # 会话状态字典，session_state本身也是字典
    if not os.getenv("DASHSCOPE_API_KEY"):
        st.error("未检测到 DASHSCOPE_API_KEY。请复制 .env.example 为 .env，填入你的百炼 Key 后刷新页面。")
        st.stop()
    st.session_state["service"]=KnowledgeBaseService()

if uploader_file is not None:
    # 提取文件信息
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size /1024
    st.subheader(f"文件名:{file_name}")
    st.write(f"格式:{file_type}  | 大小:{file_size:.2f}KB")

    raw = uploader_file.getvalue()

    # 解析阶段单独兜错：格式不支持 / 扫描件无文字 / 编码异常，都在这里给出人话提示，
    # 而不是让用户面对一屏 traceback
    try:
        text, meta = load_document(file_name, raw)
    except UnsupportedFileType as e:
        st.error(str(e))
        st.stop()
    except ParseError as e:
        st.error(f"解析失败：{e}")
        st.stop()

    st.write(f"已抽取正文 {meta['chars']} 字"
             + (f" ｜ 共 {meta['pages']} 页" if meta.get("pages") else "")
             + (f" ｜ 工作表 {meta['sheets']} 个 / {meta['rows']} 行" if meta.get("rows") else "")
             + (f" ｜ 段落 {meta['paragraphs']} 段" if meta.get("paragraphs") else ""))
    if meta.get("empty_pages"):
        st.warning(f"以下页面未抽到文字（疑似图片页，建议 OCR）：{meta['empty_pages']}")

    with st.expander("预览抽取结果（前 500 字）"):
        st.text(text[:500])

    with st.spinner("载入知识库中。。。"):     # 在 spinner内的代码执行过程中，会有一个转圈动画，优化用户体验
        result= st.session_state["service"].upload_by_str(text, file_name, extra_metadata=meta)
        st.write(result)
