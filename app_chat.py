import os

import streamlit as  st
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))  # 读取项目根目录 .env 中的 DASHSCOPE_API_KEY

from rag import RagService
import config_data as config

# 标题
st.title("智能客服")
# 分隔符
st.divider()

# 避免性能压力，session_state 存入对象
if "message" not in st.session_state:
    st.session_state["message"]=[{"role":"assistant","content":"你好，有什么可以帮助你？"}]

if "rag" not in st.session_state:
    if not os.getenv("DASHSCOPE_API_KEY"):
        st.error("未检测到 DASHSCOPE_API_KEY。请复制 .env.example 为 .env，填入你的百炼 Key 后刷新页面。")
        st.stop()
    st.session_state["rag"]= RagService()

# 侧边栏：实时展示分层记忆的工作状态，以及检索链路的关键配置
with st.sidebar:
    st.subheader("🔍 检索链路")
    st.caption(f"召回 Top-{config.retrieval_top_k} → 重排（{config.rerank_model}）→ 注入 Top-{config.retrieval_final_k}")
    st.caption(
        f"重排阈值 {config.rerank_threshold} · 向量相似度阈值 {config.vector_similarity_threshold}"
    )
    st.caption("低于阈值直接拒答，不调用大模型（从源头消除幻觉）")

    st.divider()
    st.subheader("🧠 记忆分层状态")
    st.caption("短期记忆保留最近若干条原文，更早的内容自动压缩成摘要")
    try:
        stats = st.session_state["rag"].memory_stats(
            config.session_config["configurable"]["session_id"]
        )
    except Exception as e:
        stats = None
        st.warning(f"记忆状态读取失败：{e}")

    if stats:
        col_a, col_b = st.columns(2)
        col_a.metric("长期记忆", f"{stats['summary_chars']} 字")
        col_b.metric("短期记忆", f"{stats['recent_messages']} 条")
        st.metric("累计压缩次数", stats["compressed_count"])
        st.progress(
            min(stats["recent_messages"] / stats["summary_trigger"], 1.0),
            text=f"达到 {stats['summary_trigger']} 条触发压缩",
        )
        st.caption(
            f"配置：短期窗口 {stats['raw_message_limit']} 条 · "
            f"触发阈值 {stats['summary_trigger']} 条"
        )


def render_retrieval_panel(outcome):
    """把一次检索的中间状态摊开给用户看——检索质量必须能被观察，否则调参只能靠感觉。"""
    if outcome is None:
        return
    debug = outcome.debug or {}
    label = "🔍 检索过程（已拒答）" if outcome.refused else "🔍 检索过程"
    with st.expander(label, expanded=outcome.refused):
        col1, col2, col3 = st.columns(3)
        col1.metric("候选片段", debug.get("candidates", 0))
        col2.metric("最高相关度", f"{debug.get('best_score', 0):.4f}")
        col3.metric("通过阈值", len(outcome.chunks))
        st.caption(
            f"判定依据：{debug.get('score_source', '-')} · "
            f"阈值 {debug.get('threshold')} · "
            f"被阈值挡下 {debug.get('dropped_by_threshold', 0)} 条"
        )
        if debug.get("rerank_degraded"):
            st.warning(f"重排已降级，退回向量顺序：{debug.get('rerank_error', '')}")
        if outcome.refused:
            st.info(f"拒答原因：{outcome.reason}")
        for i, chunk in enumerate(outcome.chunks, 1):
            st.markdown(f"**片段 {i}** ｜ 相关度 `{chunk.score:.4f}` ｜ 来源 `{chunk.metadata.get('source', '未知')}`")
            st.caption(chunk.preview(120))


#循环 输出历史信息，原本只记录但页面不显示
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])
# 在页面最下方提供用户输入栏
prompt= st.chat_input()

if prompt :
    # 在页面输出用户的提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})

    ai_res_list= []
    with st.spinner("AI 思考中......."):
        # 调用 RAG 服务：检索（含重排与阈值门控）→ 命中才调用大模型
        res_stream = st.session_state["rag"].stream(prompt, config.session_config)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk
        st.chat_message("assistant").write_stream(capture(res_stream,ai_res_list))
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})

    # 无论是否拒答，都把这次检索的依据展示出来
    render_retrieval_panel(st.session_state["rag"].last_retrieval)
