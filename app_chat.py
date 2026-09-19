import os

import streamlit as  st
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))  # 读取项目根目录 .env 中的 DASHSCOPE_API_KEY

from rag import RagService
import config_data as config
import time

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

# 侧边栏：实时展示分层记忆的工作状态
# 短期 = 保留原文的最近若干条；长期 = 被压缩成摘要的更早对话
with st.sidebar:
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
        # 调用 RAG服务

        # 直接输出
        # res= st.session_state["rag"].chain.invoke({"input":prompt},config.session_config)
        #
        # st.chat_message("assistant").write(res)
        # st.session_state["message"].append({"role":"assistant","content":res})

        # 流式输出
        res_stream = st.session_state["rag"].chain.stream({"input": prompt}, config.session_config)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk
        st.chat_message("assistant").write_stream(capture(res_stream,ai_res_list))
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})

