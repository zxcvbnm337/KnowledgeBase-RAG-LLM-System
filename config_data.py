
md5_path = "./md5.text"

# Chroma
collection_name="rag"
persist_directory="./chroma_db"

# spliter
chunk_size= 1000
chunk_overlap= 100
separators =["\n\n","\n",".","!","?","。","！","？"," ",""]

max_spliter_char_number= 1000  # 文本分割阈值

# [已废弃] 早期版本把这个值当作「检索返回条数」使用，命名与语义不符。
# 现由下方 retrieval_top_k / retrieval_final_k / *_threshold 取代，保留仅为兼容旧脚本。
similarity_threshold =1     # 检索返回匹配的文档数量

# ==================== 检索链路：召回 → 重排 → 阈值门控 ====================
# 候选召回条数：向量检索先取这么多，再交给重排精排
retrieval_top_k = 8

# 最终注入 prompt 的片段数
retrieval_final_k = 3

# 是否启用交叉编码重排（DashScope gte-rerank-v2）
enable_rerank = True
rerank_model = "gte-rerank-v2"

# ---- 两个阈值为什么不共用一个？----
# 二者量纲不同，实测分布差别很大，混用必然误判：
#   重排分数（gte-rerank-v2）：库内问题 top1 ≥ 0.16，库外问题 ≤ 0.006  → 阈值取 0.05，两侧留有 26 倍间隔
#   归一化相似度 1/(1+d)      ：库内问题 top1 ≥ 0.51，库外问题 ≤ 0.40  → 阈值取 0.45，间隔较窄
# 结论：重排不仅提升排序质量，更让「阈值拒答」真正可用——向量相似度对库外问题
#       仍会给出 0.39~0.40 的「看似合理」分数，光靠它拒答不可靠。
rerank_threshold = 0.05
vector_similarity_threshold = 0.45

# 命中不足时是否直接拒答（短路，不调用大模型）
# 好处：① 从源头消除幻觉；② 省下一次 LLM 调用
refusal_short_circuit = True
refusal_text = ("抱歉，这个问题不在我的知识库范围内，我暂时无法回答。\n\n"
                "目前知识库覆盖：商品尺码推荐、衣物洗涤养护、颜色搭配建议。"
                "你可以换成这几类问题再问一次。")

# 是否把每次组装好的完整 prompt 打印到控制台（调试用，默认关闭以免刷屏）
debug_print_prompt = False

embedding_model_name="text-embedding-v4"
chat_model_name="qwen3-max"

#
session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}

# ==================== 对话记忆分层 ====================
chat_history_dir = "./chat_history"  # 会话历史落盘目录

memory_recent_messages = 6      # 短期记忆：始终保留最近 6 条消息原文（约 3 轮对话）
memory_summary_trigger = 10     # 长期记忆：消息数超过 10 条时，把更早的内容压缩成摘要
memory_summary_model = "qwen3-max"  # 生成摘要用的模型（与对话模型分开配置，便于降本）

# 历史压缩提示词：只做信息归并，明确禁止编造
memory_summary_prompt = """你是一个对话记忆压缩助手，负责把冗长的对话历史压缩成一段简洁摘要。

要求：
1. 保留事实性信息：用户提到的关键数据、偏好、约束条件、已经确认的结论；
2. 保留尚未解决的问题，方便后续继续追问；
3. 丢弃寒暄、重复确认、以及和业务无关的闲聊；
4. 只依据给定对话内容，不要推测或编造任何信息。

已有摘要（为空表示这是首次压缩）：
{previous_summary}

需要并入摘要的新对话：
{dialogue}

直接输出合并后的摘要正文，不要加标题、编号或任何解释。"""
