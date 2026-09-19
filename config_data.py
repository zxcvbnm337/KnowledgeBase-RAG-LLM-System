
md5_path = "./md5.text"

# Chroma
collection_name="rag"
persist_directory="./chroma_db"

# spliter
chunk_size= 1000
chunk_overlap= 100
separators =["\n\n","\n",".","!","?","。","！","？"," ",""]

max_spliter_char_number= 1000  # 文本分割阈值

# 相似度K值
similarity_threshold =1     # 检索返回匹配的文档数量

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
