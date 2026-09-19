# KnowledgeBase-RAG-LLM-System

> 基于 **Streamlit + LangChain + Chroma** 的本地知识库 RAG 问答系统：
> 网页上传 `txt` 自动切分入库，聊天式提问走检索增强回答，并带**分层对话记忆**。

---

## ⚠️ 来源说明（请先读这段）

本仓库是一份**二次开发的学习项目**，上游为
**[lhh737/KnowledgeBase-RAG-LLM-System](https://github.com/lhh737/KnowledgeBase-RAG-LLM-System)**（MIT License，
版权归原作者所有，见 [LICENSE](./LICENSE)）。

上游提供了完整的 RAG 基础链路（切分 / 向量化 / 检索 / 流式对话）。
本仓库在其基础上做了**工程化改造**，改动集中在「依赖可安装性、配置与密钥管理、
对话记忆分层」三块，逐条列在 [我对上游做的改动](#-我对上游做的改动) 中。

> Git 历史保留了上游的原始提交，本仓库新增的提交均可在 `git log` 中按作者区分。
> 这不是需要回避的事——MIT 许可证允许修改与再分发，**唯一的要求就是保留原版权声明**。

---

## ✨ 功能一览

### 1) 知识库更新服务（Upload）
- Streamlit 页面上传 `txt`，自动读取内容
- `RecursiveCharacterTextSplitter` 分段（可配置 chunk / overlap / 分隔符）
- 写入 Chroma 向量库（本地持久化）
- **MD5 去重**：相同内容不重复入库

### 2) 智能客服（RAG Chat）
- Streamlit Chat UI，支持 **流式输出**
- LangChain LCEL 链式编排：`Retrieval -> Prompt -> LLM -> Output`
  并行分支同时取用户输入与检索结果，再汇入同一个 prompt
- 会话历史落盘，支持连续追问

### 3) 分层对话记忆（本项目新增）
- **短期记忆**：保留最近 N 条消息原文，保证「上面那个」「它」这类指代能被正确理解
- **长期记忆**：更早的消息由模型压缩成一段摘要，只保留关键事实与未解决问题
- 于是对话再长，注入 prompt 的历史长度都稳定在一个上界内 —— 详见下节

---

## 📸 界面预览

> 下面两张是**上游项目**的运行截图，本仓库的界面在此基础上新增了左侧的「记忆分层状态」
> 面板。建议替换为你自己的运行截图。

<div align="left">
  <img src="./assets/chat_demo1.png" width="720" alt="智能客服聊天界面">
</div>

<div align="left">
  <img src="./assets/chat_demo2.png" width="720" alt="连续问答界面">
</div>

本地知识库预置了衣物尺码推荐、材质维护、穿衣搭配等示例内容（见 `assets/`），
对标电商客服场景，可替换为自己的业务文本。

---

## 🔧 我对上游做的改动

### 1. 对话记忆分层（核心改动）

**上游的问题**：`add_messages` 每次把整个历史文件读出来、追加、再整个写回，
消息**只增不减**。对话到几十轮后：

- 注入 prompt 的历史随轮次线性增长，token 成本持续上涨；
- 最终必然超出模型上下文窗口，直接报错中断对话。

**改造后的行为**：把记忆拆成两层，落盘为
`{"summary": "...", "messages": [...], "compressed_count": n}`：

| 层 | 内容 | 作用 |
|---|---|---|
| 短期 | 最近 `memory_recent_messages` 条消息原文 | 保留细节与指代关系 |
| 长期 | 更早消息压缩成的摘要（会与旧摘要继续合并） | 保留关键事实，不丢上下文 |

消息数超过 `memory_summary_trigger` 时触发压缩：取最早的一批交给模型生成摘要，
与已有摘要合并后写回，磁盘上只保留短期窗口内的原文。
压缩失败或摘要为空时**保留全部原文**——宁可多花 token，也不能把历史丢掉。

**实测效果**（真实模型 12 轮对话，窗口 6 条 / 阈值 10 条）：

| 指标 | 改造前 | 改造后 |
|---|---|---|
| 注入 prompt 的历史 | 25 条，且随轮次继续增长 | **8 条，且始终不超过 12 条** |
| 磁盘上的原文条数 | 25 条，只增不减 | **稳定在 6 ~ 10 条之间** |
| 长期摘要长度 | — | 459 字（12 轮对话的全部关键信息） |
| 压缩次数 | 0 | 3 次（摘要逐次合并，不是每次从头摘要） |

**关键验证**：第 1 轮用户说的「身高 180、体重 75」，此时原文早已被压缩出短期窗口 ——
但第 9 轮和第 12 轮追问「我的身高体重是多少」，模型都准确答出。
说明分层记忆是**真的记住了**，而不是简单截断历史。

（长跑测试：25 轮对话下注入历史峰值 12 条，不压缩则为 51 条。）

涉及的实现：`file_history_store.py`（分层存储）、`rag.py`（摘要链路与历史工厂）、
`config_data.py`（参数与摘要提示词）。

### 2. 依赖在 Python 3.13 上可安装

上游 `requirements.txt` 锁定 `numpy<2.0` / `langchain<0.2` / `chromadb<0.5`，
这些版本**在 Python 3.13 上没有可用的预编译包**，`pip install -r requirements.txt` 会直接失败。
已升级为一组互相兼容的版本：`langchain 0.3.x` / `langchain-chroma` / `chromadb 1.x` /
`streamlit 1.4x` / `dashscope 1.2x`。

> 兼容性已实测：`ChatTongyi` 与新版 `dashscope` 协同正常，嵌入与对话模型均可用。

### 3. 密钥与配置管理

- 新增 `.env` 支持：`python-dotenv` 原本写在依赖里、注释也说要「安全存储 API Key」，
  但代码里**从未调用过 `load_dotenv()`**，Key 只能靠手动设系统环境变量。已补上（显式指定
  项目根目录的 `.env`，与启动时的工作目录无关）。
- 新增 `.env.example` 占位模板。
- Key 缺失时页面给出明确提示，而不是抛一屏 traceback。

### 4. `.gitignore` 补漏

上游写的是 `vector_db/`，但代码实际使用 `./chroma_db`；且 `.env`、`md5.text`、
`chat_history/` 均未被忽略。也就是说执行一次 `git add .` 就会把 **API Key、向量库、
完整对话记录**一起提交上去。已按代码里真实使用的路径补齐。

### 5. 一键启动

新增 `start.bat`：双击后启动两个页面并自动打开浏览器。

---

## 🧩 项目结构

```text
KnowledgeBase-RAG-LLM-System/
├─ app_upload.py              # 知识库上传服务（Streamlit）
├─ app_chat.py                # 智能客服问答（Streamlit，含记忆状态侧边栏）
├─ knowledge_base.py          # 知识库处理：读取、切分、写库、去重
├─ rag.py                     # RAG 链组装 + 摘要链路
├─ vector_stores.py           # 向量库检索封装（持久化）
├─ file_history_store.py      # 分层对话记忆（短期原文 + 长期摘要）
├─ config_data.py             # 模型、路径、chunk、记忆分层等参数配置
├─ requirements.txt           # 项目依赖
├─ start.bat                  # 一键启动两个页面
├─ .env.example               # 环境变量模板
├─ tests/                     # 记忆分层的测试
│  ├─ test_memory_layering.py #   离线测试（桩摘要器，不消耗 API 额度）
│  └─ test_memory_live.py     #   真实联调（需 API Key，会消耗额度）
└─ assets/                    # 演示图片与示例素材文本
```

---

## ✅ 环境准备

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows（macOS/Linux: source .venv/bin/activate）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## ⚙️ 配置说明

### 1) API Key

复制 `.env.example` 为 `.env`，填入百炼（DashScope）Key：

```ini
DASHSCOPE_API_KEY=sk-你的key
```

`.env` 已被 `.gitignore` 忽略，不会进版本库。

### 2) 模型与检索参数

在 `config_data.py` 中修改：嵌入模型、对话模型、chunk 大小、检索条数等。
默认嵌入 `text-embedding-v4`、对话 `qwen3-max`。

---

## 🎛 记忆分层参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `chat_history_dir` | `./chat_history` | 会话历史落盘目录 |
| `memory_recent_messages` | `6` | 短期记忆窗口（条），保留最近约 3 轮原文 |
| `memory_summary_trigger` | `10` | 消息数超过该值触发压缩（内部保证 > 短期窗口） |
| `memory_summary_model` | `qwen3-max` | 生成摘要用的模型，可与对话模型分开配置以降本 |
| `memory_summary_prompt` | 见文件 | 摘要提示词，明确要求「只归并、不编造」 |

启动对话页后，**侧边栏「🧠 记忆分层状态」**会实时显示长期摘要字数、短期原文条数、
累计压缩次数与距下次压缩的进度。想直观看到压缩发生，把 `memory_summary_trigger`
临时改成 `6` 聊几轮即可。

---

## 🚀 快速运行

### 方式一：一键启动

双击 `start.bat`（内部会先 `cd` 到项目目录，避免相对路径失效）。

### 方式二：分别启动

```bash
streamlit run app_upload.py        # 知识库上传页，默认 8502
streamlit run app_chat.py          # 智能客服页，默认 8501
```

> ⚠️ **必须从项目根目录启动**。代码使用相对路径（`./chroma_db`、`./md5.text`、
> `./chat_history`），换目录启动会读到一个空的向量库，看起来就像「数据丢了」。

---

## 🧪 测试

分层记忆配了两套测试，都不需要额外依赖：

```bash
# 1) 离线测试：用桩摘要器替代模型，不消耗 API 额度、不需要 API Key
python tests/test_memory_layering.py

# 2) 真实联调：跑 12 轮对话，验证摘要生成与跨压缩的记忆召回
python tests/test_memory_live.py
```

**离线测试**（26 项断言全部通过）覆盖：

- 短期窗口与压缩触发时机
- **注入 prompt 的历史存在上界**（25 轮对话峰值 12 条，不压缩则为 51 条）
- 兼容旧版「纯消息列表」历史文件并自动迁移
- 摘要失败时保留全部原文（不丢历史）
- `clear`、多会话隔离、`session_id` 路径字符过滤

**真实联调**（8 项断言全部通过）验证：摘要由真实模型生成且长度受控；
第 1 轮说过的事实，在经历 3 次压缩后仍能被准确追问出来。

---

## 🛠 常见问题

### Q1：上传文件后，问答仍然像「没有检索到资料」？
- 上传页与问答页使用了不同的向量库目录，或 `collection_name` 不一致
- 上传后未正确写入本地数据目录

### Q2：回答变慢或没有输出？
- 切分/检索参数不合适，调整 `chunk_size` 与检索条数
- 模型接口或网络响应慢

### Q3：报路径或配置错误？
- 检查 `config_data.py` 中的路径配置、本地数据目录是否存在
- 检查 `.env` 是否已创建且 Key 正确
- 确认从**项目根目录**启动

### Q4：历史记录文件长什么样？
`chat_history/<session_id>.json`，结构为：

```json
{
  "version": 2,
  "summary": "更早对话的摘要……",
  "messages": [ { "type": "human", "data": { "content": "……" } } ],
  "compressed_count": 3
}
```

旧版本的无扩展名历史文件（纯消息列表）会在首次读取时自动兼容迁移。

---

## 🚧 后续可改进方向

- **Rerank 重排**：当前只有向量召回，加一层重排可明显提升命中质量
- **检索阈值拒答**：知识库外的问题目前照样硬答，是幻觉的主要来源
- 支持更多文件类型（PDF / Markdown / Word）
- 短期窗口按 token 数而非消息条数控制

---

## 📄 License

MIT License，见 [LICENSE](./LICENSE)。

- 原始版权：上游项目作者
- 本仓库改造部分：见 `git log` 中对应提交

---

## 🙌 致谢

- [lhh737/KnowledgeBase-RAG-LLM-System](https://github.com/lhh737/KnowledgeBase-RAG-LLM-System) —— 本项目的基础
- Streamlit / LangChain / Chroma / chromadb
- 阿里云百炼（DashScope）/ 通义千问
