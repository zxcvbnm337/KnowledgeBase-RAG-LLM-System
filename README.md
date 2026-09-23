# KnowledgeBase-RAG-LLM-System

> 基于 **Streamlit + LangChain + Chroma** 的本地知识库 RAG 问答系统：
> 网页上传 **六种格式的文档**自动解析切分入库，聊天式提问走
> **召回 → 重排 → 阈值门控**的检索增强回答，带**分层对话记忆**，
> 并对检索质量做**可量化评测**。

---

## ⚠️ 来源说明（请先读这段）

本仓库是一份**二次开发的学习项目**，上游为
**[lhh737/KnowledgeBase-RAG-LLM-System](https://github.com/lhh737/KnowledgeBase-RAG-LLM-System)**（MIT License，
版权归原作者所有，见 [LICENSE](./LICENSE)）。

上游提供了完整的 RAG 基础链路（切分 / 向量化 / 检索 / 流式对话）。
本仓库在其基础上做了**工程化改造**，改动集中在四块：

1. **对话记忆分层** —— 短期原文 + 长期摘要，注入历史长度有上界
2. **检索链路补齐** —— 召回带分数、交叉编码重排、阈值拒答、离线评测
3. **多格式文档解析** —— `txt / md / csv / pdf / docx / xlsx`
4. **工程细节** —— 依赖可安装性、配置与密钥治理、若干真实缺陷修复

逐条列在 [我对上游做的改动](#-我对上游做的改动) 中。

> Git 历史保留了上游的原始提交，本仓库新增的提交均可在 `git log` 中按作者区分。
> 这不是需要回避的事——MIT 许可证允许修改与再分发，**唯一的要求就是保留原版权声明**。

---

## ✨ 功能一览

### 1) 知识库更新服务（Upload）
- Streamlit 页面上传文件，自动解析正文
- **支持 `txt` / `md` / `csv` / `pdf` / `docx` / `xlsx`** 六种格式：
  PDF 逐页抽文（并标出未抽到文字的疑似图片页）、Word 段落 + 表格、Excel 逐工作表转文本
- 文本编码自动兜底（`utf-8-sig` → `utf-8` → `gbk` → `gb18030`），GBK 文件不再直接报错
- `RecursiveCharacterTextSplitter` 分段（可配置 chunk / overlap / 分隔符）
- 写入 Chroma 向量库（本地持久化），并把解析阶段的元数据（类型 / 页数 / 工作表 / 行数）一并入库
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

### 4) 检索链路：召回 → 重排 → 阈值拒答（本项目新增）
- **召回**：向量检索取 Top-K 候选，同时带出距离分数并归一化为相似度
- **重排**：用交叉编码模型 `gte-rerank-v2` 对候选精排，解决「向量相似 ≠ 语义相关」
- **阈值门控**：最高相关度低于阈值时**直接拒答、不调用大模型** —— 从源头消除幻觉，同时省下一次 LLM 调用
- **优雅降级**：重排服务不可用时自动退回向量顺序，问答链路不中断
- 页面侧边栏实时展示每次检索的候选数、最高相关度、判定依据与命中片段

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

### 6. 检索链路补齐：召回 → 重排 → 阈值拒答

**上游的问题**：`as_retriever()` 只把文档正文交给下游，**距离分数被丢掉了**。
没有分数，就无法判断「这次召回到底靠不靠谱」，只能无条件把检索结果塞进 prompt——
知识库里没有的问题也照样硬答，这正是 RAG 幻觉最主要的来源。

**改造后**：新增 `retrieval.py`，把检索拆成三个可独立观测的环节：

| 环节 | 做什么 | 关键点 |
|---|---|---|
| 召回 | 向量检索 Top-K，保留距离并归一化为相似度 | `1/(1+d)` 而非 `1-d`，因为 L2 距离可能大于 1，相减会出负数 |
| 重排 | `gte-rerank-v2` 交叉编码精排 | 失败自动降级为向量顺序，且**管线层再加一道 try/except 兜底** |
| 门控 | 最高分低于阈值 → 直接拒答 | 不调用大模型，从源头消除幻觉 |

**为什么要两个阈值**：两种分数**量纲不同**，实测分布差别很大，共用一个必然误判。

| 分数 | 库内问题 top1 | 库外问题 top1 | 判定间隔 |
|---|---|---|---|
| 归一化相似度 `1/(1+d)` | 0.516 ~ 0.668 | 0.382 ~ 0.408 | **1.26×** |
| 重排分数 `gte-rerank-v2` | 0.161 ~ 0.827 | 0.0060 ~ 0.0061 | **26.52×** |

所以默认取 `vector_similarity_threshold = 0.45`、`rerank_threshold = 0.05`。

> **一个需要说明白的结论**：在本仓库当前 **6 个片段**的小语料上，加与不加
> 重排的 **Hit@1 / Hit@3 / MRR 完全相同**（均为 7/7、1.000）——语料太小，
> 评测集区分度不足，重排的**排序收益体现不出来**。
> 重排真正带来的是**判定间隔从 1.26× 放大到 26.52×**：
> 阈值从「必须卡在两个分布之间 0.11 的缝隙里」变成「留出 26 倍安全裕度」。
> 换句话说，**重排的价值不只是排得更准，更是让「知道自己不知道」变得可靠**。
> 语料扩大后需要重新评测——这也正是把评测脚本固化进仓库的意义。

### 7. 多格式文档解析

上游只支持上传 `txt`。但真实业务里的产品手册是 PDF、规格表是 Excel、说明是 Word，
只能上传 txt 等于把「RAG 全流程」的第一步让给了手工预处理。

新增 `loaders.py`，统一解析 `txt / md / csv / pdf / docx / xlsx` 并返回「纯文本 + 元数据」：
解析错误有明确分类（`UnsupportedFileType` / `ParseError`），
扫描件（图片型 PDF）会明确提示需要 OCR，而不是静默入库一段空文本。

### 8. 若干工程细节

- `vector_stores.py` 的 `search_kwargs={"k": config.similarity_threshold}` 属于命名与语义
  不符的历史遗留（叫 threshold，实际当 k 用），已拆分为 `retrieval_top_k` / `retrieval_final_k`
  与两个真正的阈值；旧变量保留并标注废弃，避免影响既有脚本。
- `knowledge_base.upload_by_str()` 中所有 chunk 共用**同一个 metadata dict 对象**，
  已改为逐 chunk 独立副本——共享可变对象一旦被底层写入行为波及，会同时污染全部片段。
- `print_prompt` 默认打印完整 prompt，多轮对话时刷屏严重，改为由
  `debug_print_prompt`（默认 `False`）控制。
- 若知识库为空，`search_with_scores` 会返回空列表，门控会走
  `stage = "no_candidate"` 分支给出明确原因，而不是抛 IndexError。

---

## 🧩 项目结构

```text
KnowledgeBase-RAG-LLM-System/
├─ app_upload.py              # 知识库上传服务（Streamlit，支持六种文件格式）
├─ app_chat.py                # 智能客服问答（Streamlit，含检索过程 + 记忆状态侧边栏）
├─ knowledge_base.py          # 知识库处理：切分、写库、去重
├─ loaders.py                 # 多格式解析：txt/md/csv/pdf/docx/xlsx → 纯文本 + 元数据
├─ retrieval.py               # 检索链路：召回 → 重排 → 阈值门控（纯逻辑，零依赖可测）
├─ rag.py                     # RAG 链组装 + 摘要链路 + 拒答短路
├─ vector_stores.py           # 向量库检索封装（带分数召回，本地持久化）
├─ file_history_store.py      # 分层对话记忆（短期原文 + 长期摘要）
├─ config_data.py             # 模型、路径、chunk、检索链路、记忆分层等参数配置
├─ requirements.txt           # 项目依赖
├─ start.bat                  # 一键启动两个页面
├─ .env.example               # 环境变量模板
├─ tests/                     # 五个测试脚本（三个离线 + 两个联调）
│  ├─ test_memory_layering.py #   离线：分层记忆（桩摘要器，不消耗 API 额度）
│  ├─ test_retrieval_gate.py  #   离线：检索门控（零依赖，连 langchain 都不用装）
│  ├─ test_loaders.py         #   离线：多格式解析（缺可选依赖时按格式 SKIP）
│  ├─ test_memory_live.py     #   联调：12 轮真实对话验证记忆召回
│  └─ eval_retrieval.py       #   联调：检索效果评测（Hit@K / MRR / 拒答 / 判定间隔）
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

在 `config_data.py` 中修改：嵌入模型、对话模型、chunk 大小、检索链路参数等。
默认嵌入 `text-embedding-v4`、对话 `qwen3-max`、重排 `gte-rerank-v2`。

---

## 🔍 检索链路参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `retrieval_top_k` | `8` | 向量召回候选数，交给重排精排 |
| `retrieval_final_k` | `3` | 最终注入 prompt 的片段数 |
| `enable_rerank` | `True` | 是否启用交叉编码重排 |
| `rerank_model` | `gte-rerank-v2` | 重排模型（DashScope） |
| `rerank_threshold` | `0.05` | 重排分数阈值（启用重排时生效，实测两侧留 26× 裕度） |
| `vector_similarity_threshold` | `0.45` | 归一化相似度阈值（关闭重排时生效，实测裕度 1.26×） |
| `refusal_short_circuit` | `True` | 命中不足时是否直接拒答、不调用大模型 |
| `refusal_text` | 见文件 | 拒答文案（会告知用户知识库覆盖范围） |
| `debug_print_prompt` | `False` | 是否把完整 prompt 打到控制台（调试用） |

**调阈值的方法**：先跑 `python tests/eval_retrieval.py`，看输出的「判定间隔」——
该值越接近 1，说明库内/库外分数越难分开，阈值就越需要保守；
间隔足够大时，取两侧中间值即可。

> 注：把 `enable_rerank` 设为 `False` 即可退化为纯向量检索，
> 管线会自动改用 `vector_similarity_threshold` 判定，无需改代码。

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

五套测试，按「是否花钱」分成两组。**离线组不需要 API Key、不联网、不消耗任何额度**：

```bash
# ---- 离线组：随时可跑，不花钱 ----

# 1) 分层记忆：用桩摘要器替代模型
python tests/test_memory_layering.py        # 26 项断言

# 2) 检索门控：零依赖，连 langchain / chromadb 都不需要装
python tests/test_retrieval_gate.py         # 45 项断言

# 3) 多格式解析：PDF / Word / Excel 三项需可选依赖，缺失时按格式 SKIP
python tests/test_loaders.py                # 22 项断言


# ---- 联调组：需要 API Key，会消耗额度 ----

# 4) 分层记忆真实联调：跑 12 轮对话，验证跨压缩的记忆召回
python tests/test_memory_live.py            # 8 项断言

# 5) 检索效果评测：Hit@K / MRR / 拒答准确率 / 判定间隔，对比「加不加重排」
python tests/eval_retrieval.py
```

**分层记忆**（离线 26 项）覆盖：

- 短期窗口与压缩触发时机
- **注入 prompt 的历史存在上界**（25 轮对话峰值 12 条，不压缩则为 51 条）
- 兼容旧版「纯消息列表」历史文件并自动迁移
- 摘要失败时保留全部原文（不丢历史）
- `clear`、多会话隔离、`session_id` 路径字符过滤

**检索门控**（离线 45 项）覆盖：

- 距离 → 相似度的归一化与边界兜底（负值 / NaN / 非数值输入）
- 空查询、零召回、全部低于阈值三种拒答路径及其 `stage` 标记
- `top_k` 透传、`final_k` 截断、被阈值挡下的计数
- 重排改变排序，且**重排分数优先于向量相似度**（否则门控会拿错分数）
- **重排器抛异常时优雅降级**：退回向量顺序、记录原因、判定依据自动切回向量相似度
- 关闭重排 / `Noop` 占位重排时改判向量阈值

**多格式解析**（离线 22 项）覆盖：

- UTF-8 / GBK 编码兜底（只用 UTF-8 会在 GBK 文件上直接崩）
- Word 的**表格**必须被抽出（只读段落会整片丢表格，且不报错）
- Excel 逐工作表转文本（只读首个 sheet 会漏数据）
- 扫描件 PDF 抛出带 OCR 提示的 `ParseError`
- 不支持格式 / 空内容 / 损坏文件都收敛为明确异常，不裸抛底层错误

> `test_loaders.py` 在编写过程中**真的抓到过一个缺陷**：损坏的 docx 会裸抛
> `BadZipFile`，上传页面的 `except ParseError` 拦不住，用户会看到 traceback。
> 修复方式是把各解析器的底层异常统一收敛为 `ParseError`。

**检索效果评测**（联调）实测结论见上文「检索链路补齐」一节：
6 片段语料下 Hit@K 无差异，但**判定间隔从 1.26× 提升到 26.52×**。

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

### Q5：为什么问什么都回答「不在知识库范围内」？

说明阈值定得太保守。按顺序排查：

1. 展开回答下方的「🔍 检索过程」，看**候选片段数**——如果是 0，是向量库为空或路径不对（见 Q1）；
2. 看**最高相关度**距离阈值有多远。若普遍比阈值低一点点，把对应阈值调小
   （启用重排时调 `rerank_threshold`，关闭时调 `vector_similarity_threshold`）；
3. 跑 `python tests/eval_retrieval.py` 看**判定间隔**：若间隔接近 1，
   说明当前语料与问题确实难分，硬调阈值必然顾此失彼，需要先补语料。

### Q6：为什么明明配了重排，检索过程里却显示「重排已降级」？

看展开面板里的降级原因。常见三种：`dashscope` 未安装、API Key 缺失或错误、
网络不可达。降级不会中断问答——会自动退回向量召回顺序并改用向量相似度阈值判定，
只是排序质量与拒答裕度都会下降。

### Q7：上传 PDF 后提示「未能抽取到任何文字」？

这是**扫描件（图片型 PDF）**，里面只有图像没有文本层，需要先做 OCR。
上传页会额外提示哪些页面没抽到文字，方便定位是整份扫描件还是夹了几页图片。

### Q8：为什么拒答没有出现在对话历史里？

刻意如此。拒答内容既不含业务知识、也不承载上下文，写进历史只会污染后续的摘要压缩
（把有限的摘要字数浪费在「我不清楚」上）。因此门控拒答在调用大模型之前就短路了，
不会进入 `chat_history`。

旧版本的无扩展名历史文件（纯消息列表）会在首次读取时自动兼容迁移。

---

## 🚧 后续可改进方向

README 里列过的三条已经在本次改造中完成，留在这里做记录：

- ~~**Rerank 重排**~~ → 已实现（`gte-rerank-v2`，见 `retrieval.py`）
- ~~**检索阈值拒答**~~ → 已实现（双阈值门控 + 拒答短路，不出幻觉）
- ~~**支持更多文件类型**~~ → 已支持 `txt / md / csv / pdf / docx / xlsx`（见 `loaders.py`）

下一批候选，按投入产出排序：

1. **扩大评测集与语料**：当前只有 6 个片段、7 条库内问题，评测集区分度不足，
   重排的排序收益测不出来。补一批更接近真实客服的混淆问题（如「XL 码的洗涤说明」
   同时涉及尺码与洗涤），才能拉开 Hit@K 的差异。
2. **短期窗口按 token 数而非消息条数控制**：当前按条数（`memory_recent_messages`），
   但一条消息可能 5 字也可能 500 字，成本波动被低估。
3. **切分策略升级**：现阶段是定长字符切分，长文档会切断语义单元；
   可换成按标题层级 / 段落结构切分。
4. **结构化输出的引用溯源**：目前回答里没有标注「这句话来自哪个片段」，
   加上引用编号能显著提升可信度与可排查性。
5. **`RunnableWithMessageHistory` 已标记废弃**（langchain 0.3 起提示改用 LangGraph 的
   持久化机制），未来需要迁移；当前仍可正常工作，暂不动以免引入回归。

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
