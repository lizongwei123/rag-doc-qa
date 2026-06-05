# 🤖 RAG 智能文档问答系统

> **基于 LangChain + ChromaDB + DeepSeek 的检索增强生成（RAG）文档问答系统**
>
> 上传 PDF/TXT 文档 → 自动分片向量化 → 用自然语言提问 → AI 基于文档内容精准回答

---

## 🎯 项目简介

这是我在学习大模型应用开发过程中的实战项目，完整实现了 RAG（Retrieval-Augmented Generation）的核心流程：

```
📄 文档上传 → ✂️ 文本分片 → 🧮 向量化 → 💾 ChromaDB存储
                                              ↓
❓ 用户提问 → 🔍 向量检索 → 📎 召回相关片段 → 🧠 LLM生成回答
```

**为什么需要 RAG？**
- 大模型的知识有截止日期，无法回答你的私有文档内容
- RAG 让 LLM「先查资料再回答」，大幅减少幻觉
- 无需微调模型，成本极低，效果显著

---

## 🛠️ 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| **LLM** | DeepSeek-Chat | 国产大模型，API价格极低，国内可直接访问 |
| **Embedding** | text2vec-base-chinese | 中文语义向量模型，本地运行，免费 |
| **向量数据库** | ChromaDB | 轻量级，嵌入式，无需单独部署 |
| **框架** | LangChain | RAG链路编排 |
| **界面** | Streamlit | Python Web UI，快速搭建 |
| **文档解析** | PyPDF / Unstructured | 支持 PDF、TXT、Markdown |

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 获取 API Key

注册 DeepSeek 获取 API Key（新用户送免费额度）：
👉 [https://platform.deepseek.com](https://platform.deepseek.com)

### 3. 设置环境变量

```bash
# Linux / Mac
export DEEPSEEK_API_KEY=sk-your-key

# Windows PowerShell
$env:DEEPSEEK_API_KEY="sk-your-key"

# Windows CMD
set DEEPSEEK_API_KEY=sk-your-key
```

### 4. 启动 Web 界面

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`，开始使用！

---

## 📖 使用方式

### Web 界面（推荐）

1. 左侧输入 API Key
2. 上传你的文档（PDF/TXT/MD）
3. 点击「构建知识库」
4. 在对话框输入问题，AI 基于文档回答

### 命令行

```bash
# 直接对文档提问
python rag_engine.py document.pdf

# 支持多个文档
python rag_engine.py doc1.pdf doc2.txt doc3.md
```

### Python 代码调用

```python
from rag_engine import create_engine

# 初始化引擎
engine = create_engine(
    api_key="sk-xxx",
    files=["resume.pdf", "notes.txt"]
)

# 提问
result = engine.ask("这份文档的主要内容是什么？")
print(result["answer"])
print(f"参考来源: {result['sources']}")
```

---

## 📁 项目结构

```
rag-qa/
├── app.py              # Streamlit Web 界面
├── rag_engine.py       # RAG 核心引擎
├── requirements.txt    # Python 依赖
├── .env.example        # 环境变量模板
├── README.md           # 项目文档
└── chroma_db/          # 向量数据库文件（自动生成）
```

---

## 🔬 技术亮点

### 1. 中文优化
- 使用 `text2vec-base-chinese` 中文 embedding 模型，语义理解准确
- 自定义中文 Prompt 模板，减少英文直译的生硬感
- 针对中文标点优化的分片策略

### 2. 分片策略
- 递归字符分割，保持语义完整性
- 分片重叠（overlap）设计，防止上下文在边界断裂
- 可根据文档类型调整 `chunk_size` 和 `chunk_overlap`

### 3. 工程实践
- 文档 MD5 去重，避免重复处理
- 向量索引持久化，重启不丢失
- 延迟初始化，减少启动时间
- 完整的错误处理和日志

---

## 📊 RAG vs 直接问LLM

| 场景 | 直接问 ChatGPT | 使用 RAG |
|------|--------------|---------|
| 问最新信息 | ❌ 知识截止 | ✅ 基于文档 |
| 问公司内部文档 | ❌ 无法回答 | ✅ 精准回答 |
| 回答可信度 | ⚠️ 可能幻觉 | ✅ 有据可查 |
| 隐私安全 | ❌ 数据上传 | ✅ 本地处理 |

---

## 🔜 后续计划

- [ ] 支持更多文档格式（Word、Excel、HTML）
- [ ] 添加多轮对话记忆
- [ ] 支持切换不同 LLM（通义千问、GLM等）
- [ ] 添加引用高亮显示
- [ ] Docker 一键部署

---

## 👤 作者

**李宗蔚** · 人工智能专业在读

- 📧 2673369774@qq.com
- 🔗 GitHub: https://github.com/lizongwei123/rag-doc-qa

> 本项目为实习求职项目，欢迎 Star 和交流！
