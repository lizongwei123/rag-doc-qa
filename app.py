"""
RAG 智能文档问答系统 - Web界面
================================
基于 Streamlit 的交互式文档问答应用

运行方式:
    streamlit run app.py

首次使用:
    1. 注册 DeepSeek 获取 API Key: https://platform.deepseek.com
    2. 设置环境变量: export DEEPSEEK_API_KEY=sk-xxx
    3. 或直接在界面左侧输入 API Key

作者：李宗蔚 | 实习求职项目
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from rag_engine import RAGEngine, CHROMA_DIR

# ===================== 页面配置 =====================

st.set_page_config(
    page_title="RAG 智能文档问答",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================== 样式 =====================

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 24px;
    }
    .answer-box {
        background: linear-gradient(135deg, #1e293b, #273548);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        font-size: 1rem;
        line-height: 1.8;
    }
    .source-box {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .source-label {
        color: #10b981;
        font-weight: 600;
        font-size: 0.78rem;
        margin-bottom: 4px;
    }
    .stat-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .stat-num {
        font-size: 2rem;
        font-weight: 800;
        color: #818cf8;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 2px;
    }
    /* 聊天消息样式 */
    .chat-message {
        padding: 16px;
        border-radius: 12px;
        margin: 8px 0;
        line-height: 1.7;
    }
    .user-message {
        background: #1e293b;
        border: 1px solid #334155;
    }
    .ai-message {
        background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(124,58,237,0.1));
        border: 1px solid rgba(99,102,241,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ===================== 侧边栏 =====================

with st.sidebar:
    st.markdown("### ⚙️ 配置")

    # API Key
    api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        value=os.getenv("DEEPSEEK_API_KEY", ""),
        help="在 https://platform.deepseek.com 注册获取（新用户送免费额度）",
        placeholder="sk-xxxxxxxx",
    )
    if not api_key:
        st.warning("⚠️ 请先输入 API Key")
        st.markdown("[🔗 获取 DeepSeek API Key](https://platform.deepseek.com)")

    st.divider()

    # 文档上传
    st.markdown("### 📁 上传文档")
    uploaded_files = st.file_uploader(
        "支持 PDF / TXT / MD 格式",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="上传你要提问的文档，支持同时上传多个",
    )

    # 分片设置
    st.divider()
    st.markdown("### 🔧 高级设置")
    chunk_size = st.slider("文本分片大小", 200, 1000, 500, 50,
                           help="每个文本块的字符数。越小检索越精准，越大上下文越完整")
    chunk_overlap = st.slider("分片重叠大小", 0, 200, 80, 10,
                              help="相邻文本块的重叠字符数，防止语义在边界断裂")

    st.divider()

    # 处理按钮
    process_btn = st.button("🔨 构建知识库", type="primary", use_container_width=True)

    # 状态显示
    if "engine" in st.session_state and st.session_state.engine is not None:
        st.divider()
        st.markdown("### 📊 知识库状态")
        stats = st.session_state.engine.get_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("文档块数", stats["chunks"])
        with col2:
            st.metric("已处理文件", stats.get("processed_files", 0))

    st.divider()
    st.markdown("### 🔗 关于项目")
    st.markdown("""
    **技术栈**：
    - LangChain · ChromaDB
    - DeepSeek · Streamlit
    - DeepSeek Embedding

    **GitHub**: [项目地址]

    **作者**: 李宗蔚
    """)

# ===================== 主页面 =====================

st.markdown('<div class="main-header">🤖 RAG 智能文档问答</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">上传文档，基于AI进行智能问答——检索增强生成（RAG）系统实战</div>',
            unsafe_allow_html=True)

# ---------- 初始化Session State ----------

if "engine" not in st.session_state:
    st.session_state.engine = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "files_processed" not in st.session_state:
    st.session_state.files_processed = []

# ---------- 处理文档上传 ----------

if process_btn:
    if not api_key:
        st.error("❌ 请先在左侧输入 DeepSeek API Key！")
    elif not uploaded_files:
        st.warning("⚠️ 请先上传至少一个文档")
    else:
        with st.spinner("🔨 正在构建知识库..."):
            try:
                # 保存上传的文件到临时目录
                temp_dir = Path(tempfile.mkdtemp())
                saved_paths = []

                for uf in uploaded_files:
                    save_path = temp_dir / uf.name
                    with open(save_path, "wb") as f:
                        f.write(uf.getbuffer())
                    saved_paths.append(str(save_path))

                # 初始化引擎
                engine = RAGEngine(
                    api_key=api_key,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                # 先尝试加载已有索引
                engine.load_existing_index()

                # 构建新索引
                count = engine.build_index(saved_paths)

                # 保存到session
                st.session_state.engine = engine
                st.session_state.files_processed = [uf.name for uf in uploaded_files]
                st.session_state.chat_history = []

                st.success(f"✅ 知识库构建完成！共 {count} 个文本块，已处理 {len(uploaded_files)} 个文件")
                st.rerun()

            except Exception as e:
                st.error(f"❌ 构建失败: {str(e)}")

# ---------- 如果已有引擎，显示信息 ----------

if st.session_state.engine is not None and st.session_state.files_processed:
    stats = st.session_state.engine.get_stats()
    cols = st.columns(4)
    cols[0].markdown(f'<div class="stat-card"><div class="stat-num">{stats["chunks"]}</div><div class="stat-label">📊 文档块</div></div>', unsafe_allow_html=True)
    cols[1].markdown(f'<div class="stat-card"><div class="stat-num">{len(st.session_state.files_processed)}</div><div class="stat-label">📁 文件数</div></div>', unsafe_allow_html=True)
    cols[2].markdown(f'<div class="stat-card"><div class="stat-num">{chunk_size}</div><div class="stat-label">✂️ 分片大小</div></div>', unsafe_allow_html=True)
    cols[3].markdown(f'<div class="stat-card"><div class="stat-num">DeepSeek</div><div class="stat-label">🧠 LLM模型</div></div>', unsafe_allow_html=True)

# ---------- 欢迎页/空状态 ----------

if st.session_state.engine is None:
    st.info("""
    ### 👋 欢迎使用 RAG 智能文档问答系统！

    **三步开始使用：**
    1. 🔑 在左侧输入 DeepSeek API Key（[免费获取](https://platform.deepseek.com)）
    2. 📁 上传你的文档（PDF / TXT / MD）
    3. 🔨 点击「构建知识库」按钮
    4. 💬 在下方输入问题，开始AI问答

    ---
    ### 🤔 什么是RAG？

    **Retrieval-Augmented Generation（检索增强生成）** 是大模型应用开发的核心技术。

    ```
    你的文档 → 文本分片 → 向量化存储
                                ↓
    你的问题 → 向量检索 → 找到相关片段 → LLM生成回答
    ```

    **为什么需要RAG？**
    - 📚 LLM的知识有截止日期，无法回答你的私有文档内容
    - 🎯 RAG让LLM「先查资料再回答」，大幅减少幻觉
    - 💰 无需微调模型，成本极低，效果显著
    """)

# ---------- 聊天区域 ----------

if st.session_state.engine is not None:
    st.divider()
    st.markdown("### 💬 对话")

    # 显示历史消息
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-message user-message">🙋 <strong>你：</strong>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-message ai-message">🤖 <strong>AI：</strong>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            if msg.get("sources"):
                with st.expander("📚 参考来源"):
                    for i, (src, txt) in enumerate(zip(msg["sources"], msg.get("source_texts", [])), 1):
                        st.markdown(
                            f'<div class="source-box"><span class="source-label">来源 {i}：{src}</span><br>{txt}</div>',
                            unsafe_allow_html=True,
                        )

    # 输入框
    st.divider()
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        question = st.text_input(
            "输入你的问题",
            placeholder="例如：这份文档的主要内容是什么？请总结一下...",
            label_visibility="collapsed",
            key="question_input",
        )
    with col_btn:
        ask_btn = st.button("🚀 发送", use_container_width=True, type="primary")

    if ask_btn and question.strip():
        with st.spinner("🤔 AI思考中..."):
            try:
                result = st.session_state.engine.ask(question.strip())

                # 保存到历史
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question.strip(),
                })
                st.session_state.chat_history.append({
                    "role": "ai",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "source_texts": result["source_texts"],
                })

                st.rerun()

            except Exception as e:
                st.error(f"❌ 问答出错: {str(e)}")

    # 清空对话按钮
    if st.session_state.chat_history:
        if st.button("🗑️ 清空对话", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()

# ===================== 底部说明 =====================

st.divider()
st.markdown(
    """
    <div style="text-align:center;color:#64748b;font-size:0.82rem;padding:16px;">
    🛠️ 技术栈：LangChain + ChromaDB + DeepSeek + Streamlit |
    📧 联系方式：2673369774@qq.com |
    © 2026 李宗蔚 · 实习求职项目
    </div>
    """,
    unsafe_allow_html=True,
)
