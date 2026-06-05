"""
RAG (Retrieval-Augmented Generation) 文档问答引擎
==================================================
核心功能：加载文档 → 文本分片 → 向量化存储 → 检索 → LLM生成回答

技术栈：LangChain + ChromaDB + DeepSeek API
作者：李宗蔚 | 项目用于实习求职
"""

import os
import hashlib
from typing import List, Optional, Dict, Any
from pathlib import Path

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# ===================== 配置 =====================

# 使用本地embedding模型（免费，无需API key）
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"

# LLM配置 - 使用DeepSeek（国内可直接访问，价格极低）
# 注册获取API Key: https://platform.deepseek.com
LLM_CONFIG = {
    "model": "deepseek-chat",
    "temperature": 0.3,
    "max_tokens": 2048,
}

# 文本分片配置
CHUNK_SIZE = 500       # 每个分片的字符数
CHUNK_OVERLAP = 80     # 分片之间的重叠字符数（保持上下文连贯）

# 向量数据库路径
CHROMA_DIR = "./chroma_db"

# ===================== 中文优化Prompt =====================

CHINESE_PROMPT = PromptTemplate(
    template="""你是一个专业的文档问答助手。请根据以下文档内容回答用户的问题。

要求：
1. 只根据提供的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，请明确说「根据提供的文档，未找到相关信息」
3. 回答要准确、简洁、有条理
4. 如果涉及技术细节，请引用文档中的原文支持你的答案

文档内容：
{context}

用户问题：{question}

回答：""",
    input_variables=["context", "question"],
)


class RAGEngine:
    """RAG文档问答引擎"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        """
        初始化RAG引擎

        Args:
            api_key: DeepSeek API Key（不提供则从环境变量 DEEPSEEK_API_KEY 读取）
            api_base: API地址（默认DeepSeek官方地址）
            chunk_size: 文本分片大小
            chunk_overlap: 分片重叠大小
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = api_base or "https://api.deepseek.com"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 延迟初始化，节省启动时间
        self._embeddings = None
        self._llm = None
        self._vectorstore: Optional[Chroma] = None
        self._qa_chain = None

        # 记录已处理的文件
        self._processed_files: Dict[str, str] = {}  # path -> hash

    # ---------- 属性（延迟加载） ----------

    @property
    def embeddings(self):
        """本地中文embedding模型（首次使用时自动下载）"""
        if self._embeddings is None:
            print("⏳ 加载embedding模型...")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            print("✅ Embedding模型加载完成")
        return self._embeddings

    @property
    def llm(self):
        """LLM实例"""
        if self._llm is None:
            if not self.api_key:
                raise ValueError(
                    "请设置 DeepSeek API Key！\n"
                    "方法1: 设置环境变量 export DEEPSEEK_API_KEY=your_key\n"
                    "方法2: 创建 .env 文件写入 DEEPSEEK_API_KEY=your_key\n"
                    "获取免费额度: https://platform.deepseek.com"
                )
            self._llm = ChatOpenAI(
                model=LLM_CONFIG["model"],
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"],
                api_key=self.api_key,
                base_url=self.api_base,
            )
        return self._llm

    # ---------- 文档处理 ----------

    def load_document(self, file_path: str) -> List:
        """
        加载文档（支持 PDF / TXT / MD 格式）

        Args:
            file_path: 文件路径

        Returns:
            加载后的文档列表
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()

        if ext == ".pdf":
            loader = PyPDFLoader(str(path))
        elif ext == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        elif ext in [".md", ".markdown"]:
            loader = UnstructuredMarkdownLoader(str(path))
        else:
            raise ValueError(f"暂不支持的格式: {ext}（支持 PDF、TXT、MD）")

        documents = loader.load()
        print(f"📄 加载文档: {path.name} ({len(documents)} 页/段)")

        # 添加来源标记
        for doc in documents:
            doc.metadata["source"] = path.name

        return documents

    def split_documents(self, documents: List) -> List:
        """
        将文档切分为固定大小的文本块

        为什么需要分片？
        - LLM的上下文窗口有限，不能一次塞入整本书
        - 小块文本检索更精准
        - 重叠设计保证上下文不会在边界断裂
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
        print(f"✂️  分片完成: {len(documents)} → {len(chunks)} 个文本块")
        return chunks

    def build_index(self, file_paths: List[str], force_rebuild: bool = False) -> int:
        """
        构建向量索引

        Args:
            file_paths: 文档路径列表
            force_rebuild: 是否强制重建索引

        Returns:
            索引中的文档块数量
        """
        all_chunks = []

        for fp in file_paths:
            # 检查文件是否已处理（通过MD5判断内容是否变化）
            file_hash = self._compute_hash(fp)
            if not force_rebuild and fp in self._processed_files:
                if self._processed_files[fp] == file_hash:
                    print(f"⏭️  跳过（未变化）: {Path(fp).name}")
                    continue

            documents = self.load_document(fp)
            chunks = self.split_documents(documents)
            all_chunks.extend(chunks)
            self._processed_files[fp] = file_hash

        if not all_chunks:
            print("⚠️  没有新文档需要处理")
            return self._vectorstore._collection.count() if self._vectorstore else 0

        # 构建向量数据库
        print(f"🔨 构建向量索引 ({len(all_chunks)} 个文本块)...")
        self._vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory=CHROMA_DIR,
        )
        # 确保数据持久化到磁盘
        self._vectorstore.persist()

        # 重置QA链（因为知识库更新了）
        self._qa_chain = None

        count = self._vectorstore._collection.count()
        print(f"✅ 向量索引构建完成！共 {count} 个文本块")
        return count

    def load_existing_index(self) -> bool:
        """加载已有的向量索引"""
        if not os.path.exists(CHROMA_DIR):
            return False

        try:
            self._vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=self.embeddings,
            )
            count = self._vectorstore._collection.count()
            print(f"📂 加载已有索引: {count} 个文本块")
            return count > 0
        except Exception as e:
            print(f"⚠️  加载索引失败: {e}")
            return False

    # ---------- 问答 ----------

    def _get_qa_chain(self):
        """获取QA链（延迟创建）"""
        if self._qa_chain is None:
            if self._vectorstore is None:
                raise ValueError("请先构建或加载向量索引！")

            retriever = self._vectorstore.as_retriever(
                search_kwargs={"k": 4}  # 返回最相关的4个文档块
            )

            self._qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": CHINESE_PROMPT},
            )

        return self._qa_chain

    def ask(self, question: str) -> Dict[str, Any]:
        """
        向文档提问

        Args:
            question: 用户问题

        Returns:
            {
                "question": 问题,
                "answer": 回答,
                "sources": [来源文档列表],
                "source_texts": [相关原文片段]
            }
        """
        qa = self._get_qa_chain()
        result = qa.invoke({"query": question})

        # 提取来源信息（去重）
        seen = set()
        sources = []
        source_texts = []
        for doc in result.get("source_documents", []):
            content = doc.page_content[:200] + "..."
            if content not in seen:
                seen.add(content)
                sources.append(doc.metadata.get("source", "未知"))
                source_texts.append(content)

        return {
            "question": question,
            "answer": result["result"],
            "sources": sources,
            "source_texts": source_texts,
        }

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """仅检索相关文档，不生成回答"""
        if self._vectorstore is None:
            return []

        docs = self._vectorstore.similarity_search_with_score(query, k=top_k)

        results = []
        for doc, score in docs:
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "未知"),
                "relevance": round(float(score), 4),  # 分数越低越相关
            })
        return results

    # ---------- 工具方法 ----------

    @staticmethod
    def _compute_hash(filepath: str) -> str:
        """计算文件MD5"""
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_stats(self) -> Dict:
        """获取索引统计信息"""
        if self._vectorstore is None:
            return {"status": "未初始化", "chunks": 0}

        return {
            "status": "已就绪",
            "chunks": self._vectorstore._collection.count(),
            "processed_files": len(self._processed_files),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


# ===================== 快捷函数 =====================

def create_engine(
    api_key: Optional[str] = None,
    files: Optional[List[str]] = None,
) -> RAGEngine:
    """
    快速创建并初始化RAG引擎

    使用示例:
        engine = create_engine(
            api_key="sk-xxx",
            files=["doc1.pdf", "doc2.txt"]
        )
        result = engine.ask("这份文档的主要内容是什么？")
    """
    engine = RAGEngine(api_key=api_key)

    if files:
        # 先尝试加载已有索引
        engine.load_existing_index()
        # 处理新文件
        engine.build_index(files)
    else:
        engine.load_existing_index()

    return engine


if __name__ == "__main__":
    # 命令行测试
    import sys

    if len(sys.argv) < 2:
        print("用法: python rag_engine.py <文档路径> [文档路径2 ...]")
        print("示例: python rag_engine.py resume.pdf")
        sys.exit(1)

    engine = create_engine(files=sys.argv[1:])

    print("\n" + "=" * 60)
    print("🤖 RAG文档问答系统已就绪！输入问题开始对话，输入 'quit' 退出")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("❓ 你的问题: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                print("👋 再见！")
                break
            if not question:
                continue

            result = engine.ask(question)
            print(f"\n🤖 回答: {result['answer']}")
            print(f"📚 参考来源: {', '.join(result['sources'])}\n")

        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")
