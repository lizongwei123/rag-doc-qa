"""
RAG (Retrieval-Augmented Generation) 文档问答引擎
==================================================
核心功能：加载文档 → 文本分片 → 向量化存储 → 检索 → LLM生成回答

技术栈：LangChain + ChromaDB + DeepSeek API (LLM + Embedding)
作者：李宗蔚 | 项目用于实习求职
"""

import os
import hashlib
from typing import List, Optional, Dict, Any
from pathlib import Path

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# ===================== Configuration =====================

# DeepSeek API base URL
DEEPSEEK_BASE = "https://api.deepseek.com"

# Local Chinese embedding model (pre-downloaded via ModelScope)
# If you need to re-download: python -c "from modelscope import snapshot_download; snapshot_download('iic/nlp_corom_sentence-embedding_chinese-base')"
EMBEDDING_MODEL = r"C:\Users\asus\.cache\modelscope\hub\models\iic\nlp_corom_sentence-embedding_chinese-base"

# LLM 配置
LLM_CONFIG = {
    "model": "deepseek-chat",
    "temperature": 0.3,
    "max_tokens": 2048,
}

# 文本分片配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

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
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = api_base or DEEPSEEK_BASE
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._embeddings = None
        self._llm = None
        self._vectorstore: Optional[Chroma] = None
        self._qa_chain = None
        self._processed_files: Dict[str, str] = {}

    # ---------- Lazy Properties ----------

    @property
    def embeddings(self):
        """Local Chinese embedding model (downloaded via HF mirror on first use)"""
        if self._embeddings is None:
            print("[EMBED] Loading embedding model (first time may take a while)...")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            print("[EMBED] Embedding model ready")
        return self._embeddings

    @property
    def llm(self):
        """DeepSeek Chat LLM"""
        if self._llm is None:
            if not self.api_key:
                raise ValueError(
                    "Please set DEEPSEEK_API_KEY.\n"
                    "Get one at: https://platform.deepseek.com"
                )
            self._llm = ChatOpenAI(
                model=LLM_CONFIG["model"],
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"],
                api_key=self.api_key,
                base_url=self.api_base,
            )
        return self._llm

    # ---------- Document Processing ----------

    def load_document(self, file_path: str) -> List:
        """Load document (PDF / TXT / MD)"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(str(path))
        elif ext in [".txt", ".md", ".markdown"]:
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported format: {ext} (PDF/TXT/MD only)")

        documents = loader.load()
        print(f"[DOC] Loaded: {path.name} ({len(documents)} pages/sections)")

        for doc in documents:
            doc.metadata["source"] = path.name

        return documents

    def split_documents(self, documents: List) -> List:
        """Split documents into chunks for retrieval"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""],
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
        print(f"[SPLIT] {len(documents)} docs -> {len(chunks)} chunks")
        return chunks

    def build_index(self, file_paths: List[str], force_rebuild: bool = False) -> int:
        """Build vector index from documents"""
        all_chunks = []

        for fp in file_paths:
            file_hash = self._compute_hash(fp)
            if not force_rebuild and fp in self._processed_files:
                if self._processed_files[fp] == file_hash:
                    print(f"[SKIP] Unchanged: {Path(fp).name}")
                    continue

            documents = self.load_document(fp)
            chunks = self.split_documents(documents)
            all_chunks.extend(chunks)
            self._processed_files[fp] = file_hash

        if not all_chunks:
            print("[WARN] No new documents to process")
            return self._vectorstore._collection.count() if self._vectorstore else 0

        print(f"[BUILD] Creating vector index ({len(all_chunks)} chunks)...")
        self._vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory=CHROMA_DIR,
        )
        self._vectorstore.persist()
        self._qa_chain = None

        count = self._vectorstore._collection.count()
        print(f"[OK] Index built: {count} chunks")
        return count

    def load_existing_index(self) -> bool:
        """Load existing vector index from disk"""
        if not os.path.exists(CHROMA_DIR):
            return False

        try:
            self._vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=self.embeddings,
            )
            count = self._vectorstore._collection.count()
            print(f"[LOAD] Existing index: {count} chunks")
            return count > 0
        except Exception as e:
            print(f"[WARN] Failed to load index: {e}")
            return False

    # ---------- Q&A ----------

    def _get_qa_chain(self):
        if self._qa_chain is None:
            if self._vectorstore is None:
                raise ValueError("Please build or load vector index first!")

            retriever = self._vectorstore.as_retriever(
                search_kwargs={"k": 4}
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
        """Ask a question against the document knowledge base"""
        qa = self._get_qa_chain()
        result = qa.invoke({"query": question})

        seen = set()
        sources = []
        source_texts = []
        for doc in result.get("source_documents", []):
            content = doc.page_content[:200] + "..."
            if content not in seen:
                seen.add(content)
                sources.append(doc.metadata.get("source", "unknown"))
                source_texts.append(content)

        return {
            "question": question,
            "answer": result["result"],
            "sources": sources,
            "source_texts": source_texts,
        }

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search documents without LLM generation"""
        if self._vectorstore is None:
            return []

        docs = self._vectorstore.similarity_search_with_score(query, k=top_k)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "relevance": round(float(score), 4),
            }
            for doc, score in docs
        ]

    # ---------- Utilities ----------

    @staticmethod
    def _compute_hash(filepath: str) -> str:
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_stats(self) -> Dict:
        if self._vectorstore is None:
            return {"status": "not initialized", "chunks": 0}
        return {
            "status": "ready",
            "chunks": self._vectorstore._collection.count(),
            "processed_files": len(self._processed_files),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


# ===================== Quick Start =====================

def create_engine(
    api_key: Optional[str] = None,
    files: Optional[List[str]] = None,
) -> RAGEngine:
    """Create and initialize a RAG engine"""
    engine = RAGEngine(api_key=api_key)
    if files:
        engine.load_existing_index()
        engine.build_index(files)
    else:
        engine.load_existing_index()
    return engine


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag_engine.py <file1> [file2 ...]")
        print("Example: python rag_engine.py document.pdf")
        sys.exit(1)

    engine = create_engine(files=sys.argv[1:])

    print("\n" + "=" * 60)
    print("RAG System Ready! Type 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("Q: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
            if not question:
                continue

            result = engine.ask(question)
            print(f"\nA: {result['answer']}")
            print(f"Sources: {', '.join(result['sources'])}\n")

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
