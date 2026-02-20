"""
LangChain CBT RAG Demo (148 JSON files)

Usage:
  python langchaindemo.py
    python langchaindemo.py --question "我总是担心失败，怎么做认知重评？"
    python langchaindemo.py --question "我总是担心失败" --top-k 8
    python langchaindemo.py --question "我总是担心失败" --doc-type special_situation
    python langchaindemo.py --question "考研焦虑怎么做" --doc-type session
        python langchaindemo.py --question "职业规划焦虑" --doc-type global_plan --top-k 5

This script runs retrieval only (no LLM generation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
CBT_DATA_DIR = PROJECT_ROOT / "data" / "cbt"
DEFAULT_TOP_K = 6


class Document:
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata


def _ensure_runtime_compatibility() -> None:
    if sys.version_info >= (3, 14):
        pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(
            "当前 Python 版本为 "
            f"{pyver}，LangChain 在该版本上存在兼容性问题。"
            "请切换到 Python 3.10-3.13 后再运行本脚本。"
        )





def _import_langchain_dependencies() -> Dict[str, Any]:
    _ensure_runtime_compatibility()

    from langchain_core.documents import Document as LCDocument
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.retrievers import TFIDFRetriever
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return {
        "Document": LCDocument,
        "BM25Retriever": BM25Retriever,
        "TFIDFRetriever": TFIDFRetriever,
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
    }


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "；".join(_safe_text(item) for item in value if _safe_text(item))
    if isinstance(value, dict):
        parts: List[str] = []
        for key, sub in value.items():
            sub_text = _safe_text(sub)
            if sub_text:
                parts.append(f"{key}: {sub_text}")
        return "；".join(parts)
    return str(value)


def _iter_cbt_json_files(data_dir: Path, max_files: int = 148) -> List[Path]:
    files = [p for p in data_dir.glob("*.json") if p.name[:-5].isdigit()]
    files.sort(key=lambda p: int(p.stem))
    selected = files[:max_files]
    if len(selected) < max_files:
        raise RuntimeError(
            f"需要至少 {max_files} 个 CBT JSON 文件，但只找到 {len(selected)} 个: {data_dir}"
        )
    return selected


def _build_case_documents(case_data: Dict[str, Any], file_name: str) -> List[Document]:
    docs: List[Document] = []
    client_id = case_data.get("client_id", "unknown")
    client_info = case_data.get("client_info", {})

    profile_text = (
        f"案例ID: {client_id}\n"
        f"主题: {_safe_text(client_info.get('topic'))}\n"
        f"主要问题: {_safe_text(client_info.get('main_problem'))}\n"
        f"核心诉求: {_safe_text(client_info.get('core_demands'))}\n"
        f"核心信念: {_safe_text(client_info.get('core_beliefs'))}\n"
        f"静态特征: {_safe_text(client_info.get('static_traits'))}\n"
        f"成长经历: {_safe_text(client_info.get('growth_experiences'))}\n"
    )
    docs.append(
        Document(
            page_content=profile_text,
            metadata={
                "source": file_name,
                "client_id": client_id,
                "doc_type": "client_profile",
                "topic": _safe_text(client_info.get("topic")),
            },
        )
    )

    for idx, item in enumerate(client_info.get("special_situations", []), start=1):
        situation_text = (
            f"案例ID: {client_id}\n"
            f"认知情境#{idx}\n"
            f"事件: {_safe_text(item.get('event'))}\n"
            f"条件性假设: {_safe_text(item.get('conditional_assumptions'))}\n"
            f"自动化思维: {_safe_text(item.get('automatic_thoughts'))}\n"
            f"认知模式: {_safe_text(item.get('cognitive_pattern'))}\n"
            f"补偿策略: {_safe_text(item.get('compensatory_strategies'))}\n"
        )
        docs.append(
            Document(
                page_content=situation_text,
                metadata={
                    "source": file_name,
                    "client_id": client_id,
                    "doc_type": "special_situation",
                    "topic": _safe_text(client_info.get("topic")),
                    "cognitive_pattern": _safe_text(item.get("cognitive_pattern")),
                },
            )
        )

    for stage in case_data.get("global_plan", []):
        stage_name = _safe_text(stage.get("stage_name"))
        sessions_desc = _safe_text(stage.get("sessions"))
        stage_content = _safe_text(stage.get("content"))
        plan_text = (
            f"案例ID: {client_id}\n"
            f"治疗计划阶段: {stage_name}\n"
            f"会谈范围: {sessions_desc}\n"
            f"阶段内容: {stage_content}\n"
        )
        docs.append(
            Document(
                page_content=plan_text,
                metadata={
                    "source": file_name,
                    "client_id": client_id,
                    "doc_type": "global_plan",
                    "topic": _safe_text(client_info.get("topic")),
                    "stage_name": stage_name,
                },
            )
        )

    for session in case_data.get("sessions", []):
        session_no = session.get("session_number", "unknown")
        goals = _safe_text(session.get("session_goals"))
        dialogue = _safe_text(session.get("dialogue"))
        session_summary = _safe_text(session.get("session_summary"))
        session_text = (
            f"案例ID: {client_id}\n"
            f"会谈编号: {session_no}\n"
            f"会谈目标: {goals}\n"
            f"关键对话: {dialogue}\n"
            f"会谈总结: {session_summary}\n"
        )
        docs.append(
            Document(
                page_content=session_text,
                metadata={
                    "source": file_name,
                    "client_id": client_id,
                    "doc_type": "session",
                    "topic": _safe_text(client_info.get("topic")),
                    "session_number": str(session_no),
                },
            )
        )

    return docs


def load_cbt_documents(data_dir: Path = CBT_DATA_DIR, max_files: int = 148) -> List[Document]:
    all_docs: List[Document] = []
    for file_path in _iter_cbt_json_files(data_dir, max_files=max_files):
        with open(file_path, "r", encoding="utf-8") as f:
            case_data = json.load(f)
        all_docs.extend(_build_case_documents(case_data, file_path.name))
    return all_docs


def _split_documents(documents: List[Document]) -> List[Document]:
    deps = _import_langchain_dependencies()
    lc_document_cls = deps["Document"]

    lc_documents = [
        lc_document_cls(page_content=doc.page_content, metadata=doc.metadata)
        for doc in documents
    ]

    splitter = deps["RecursiveCharacterTextSplitter"](
        chunk_size=1200,
        chunk_overlap=180,
        separators=["\n\n", "\n", "。", "；", "，", " "],
    )
    return splitter.split_documents(lc_documents)


def _format_chat_history(history: Iterable[Tuple[str, str]]) -> str:
    lines: List[str] = []
    for role, content in history:
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


class LangChainTherapyRAGRetriever:
    def __init__(self, top_k: int = DEFAULT_TOP_K):
        deps = _import_langchain_dependencies()
        bm25_retriever_cls = deps["BM25Retriever"]
        tfidf_retriever_cls = deps["TFIDFRetriever"]

        self.top_k = top_k
        self.retrieval_mode = "hybrid"

        docs = load_cbt_documents(CBT_DATA_DIR, max_files=148)
        chunks = _split_documents(docs)
        self.bm25_retriever = bm25_retriever_cls.from_documents(chunks)
        self.bm25_retriever.k = max(top_k * 5, 10)

        self.tfidf_retriever = tfidf_retriever_cls.from_documents(chunks)
        self.tfidf_retriever.k = max(top_k * 5, 10)

    @staticmethod
    def _filter_by_doc_type(documents: List[Any], doc_type: str) -> List[Any]:
        if doc_type == "all":
            return documents
        return [doc for doc in documents if doc.metadata.get("doc_type") == doc_type]

    @staticmethod
    def _dedupe_documents(documents: List[Any], max_k: int) -> List[Any]:
        seen = set()
        unique_docs: List[Any] = []
        for doc in documents:
            key = (
                doc.metadata.get("source", ""),
                str(doc.metadata.get("client_id", "")),
                doc.metadata.get("doc_type", ""),
                doc.page_content[:160],
            )
            if key in seen:
                continue
            seen.add(key)
            unique_docs.append(doc)
            if len(unique_docs) >= max_k:
                break
        return unique_docs

    def _retrieve_docs(self, query: str) -> List[Any]:
        bm25_docs = self.bm25_retriever.invoke(query)
        tfidf_docs = self.tfidf_retriever.invoke(query)

        merged: List[Any] = []
        max_len = max(len(bm25_docs), len(tfidf_docs))
        for idx in range(max_len):
            if idx < len(bm25_docs):
                merged.append(bm25_docs[idx])
            if idx < len(tfidf_docs):
                merged.append(tfidf_docs[idx])

        return self._dedupe_documents(merged, self.top_k)

    def retrieve(
        self,
        user_input: str,
        chat_history: List[Tuple[str, str]],
        doc_type_filter: str = "all",
    ) -> Dict[str, Any]:
        history_text = _format_chat_history(chat_history)
        query = user_input if not history_text else f"{history_text}\n当前问题: {user_input}"
        context_docs = self._retrieve_docs(query)
        context_docs = self._filter_by_doc_type(context_docs, doc_type_filter)

        references = []
        for doc in context_docs[: self.top_k]:
            references.append(
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "client_id": doc.metadata.get("client_id", "unknown"),
                    "doc_type": doc.metadata.get("doc_type", "unknown"),
                    "topic": doc.metadata.get("topic", ""),
                    "content": doc.page_content[:300],
                }
            )

        return {
            "mode": self.retrieval_mode,
            "doc_type_filter": doc_type_filter,
            "references": references,
        }


def run_cli(question: str | None, top_k: int, doc_type: str) -> None:

    print(
        f"\n初始化 LangChain CBT RAG 检索器（仅检索，不调用LLM，"
        f"mode=hybrid，doc_type={doc_type}）..."
    )
    retriever = LangChainTherapyRAGRetriever(top_k=top_k)
    print("初始化完成。已加载 data/cbt 前 148 个 JSON。\n")

    if question:
        result = retriever.retrieve(question, chat_history=[], doc_type_filter=doc_type)
        print(f"检索结果（mode={result['mode']}，doc_type={result['doc_type_filter']}）:")
        if not result["references"]:
            print("- 未命中结果，请尝试放宽 doc_type 或调整问题关键词。")
            return
        for ref in result["references"]:
            print(f"- {ref['source']} | case={ref['client_id']} | type={ref['doc_type']} | topic={ref['topic']}")
            print(f"  内容片段: {ref['content']}")
        return

    print("进入检索模式（输入 /exit 退出）")
    history: List[Tuple[str, str]] = []
    while True:
        user_input = input("\n来访者: ").strip()
        if not user_input:
            continue
        if user_input in {"/exit", "exit", "quit"}:
            print("\n会话结束。")
            break

        result = retriever.retrieve(user_input, chat_history=history, doc_type_filter=doc_type)
        print(f"\n检索结果（mode={result['mode']}，doc_type={result['doc_type_filter']}）:")
        if result["references"]:
            for ref in result["references"]:
                print(
                    f"- {ref['source']} | case={ref['client_id']} | "
                    f"type={ref['doc_type']} | topic={ref['topic']}"
                )
                print(f"  内容片段: {ref['content']}")
        else:
            print("- 未命中结果，请尝试放宽 doc_type 或调整问题关键词。")

        history.append(("user", user_input))
        history.append(("retriever", json.dumps(result["references"], ensure_ascii=False)))


def parse_args() -> argparse.Namespace:
    # --question:
    # 单轮提问文本；传入后执行一次检索并退出。
    # 不传时进入交互检索模式（可连续输入）。
    parser = argparse.ArgumentParser(description="CBT 148 JSON LangChain RAG Retriever (No LLM)")
    parser.add_argument("--question", type=str, default=None, help="单轮提问")

    # --top-k:
    # 最终返回给用户的结果数量上限。
    # 值越大，召回更广但可能噪声更多。
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="检索返回文档数量")

    # 检索模式固定为 hybrid（BM25 + TF-IDF 交错合并后去重）。

    # --doc-type:
    # 检索后结果的文档类型过滤器。
    # all 不过滤；其余仅返回指定类型。
    # 可选：client_profile / special_situation / global_plan / session。
    parser.add_argument(
        "--doc-type",
        type=str,
        default="all",
        choices=["all", "client_profile", "special_situation", "global_plan", "session"],
        help="文档类型过滤：all（默认）或指定类型",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cli(
        question=args.question,
        top_k=args.top_k,
        doc_type=args.doc_type,
    )