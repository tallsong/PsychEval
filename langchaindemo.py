"""
LangChain Therapy Technique RAG (Agent Tool Schema)

Usage:
  python langchaindemo.py \
    --question "我总是担心失败，怎么做认知重评？" \
    --conversation-summary "近一周因考试反复焦虑、睡眠变差" \
    --focus-tags 焦虑,认知重评 \
    --stage-hint intervention \
    --top-k 4

This script runs retrieval only (no LLM generation).
Output is JSON designed for agent tool consumption.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
CBT_DATA_DIR = PROJECT_ROOT / "data" / "cbt"
DEFAULT_TOP_K = 4


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
        parts: List[str] = []
        for item in value:
            item_text = _safe_text(item)
            if item_text:
                parts.append(item_text)
        return "；".join(parts)
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

    technique_include_keys = {
        "theme",
        "case_material",
        "rationale",
        "meta_skill",
        "skill_name",
        "skill_description",
        "when_to_use",
        "trigger",
        "strategy",
        "objective_recap",
        "homework",
        "session_focus",
        "next_session_plan",
        "method",
        "technique",
    }
    technique_exclude_keys = {
        "client_info",
        "intake_profile",
        "static_traits",
        "family_status",
        "growth_experiences",
        "special_situations",
        "persona_links",
        "dialogue",
        "name",
        "client_id",
    }

    def collect_technique_text(obj: Any) -> List[str]:
        results: List[str] = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in technique_exclude_keys:
                    continue

                if key in technique_include_keys:
                    if isinstance(value, (str, int, float, bool)):
                        text = _safe_text(value).strip()
                        if text:
                            results.append(f"{key}: {text}")
                    elif isinstance(value, list):
                        for item in value:
                            item_text = _safe_text(item).strip()
                            if item_text:
                                results.append(f"{key}: {item_text}")
                    elif isinstance(value, dict):
                        nested_text = _safe_text(value).strip()
                        if nested_text:
                            results.append(f"{key}: {nested_text}")

                results.extend(collect_technique_text(value))

        elif isinstance(obj, list):
            for item in obj:
                results.extend(collect_technique_text(item))

        return results

    for stage in case_data.get("global_plan", []):
        stage_name = _safe_text(stage.get("stage_name"))
        sessions_desc = _safe_text(stage.get("sessions"))
        technique_lines = collect_technique_text(stage.get("content", {}))
        if not technique_lines:
            continue

        technique_area = f"global_plan::{stage_name or 'unknown'}"
        technique_text = (
            f"技术域: {technique_area}\n"
            f"会谈范围: {sessions_desc}\n"
            f"技术要点:\n- " + "\n- ".join(technique_lines)
        )

        docs.append(
            Document(
                page_content=technique_text,
                metadata={
                    "source": file_name,
                    "doc_type": "technique",
                    "technique_area": technique_area,
                    "title": stage_name or "global_plan_technique",
                },
            )
        )

    for session in case_data.get("sessions", []):
        session_no = _safe_text(session.get("session_number", "unknown"))
        technique_lines = collect_technique_text(session)
        if not technique_lines:
            continue

        technique_area = f"session::{session_no}"
        technique_text = (
            f"技术域: {technique_area}\n"
            f"技术要点:\n- " + "\n- ".join(technique_lines)
        )

        docs.append(
            Document(
                page_content=technique_text,
                metadata={
                    "source": file_name,
                    "doc_type": "technique",
                    "technique_area": technique_area,
                    "title": f"session_{session_no}",
                },
            )
        )

    return docs


def load_cbt_documents(data_dir: Path = CBT_DATA_DIR, max_files: int = 148) -> List[Document]:
    all_docs: List[Document] = []
    for file_path in _iter_cbt_json_files(data_dir, max_files=max_files):
        with open(file_path, "r", encoding="utf-8") as file:
            case_data = json.load(file)
        all_docs.extend(_build_case_documents(case_data, file_path.name))
    return all_docs


def _split_documents(documents: List[Document]) -> List[Any]:
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
    split_docs = splitter.split_documents(lc_documents)

    for idx, doc in enumerate(split_docs):
        meta = doc.metadata or {}
        source = meta.get("source", "unknown")
        area = meta.get("technique_area", "unknown")
        meta["chunk_id"] = f"{source}::{area}::chunk_{idx}"
        doc.metadata = meta

    return split_docs


class LangChainTherapyRAGRetriever:
    def __init__(self, top_k: int = DEFAULT_TOP_K):
        deps = _import_langchain_dependencies()
        bm25_retriever_cls = deps["BM25Retriever"]
        tfidf_retriever_cls = deps["TFIDFRetriever"]

        self.top_k = top_k
        self.retrieval_mode = "hybrid_rrf_rerank"
        self.doc_type = "technique"

        docs = load_cbt_documents(CBT_DATA_DIR, max_files=148)
        chunks = _split_documents(docs)

        self.bm25_retriever = bm25_retriever_cls.from_documents(chunks)
        self.tfidf_retriever = tfidf_retriever_cls.from_documents(chunks)

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        return re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}|\d+", (text or "").lower())

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        stop_words = {
            "的", "了", "和", "是", "在", "我", "也", "就", "都", "很", "与", "及", "并",
            "一个", "一些", "这个", "那个", "因为", "所以", "如果", "但是", "然后", "自己",
        }
        tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}|\d+", (text or "").lower())
        return [token for token in tokens if token not in stop_words][:12]

    @staticmethod
    def _doc_key(doc: Any) -> str:
        meta = doc.metadata or {}
        return str(meta.get("chunk_id") or hash(doc.page_content[:300]))

    def _build_final_query(
        self,
        query: str,
        conversation_summary: str,
        focus_tags: List[str] | None,
        stage_hint: str | None,
    ) -> str:
        tag_text = " ".join((focus_tags or [])[:8]).strip()
        keyword_text = " ".join(
            self._extract_keywords(f"{conversation_summary} {query} {tag_text} {stage_hint or ''}")
        )
        return (
            f"当前问题: {query}\n"
            f"近期摘要: {conversation_summary or '无'}\n"
            f"聚焦标签: {tag_text or '无'}\n"
            f"阶段提示: {stage_hint or 'intervention'}\n"
            f"关键词: {keyword_text}"
        )

    def _rrf_fuse(
        self,
        bm25_docs: List[Any],
        tfidf_docs: List[Any],
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        tfidf_weight: float = 1.0,
    ) -> List[Tuple[Any, float]]:
        scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Any] = {}

        for rank, doc in enumerate(bm25_docs, start=1):
            key = self._doc_key(doc)
            scores[key] += bm25_weight / (rrf_k + rank)
            doc_map[key] = doc

        for rank, doc in enumerate(tfidf_docs, start=1):
            key = self._doc_key(doc)
            scores[key] += tfidf_weight / (rrf_k + rank)
            doc_map[key] = doc

        if not scores:
            return []

        min_score = min(scores.values())
        max_score = max(scores.values())
        denom = max(max_score - min_score, 1e-12)

        normalized_ranked: List[Tuple[Any, float]] = []
        for key, score in scores.items():
            normalized_ranked.append((doc_map[key], (score - min_score) / denom))

        normalized_ranked.sort(key=lambda item: item[1], reverse=True)
        return normalized_ranked[: max(self.top_k * 8, 20)]

    def _rerank_candidates(self, query: str, fused_docs: List[Tuple[Any, float]]) -> List[Dict[str, Any]]:
        if not fused_docs:
            return []

        query_tokens = self._tokenize_text(query)
        if not query_tokens:
            return []

        query_set = set(query_tokens)
        reranked: List[Dict[str, Any]] = []

        for doc, fusion_score in fused_docs:
            doc_tokens = self._tokenize_text(doc.page_content)
            if not doc_tokens:
                continue

            doc_set = set(doc_tokens)
            overlap = query_set & doc_set
            overlap_count = len(overlap)
            coverage = overlap_count / max(len(query_set), 1)
            precision = overlap_count / max(len(doc_set), 1)
            lexical_score = 0.7 * coverage + 0.3 * precision
            final_score = 0.7 * fusion_score + 0.3 * lexical_score

            reranked.append(
                {
                    "doc": doc,
                    "rrf_score": float(fusion_score),
                    "lexical_score": float(lexical_score),
                    "final_score": float(final_score),
                    "matched_keywords": sorted(list(overlap))[:10],
                }
            )

        reranked.sort(key=lambda item: item["final_score"], reverse=True)
        return reranked[: self.top_k]

    def _retrieve_docs(self, final_query: str, top_k: int) -> List[Dict[str, Any]]:
        self.top_k = top_k
        candidate_k = max(top_k * 5, 10)
        self.bm25_retriever.k = candidate_k
        self.tfidf_retriever.k = candidate_k

        bm25_docs = self.bm25_retriever.invoke(final_query)
        tfidf_docs = self.tfidf_retriever.invoke(final_query)

        fused_docs = self._rrf_fuse(
            bm25_docs=bm25_docs,
            tfidf_docs=tfidf_docs,
            rrf_k=60,
            bm25_weight=1.0,
            tfidf_weight=1.0,
        )
        return self._rerank_candidates(final_query, fused_docs)

    def retrieve(
        self,
        query: str,
        conversation_summary: str,
        top_k: int = DEFAULT_TOP_K,
        focus_tags: List[str] | None = None,
        stage_hint: str | None = None,
        safety_level: str = "low",
        language: str = "zh",
        debug: bool = False,
    ) -> Dict[str, Any]:
        final_query = self._build_final_query(
            query=query,
            conversation_summary=conversation_summary,
            focus_tags=focus_tags,
            stage_hint=stage_hint,
        )
        ranked_docs = self._retrieve_docs(final_query=final_query, top_k=top_k)

        references: List[Dict[str, Any]] = []
        score_values: List[float] = []

        for index, item in enumerate(ranked_docs, start=1):
            doc = item["doc"]
            meta = doc.metadata or {}
            final_score = float(item["final_score"])
            score_values.append(final_score)

            references.append(
                {
                    "id": meta.get("chunk_id") or f"tech::{meta.get('technique_area', 'unknown')}::chunk_{index}",
                    "title": meta.get("title") or meta.get("technique_area") or "technique",
                    "technique_area": meta.get("technique_area", "unknown"),
                    "content": doc.page_content[:600],
                    "score": round(final_score, 4),
                    "score_breakdown": {
                        "rrf_score": round(float(item["rrf_score"]), 4),
                        "lexical_score": round(float(item["lexical_score"]), 4),
                    },
                    "matched_keywords": item["matched_keywords"],
                }
            )

        max_score = max(score_values) if score_values else 0.0
        avg_score = (sum(score_values) / len(score_values)) if score_values else 0.0
        low_confidence = max_score < 0.45
        need_clarification = low_confidence or avg_score < 0.35

        result: Dict[str, Any] = {
            "query_meta": {
                "final_query": final_query,
                "top_k": top_k,
                "mode": self.retrieval_mode,
                "doc_type": self.doc_type,
                "safety_level": safety_level,
                "language": language,
            },
            "references": references,
            "retrieval_quality": {
                "max_score": round(max_score, 4),
                "avg_score": round(avg_score, 4),
                "low_confidence": low_confidence,
                "need_clarification": need_clarification,
            },
        }

        if debug:
            result["debug"] = {
                "returned_count": len(references),
            }

        return result


def run_cli(
    question: str,
    conversation_summary: str,
    top_k: int,
    focus_tags: str,
    stage_hint: str,
    safety_level: str,
    language: str,
    debug: bool,
) -> None:
    tag_list = [tag.strip() for tag in focus_tags.split(",") if tag.strip()] if focus_tags else []

    retriever = LangChainTherapyRAGRetriever(top_k=top_k)
    result = retriever.retrieve(
        query=question,
        conversation_summary=conversation_summary,
        top_k=top_k,
        focus_tags=tag_list,
        stage_hint=stage_hint,
        safety_level=safety_level,
        language=language,
        debug=debug,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Therapy technique-only RAG retriever (Agent Tool Output)")
    parser.add_argument("--question", type=str, required=True, help="当前用户问题")
    parser.add_argument("--conversation-summary", type=str, default="", help="最近3-5轮会话摘要")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="返回条数，建议1-8")
    parser.add_argument("--focus-tags", type=str, default="", help="可选标签，逗号分隔，例如: 焦虑,回避,认知重评")
    parser.add_argument(
        "--stage-hint",
        type=str,
        default="intervention",
        choices=["assessment", "intervention", "homework"],
        help="会谈阶段提示",
    )
    parser.add_argument(
        "--safety-level",
        type=str,
        default="low",
        choices=["low", "medium", "high"],
        help="风险等级提示",
    )
    parser.add_argument("--language", type=str, default="zh", help="语言")
    parser.add_argument("--debug", action="store_true", help="输出调试信息")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cli(
        question=args.question,
        conversation_summary=args.conversation_summary,
        top_k=args.top_k,
        focus_tags=args.focus_tags,
        stage_hint=args.stage_hint,
        safety_level=args.safety_level,
        language=args.language,
        debug=args.debug,
    )
