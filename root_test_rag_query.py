"""
Simple RAG query test script.

Usage:
    python test_rag_query.py --modality cbt --question "我的工作让我很焦虑"
# python -u test_rag_query.py --modality cbt --question "我对换工作感到非常焦虑" --top_k 1
Outputs a JSON object with the retriever results for the chosen modality.
"""
import sys
import json
from pathlib import Path
import argparse

# ensure eval.rag is importable
sys.path.insert(0, str(Path(__file__).parent / "eval" / "rag"))

from eval.rag import CBTRetriever
from eval.rag import HETRetriever
from eval.rag import PDTRetriever


def query_rag(modality: str, question: str, top_k: int = 3):
    project_root = Path(__file__).parent
    kb_dir = str(project_root / "eval" / "rag" / "knowledge_base")

    modality = modality.lower()
    if modality == "cbt":
        retriever = CBTRetriever(kb_dir)
        res = retriever.retrieve(
            client_problem=question,
            current_cognitive_patterns=None,
            therapy_stage=None,
            client_topic=None,
            top_k=top_k,
        )
        return {
            "modality": "cbt",
            "cognitive_frameworks": res.cognitive_frameworks,
            "intervention_strategies": res.intervention_strategies,
            "relevance_scores": res.relevance_scores,
        }

    if modality == "het":
        retriever = HETRetriever(kb_dir)
        res = retriever.retrieve(
            client_problem=question,
            self_perception=None,
            existential_concern=None,
            top_k=top_k,
        )
        return {
            "modality": "het",
            "self_concepts": res.self_concepts,
            "existential_themes": res.existential_themes,
            "strategies": res.strategies,
            "relevance_scores": res.relevance_scores,
        }

    if modality == "pdt":
        retriever = PDTRetriever(kb_dir)
        res = retriever.retrieve(
            client_problem=question,
            relational_patterns=None,
            defensive_behaviors=None,
            top_k=top_k,
        )
        return {
            "modality": "pdt",
            "core_conflicts": res.core_conflicts,
            "object_relations": res.object_relations,
            "unconscious_patterns": res.unconscious_patterns,
            "interventions": res.interventions,
            "relevance_scores": res.relevance_scores,
        }

    raise ValueError(f"Unknown modality: {modality}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["cbt", "het", "pdt"], required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    out = query_rag(args.modality, args.question, args.top_k)
    print(json.dumps(out, ensure_ascii=False, indent=2))
# python test_rag_query.py --modality cbt --question "我对换工作感到非常焦虑" --top_k 1