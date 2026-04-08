import time
import json
from pathlib import Path

from eval.rag.retriever import CBTRetriever
from eval.rag.het_retriever import HETRetriever
from eval.rag.pdt_retriever import PDTRetriever

def benchmark_cbt():
    retriever = CBTRetriever("eval/rag/knowledge_base")
    start = time.time()
    for _ in range(100):
        retriever.retrieve(
            client_problem="我总是觉得自己不够好，工作上也经常出错，非常焦虑",
            current_cognitive_patterns=["Perfectionism"],
            therapy_stage="core_intervention",
            client_topic="职业发展"
        )
    return time.time() - start

def benchmark_het():
    retriever = HETRetriever("eval/rag/knowledge_base")
    start = time.time()
    for _ in range(100):
        retriever.retrieve(
            client_problem="我感到很孤独，觉得生活没有意义",
            self_perception="我是一个不值得被爱的人",
            existential_concern="孤独"
        )
    return time.time() - start

def benchmark_pdt():
    retriever = PDTRetriever("eval/rag/knowledge_base")
    start = time.time()
    for _ in range(100):
        retriever.retrieve(
            client_problem="我总是害怕被人抛弃，所以不敢建立亲密关系",
            relational_patterns=["主动疏远", "过度依赖"],
            defensive_behaviors=["否认", "投射"]
        )
    return time.time() - start

if __name__ == "__main__":
    print(f"CBT Retriever 100x: {benchmark_cbt():.4f}s")
    print(f"HET Retriever 100x: {benchmark_het():.4f}s")
    print(f"PDT Retriever 100x: {benchmark_pdt():.4f}s")
