import time
import json
from pathlib import Path
from eval.rag.retriever import CBTRetriever

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

if __name__ == "__main__":
    print(f"CBT Retriever 100x: {benchmark_cbt():.4f}s")
