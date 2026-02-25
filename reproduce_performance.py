
import time
import sys
from pathlib import Path
from eval.rag import CBTRetriever

def benchmark_retrieval():
    # Setup paths
    project_root = Path(__file__).parent
    kb_dir = project_root / "eval" / "rag" / "knowledge_base"

    if not kb_dir.exists():
        print(f"Knowledge base directory not found: {kb_dir}")
        return

    print("Initializing retriever...")
    start_init = time.time()
    retriever = CBTRetriever(str(kb_dir))
    end_init = time.time()
    print(f"Initialization took {end_init - start_init:.4f} seconds")

    # Define test queries
    queries = [
        ("我对换工作感到非常焦虑，总是想'如果我失败了怎么办？'这让我无法专注", ["Catastrophizing", "Fortune Telling"], "职业发展"),
        ("我总是觉得伴侣不爱我，每次他晚点回家我都开始想象最坏的情况", ["Mind Reading", "Personalization"], "人际关系"),
        ("我觉得自己很没用，什么都做不好", ["All-or-Nothing Thinking"], "情绪管理"),
        ("虽然我考了第一名，但这只是运气好", ["Disqualifying the Positive"], "学业压力"),
        ("大家都在看我，觉得我很奇怪", ["Mind Reading", "Social Anxiety"], "人际关系")
    ]

    # Run benchmark
    print("\nStarting benchmark...")
    start_bench = time.time()
    iterations = 20

    for _ in range(iterations):
        for client_problem, patterns, topic in queries:
            retriever.retrieve(
                client_problem=client_problem,
                current_cognitive_patterns=patterns,
                therapy_stage="initial_conceptualization",
                client_topic=topic,
                top_k=3
            )

    end_bench = time.time()
    total_time = end_bench - start_bench
    avg_time = total_time / (iterations * len(queries))

    print(f"Total time for {iterations * len(queries)} retrievals: {total_time:.4f} seconds")
    print(f"Average time per retrieval: {avg_time:.4f} seconds")

if __name__ == "__main__":
    benchmark_retrieval()
