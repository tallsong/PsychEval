import time
from eval.rag.retriever import CBTRetriever
from eval.rag.pdt_retriever import PDTRetriever
from eval.rag.het_retriever import HETRetriever

def bench():
    start = time.time()
    for _ in range(100):
        cbt = CBTRetriever("eval/rag/knowledge_base/")
        pdt = PDTRetriever("eval/rag/knowledge_base/")
        het = HETRetriever("eval/rag/knowledge_base/")

        cbt.retrieve(
            "我对换工作感到非常焦虑，担心自己失败而被否定",
            ["Perfectionism"],
            "initial_conceptualization",
            "职业发展"
        )
        pdt.retrieve(
            "经常在恋爱中感到被抛弃，然后主动推开对方",
            ["防御机制", "推开"],
            ["愤怒", "恐惧"]
        )
        het.retrieve(
            "感觉工作很无意义，没有成就感",
            "我是一个失败的人",
            "人生无意义"
        )
    print(f"Elapsed: {time.time() - start:.3f}s")

if __name__ == "__main__":
    bench()
