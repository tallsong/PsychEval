import json
import tempfile
from pathlib import Path

from eval.rag.retriever import CBTRetriever


def test_strategy_relevance_above_0_8():
    # Create temporary KB directory
    with tempfile.TemporaryDirectory() as td:
        kb_dir = Path(td)

        # Empty cognitive frameworks and therapy progress
        (kb_dir / "cognitive_frameworks.json").write_text(json.dumps([]), encoding='utf-8')
        (kb_dir / "therapy_progress.json").write_text(json.dumps([]), encoding='utf-8')

        # Create a strategy that should score highly when queried with matching inputs
        strategy = {
            "case_id": 999,
            "stage_number": 1,
            "stage_name": "问题概念化与目标设定",
            "session_number": 1,
            "theme": "职业发展 Perfectionism 工作 转换",
            "technique": "targeted_intervention",
            "rationale": "针对工作转换引发的焦虑，使用ABC模型和思维记录。",
            "case_material": ["示例步骤：记录自动化思维，进行证据检验"],
            "target_cognitive_pattern": "Perfectionism",
            "expected_outcome": "减少焦虑"
        }

        (kb_dir / "intervention_strategies.json").write_text(json.dumps([strategy]), encoding='utf-8')

        # Initialize retriever with temporary kb
        retriever = CBTRetriever(str(kb_dir))

        # Query that matches the strategy on multiple signals
        res = retriever.retrieve(
            client_problem="我对换工作感到非常焦虑，担心自己失败而被否定",
            current_cognitive_patterns=["Perfectionism"],
            therapy_stage="initial_conceptualization",
            client_topic="职业发展",
            top_k=1,
        )

        # Expect at least one strategy score > 0.8
        scores = {k: v for k, v in res.relevance_scores.items() if k.startswith('strategy_')}
        assert scores, "No strategy scores returned"
        # take max score
        max_score = max(scores.values())
        assert max_score > 0.8, f"Expected high relevance >0.8, got {max_score}"
