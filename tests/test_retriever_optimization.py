
import json
import tempfile
from pathlib import Path
from eval.rag import CBTRetriever

def test_keyword_extraction_and_mixed_types():
    with tempfile.TemporaryDirectory() as td:
        kb_dir = Path(td)

        # Test Case 1: automatic_thoughts is a string
        # Test Case 2: automatic_thoughts is a list
        frameworks = [
            {
                "case_id": 1,
                "event": "Event 1",
                "automatic_thoughts": "Thought String",
                "compensatory_strategies": ["Strategy List"],
                "cognitive_patterns": ["Pattern 1"],
                "problem_category": "Topic 1"
            },
            {
                "case_id": 2,
                "event": "Event 2",
                "automatic_thoughts": ["Thought", "List"],
                "compensatory_strategies": "Strategy String",
                "cognitive_patterns": ["Pattern 2"],
                "problem_category": "Topic 2"
            }
        ]

        (kb_dir / "cognitive_frameworks.json").write_text(json.dumps(frameworks), encoding='utf-8')
        (kb_dir / "intervention_strategies.json").write_text(json.dumps([]), encoding='utf-8')
        (kb_dir / "therapy_progress.json").write_text(json.dumps([]), encoding='utf-8')

        retriever = CBTRetriever(str(kb_dir))

        # Verify pre-computed keywords for Case 1
        fw1 = retriever.cognitive_frameworks[0]
        # "Thought String" should result in {'thought', 'string'}
        assert 'thought' in fw1['_combined_keywords']
        assert 'string' in fw1['_combined_keywords']
        # "Strategy List" should result in {'strategy', 'list'}
        assert 'strategy' in fw1['_combined_keywords']
        assert 'list' in fw1['_combined_keywords']

        # Verify pre-computed keywords for Case 2
        fw2 = retriever.cognitive_frameworks[1]
        # ["Thought", "List"] -> "Thought List" -> {'thought', 'list'}
        assert 'thought' in fw2['_combined_keywords']
        assert 'list' in fw2['_combined_keywords']
        # "Strategy String" -> {'strategy', 'string'}
        assert 'strategy' in fw2['_combined_keywords']
        assert 'string' in fw2['_combined_keywords']

        print("Keyword extraction test passed!")

        # Verify retrieval works
        # Search for "Thought" should match both
        res = retriever.retrieve(
            client_problem="I have a thought about it",
            top_k=2
        )
        assert len(res.cognitive_frameworks) == 2

        print("Retrieval test passed!")

if __name__ == "__main__":
    test_keyword_extraction_and_mixed_types()
