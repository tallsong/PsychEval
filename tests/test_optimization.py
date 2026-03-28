
import json
import unittest
from pathlib import Path
import shutil
import tempfile
from eval.rag.retriever import CBTRetriever

class TestCBTRetrieverOptimization(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.kb_dir = Path(self.test_dir)

        # Create dummy knowledge base
        self.frameworks = [{
            "event": "I failed the test",
            "automatic_thoughts": ["I am stupid"],
            "cognitive_patterns": ["Labeling"],
            "compensatory_strategies": ["Study harder"],
            "problem_category": "Academic"
        }]

        self.strategies = [{
            "stage_number": 2,
            "theme": "Cognitive Restructuring",
            "rationale": "Challenge the thought that failure means stupidity",
            "target_cognitive_pattern": "Labeling",
            "technique": "Evidence gathering"
        }]

        self.progress = [{
            "stage_number": 2,
            "stage_name": "Core Intervention",
            "focus_areas": ["Academic stress"],
            "therapy_content": "Discussed failure on test"
        }]

        (self.kb_dir / "cognitive_frameworks.json").write_text(json.dumps(self.frameworks), encoding='utf-8')
        (self.kb_dir / "intervention_strategies.json").write_text(json.dumps(self.strategies), encoding='utf-8')
        (self.kb_dir / "therapy_progress.json").write_text(json.dumps(self.progress), encoding='utf-8')

        self.retriever = CBTRetriever(str(self.kb_dir))

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_precomputation(self):
        # Verify pre-computed keys exist
        self.assertTrue("_cached_event_keywords" in self.retriever.cognitive_frameworks[0])
        self.assertTrue("_cached_full_keywords" in self.retriever.cognitive_frameworks[0])
        self.assertTrue("_cached_theme_rationale_keywords" in self.retriever.intervention_strategies[0])

        # Verify content
        event_keywords = self.retriever.cognitive_frameworks[0]["_cached_event_keywords"]
        self.assertIn("failed", event_keywords)
        self.assertIn("test", event_keywords)
        # "I" should be filtered out (len <= 2), "the" (len=3) should be kept
        self.assertNotIn("i", event_keywords)
        self.assertIn("the", event_keywords)

    def test_retrieval_logic(self):
        # Query matching the data
        result = self.retriever.retrieve(
            client_problem="I failed the test and I feel stupid",
            current_cognitive_patterns=["Labeling"],
            therapy_stage="core_intervention",
            client_topic="Academic"
        )

        # Should retrieve the framework
        self.assertTrue(len(result.cognitive_frameworks) > 0)
        self.assertEqual(result.cognitive_frameworks[0]["event"], "I failed the test")

        # Should retrieve the strategy
        self.assertTrue(len(result.intervention_strategies) > 0)
        self.assertEqual(result.intervention_strategies[0]["theme"], "Cognitive Restructuring")

        # Should retrieve the progress example
        self.assertTrue(len(result.therapy_progress_examples) > 0)

if __name__ == '__main__':
    unittest.main()
