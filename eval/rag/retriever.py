"""
RAG Retriever for CBT Counselor Agent

Retrieves relevant cognitive frameworks and intervention strategies
based on client presentation and current therapy stage.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class RetrievalResult:
    """Result of RAG retrieval"""
    cognitive_frameworks: List[Dict[str, Any]]
    intervention_strategies: List[Dict[str, Any]]
    therapy_progress_examples: List[Dict[str, Any]]
    relevance_scores: Dict[str, float]


class CBTRetriever:
    """
    Retrieval-Augmented Generation system for CBT counselor.
    
    Retrieves relevant:
    1. Cognitive frameworks (ABC models, patterns)
    2. Intervention strategies (techniques, homework)
    3. Therapy progress examples (similar cases, session content)
    """
    
    def __init__(self, knowledge_base_dir: str):
        """
        Initialize retriever with knowledge base
        
        Args:
            knowledge_base_dir: Directory containing extracted knowledge JSON files
        """
        self.kb_dir = Path(knowledge_base_dir)
        self.cognitive_frameworks: List[Dict[str, Any]] = []
        self.intervention_strategies: List[Dict[str, Any]] = []
        self.therapy_progress: List[Dict[str, Any]] = []
        self.case_metadata: Dict[int, Dict[str, Any]] = {}
        
        self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> None:
        """Load knowledge base from JSON files"""
        frameworks_file = self.kb_dir / "cognitive_frameworks.json"
        if frameworks_file.exists():
            with open(frameworks_file, 'r', encoding='utf-8') as f:
                self.cognitive_frameworks = json.load(f)
        
        strategies_file = self.kb_dir / "intervention_strategies.json"
        if strategies_file.exists():
            with open(strategies_file, 'r', encoding='utf-8') as f:
                self.intervention_strategies = json.load(f)
        
        progress_file = self.kb_dir / "therapy_progress.json"
        if progress_file.exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                self.therapy_progress = json.load(f)
        
        metadata_file = self.kb_dir / "case_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.case_metadata = {int(k): v for k, v in metadata.items()}
        
        # Pre-compute keyword sets for fast O(1) retrieval operations
        self._precompute_keywords()

        print(f"Loaded knowledge base:")
        print(f"  - {len(self.cognitive_frameworks)} cognitive frameworks")
        print(f"  - {len(self.intervention_strategies)} intervention strategies")
        print(f"  - {len(self.therapy_progress)} therapy progress records")

    def _extract_keywords(self, text: Any) -> set:
        """Extract keywords from text, handling non-string inputs."""
        if not text:
            return set()

        if isinstance(text, list):
            text = " ".join(str(item) for item in text)
        elif not isinstance(text, str):
            text = str(text)

        return set(w.lower() for w in text.split() if len(w) > 2)

    def _precompute_keywords(self) -> None:
        """Pre-compute keyword sets for all knowledge base items to optimize retrieval."""
        # Process cognitive frameworks
        for framework in self.cognitive_frameworks:
            # Event keywords
            framework["_event_keywords"] = self._extract_keywords(framework.get("event", ""))

            # Combined keywords for _keyword_overlap
            combined_text = " ".join([
                str(framework.get("event", "")),
                " ".join(framework.get("automatic_thoughts", []) if isinstance(framework.get("automatic_thoughts"), list) else [str(framework.get("automatic_thoughts", ""))]),
                " ".join(framework.get("compensatory_strategies", []) if isinstance(framework.get("compensatory_strategies"), list) else [str(framework.get("compensatory_strategies", ""))]),
            ])
            framework["_combined_keywords"] = self._extract_keywords(combined_text)

        # Process intervention strategies
        for strategy in self.intervention_strategies:
            theme_rationale_text = f"{strategy.get('theme', '')} {strategy.get('rationale', '')}"
            strategy["_theme_rationale_keywords"] = self._extract_keywords(theme_rationale_text)
            strategy["_theme_keywords"] = self._extract_keywords(strategy.get("theme", ""))

        # Process therapy progress examples
        for progress in self.therapy_progress:
            # Pre-compute lowercased focus areas for O(1) matching
            focus_areas = progress.get("focus_areas", [])
            if focus_areas:
                progress["_focus_keywords"] = set(f.lower() for f in focus_areas)
            else:
                progress["_focus_keywords"] = set()

            progress["_stage_name_keywords"] = self._extract_keywords(progress.get("stage_name", ""))
            progress["_content_keywords"] = self._extract_keywords(progress.get("therapy_content", ""))
    
    def retrieve(
        self,
        client_problem: str,
        current_cognitive_patterns: List[str] = None,
        therapy_stage: str = "initial_conceptualization",
        client_topic: str = None,
        top_k: int = 3,
    ) -> RetrievalResult:
        """
        Retrieve relevant knowledge for counselor
        
        Args:
            client_problem: Description of client's main problem
            current_cognitive_patterns: List of identified cognitive patterns (e.g. "Perfectionism", "Catastrophizing")
            therapy_stage: Current stage (initial_conceptualization, core_intervention, consolidation)
            client_topic: Topic category (e.g., "职业发展", "情绪管理")
            top_k: Number of top results to return per category
        
        Returns:
            RetrievalResult containing cognitive frameworks, strategies, and examples
        """
        relevance_scores = {}
        
        # Pre-extract keywords from query to avoid O(N) extraction during retrieval
        problem_keywords = self._extract_keywords(client_problem)
        topic_keywords = self._extract_keywords(client_topic) if client_topic else set()

        # Retrieve cognitive frameworks
        frameworks = self._retrieve_cognitive_frameworks(
            client_problem,
            problem_keywords,
            current_cognitive_patterns,
            client_topic,
            top_k,
            relevance_scores
        )
        
        # Retrieve intervention strategies
        strategies = self._retrieve_intervention_strategies(
            client_problem,
            problem_keywords,
            topic_keywords,
            current_cognitive_patterns,
            therapy_stage,
            client_topic,
            top_k,
            relevance_scores
        )
        
        # Retrieve therapy progress examples
        examples = self._retrieve_therapy_examples(
            client_problem,
            problem_keywords,
            topic_keywords,
            therapy_stage,
            client_topic,
            top_k,
            relevance_scores
        )
        
        return RetrievalResult(
            cognitive_frameworks=frameworks,
            intervention_strategies=strategies,
            therapy_progress_examples=examples,
            relevance_scores=relevance_scores,
        )
    
    def _retrieve_cognitive_frameworks(
        self,
        client_problem: str,
        problem_keywords: set,
        cognitive_patterns: Optional[List[str]],
        client_topic: Optional[str],
        top_k: int,
        relevance_scores: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant cognitive frameworks"""
        scores = []
        
        for idx, framework in enumerate(self.cognitive_frameworks):
            score = 0.0
            
            # Match by problem category/topic
            if client_topic and framework.get("problem_category") == client_topic:
                score += 0.3
            
            # Match by automatic thoughts (O(1) keyword intersection)
            if problem_keywords and framework.get("_event_keywords") and (problem_keywords & framework["_event_keywords"]):
                score += 0.25
            
            # Match by cognitive patterns
            if cognitive_patterns:
                framework_patterns = framework.get("cognitive_patterns", [])
                matched_patterns = set(cognitive_patterns) & set(framework_patterns)
                if matched_patterns:
                    score += 0.25 * (len(matched_patterns) / len(cognitive_patterns))
            
            # Match by keywords in problem (O(1) keyword intersection)
            if problem_keywords and framework.get("_combined_keywords") and (problem_keywords & framework["_combined_keywords"]):
                score += 0.2
            
            if score > 0:
                scores.append((idx, score, framework))
        
        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = [self._clean_result(framework) for _, score, framework in scores[:top_k]]
        
        # Store relevance scores
        for idx, score, _ in scores[:top_k]:
            relevance_scores[f"framework_{idx}"] = score
        
        return results
    
    def _retrieve_intervention_strategies(
        self,
        client_problem: str,
        problem_keywords: set,
        topic_keywords: set,
        cognitive_patterns: Optional[List[str]],
        therapy_stage: str,
        client_topic: Optional[str],
        top_k: int,
        relevance_scores: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant intervention strategies"""
        scores = []
        
        for idx, strategy in enumerate(self.intervention_strategies):
            score = 0.0
            
            # Match by therapy stage
            stage_mapping = {
                "initial_conceptualization": 1,
                "core_intervention": 2,
                "consolidation": 3,
            }
            target_stage = stage_mapping.get(therapy_stage, 2)
            strategy_stage = strategy.get("stage_number", 2)
            if abs(target_stage - strategy_stage) <= 1:
                score += 0.2
            
            # Match by cognitive pattern
            if cognitive_patterns and strategy.get("target_cognitive_pattern"):
                if strategy.get("target_cognitive_pattern") in cognitive_patterns:
                    score += 0.3
            
            # Match by theme/technique relevance to problem (O(1) keyword intersection)
            if problem_keywords and strategy.get("_theme_rationale_keywords") and (problem_keywords & strategy["_theme_rationale_keywords"]):
                score += 0.25
            
            # Match by problem category (O(1) keyword intersection)
            if topic_keywords and strategy.get("_theme_keywords") and (topic_keywords & strategy["_theme_keywords"]):
                score += 0.15
            
            # Bonus for explicit technique match
            if cognitive_patterns:
                theme_lower = strategy.get("theme", "").lower()
                if any(pattern.lower() in theme_lower for pattern in cognitive_patterns):
                    score += 0.1
            
            if score > 0:
                scores.append((idx, score, strategy))
        
        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = [self._clean_result(strategy) for _, score, strategy in scores[:top_k]]
        
        # Store relevance scores
        for idx, score, _ in scores[:top_k]:
            relevance_scores[f"strategy_{idx}"] = score
        
        return results
    
    def _retrieve_therapy_examples(
        self,
        client_problem: str,
        problem_keywords: set,
        topic_keywords: set,
        therapy_stage: str,
        client_topic: Optional[str],
        top_k: int,
        relevance_scores: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Retrieve therapy progress examples from similar cases"""
        scores = []
        
        # In _retrieve_therapy_examples, simple non-filtered split is used for overlap calculation
        # based on existing logic, so we compute a separate unfiltered split for focus areas matching.
        problem_words_unfiltered = set(client_problem.lower().split())

        stage_mapping = {
            "initial_conceptualization": 1,
            "core_intervention": 2,
            "consolidation": 3,
        }
        target_stage = stage_mapping.get(therapy_stage, 2)
        
        for idx, progress in enumerate(self.therapy_progress):
            score = 0.0
            
            # Match by stage
            progress_stage = progress.get("stage_number", 2)
            if abs(target_stage - progress_stage) <= 1:
                score += 0.3
            
            # Match by focus areas
            focus_areas = progress.get("focus_areas", [])
            if focus_areas:
                overlap = problem_words_unfiltered & progress.get("_focus_keywords", set())
                if overlap:
                    score += 0.4
            
            # Match by topic (O(1) keyword intersection)
            if topic_keywords and progress.get("_stage_name_keywords") and (topic_keywords & progress["_stage_name_keywords"]):
                score += 0.2
            
            # Match by therapy content (O(1) keyword intersection)
            if problem_keywords and progress.get("_content_keywords") and (problem_keywords & progress["_content_keywords"]):
                score += 0.1
            
            if score > 0:
                scores.append((idx, score, progress))
        
        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = [self._clean_result(progress) for _, score, progress in scores[:top_k]]
        
        # Store relevance scores
        for idx, score, _ in scores[:top_k]:
            relevance_scores[f"example_{idx}"] = score
        
        return results
    
    def _clean_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Remove internal pre-computed keys before returning result."""
        cleaned = result.copy()
        keys_to_remove = [k for k in cleaned.keys() if k.startswith("_") and k.endswith("_keywords")]
        for k in keys_to_remove:
            del cleaned[k]
        return cleaned

    def _text_similarity(self, text1: str, text2: str) -> bool:
        """Simple text similarity check based on keyword overlap"""
        if not text1 or not text2:
            return False
        
        # Convert to string if necessary
        if not isinstance(text1, str):
            text1 = str(text1)
        if not isinstance(text2, str):
            text2 = str(text2)
        
        # Extract keywords (length > 2)
        keywords1 = set(w for w in text1.lower().split() if len(w) > 2)
        keywords2 = set(w for w in text2.lower().split() if len(w) > 2)
        
        # Check for overlap
        overlap = keywords1 & keywords2
        return len(overlap) > 0
    
    def _keyword_overlap(self, problem: str, framework: Dict[str, Any]) -> bool:
        """Check keyword overlap between problem and framework"""
        problem_keywords = set(w.lower() for w in problem.split() if len(w) > 2)
        
        # Check against various framework fields
        framework_text = " ".join([
            str(framework.get("event", "")),
            " ".join(framework.get("automatic_thoughts", [])),
            " ".join(framework.get("compensatory_strategies", [])),
        ]).lower()
        
        framework_keywords = set(w for w in framework_text.split() if len(w) > 2)
        
        return len(problem_keywords & framework_keywords) > 0
    
    def get_framework_by_pattern(
        self,
        cognitive_pattern: str,
    ) -> List[Dict[str, Any]]:
        """Get cognitive frameworks for specific pattern"""
        return [
            fw for fw in self.cognitive_frameworks
            if cognitive_pattern in fw.get("cognitive_patterns", [])
        ]
    
    def get_strategies_by_stage(self, stage_name: str) -> List[Dict[str, Any]]:
        """Get intervention strategies for specific stage"""
        return [
            s for s in self.intervention_strategies
            if s.get("stage_name") == stage_name
        ]
    
    def get_similar_cases(self, case_id: int) -> List[Dict[str, Any]]:
        """Get cases similar to given case_id"""
        case_meta = self.case_metadata.get(case_id)
        if not case_meta:
            return []
        
        similar = []
        for cid, meta in self.case_metadata.items():
            if cid != case_id and meta.get("topic") == case_meta.get("topic"):
                similar.append(meta)
        
        return similar[:5]
