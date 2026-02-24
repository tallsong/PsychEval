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
        
        print(f"Loaded knowledge base:")
        print(f"  - {len(self.cognitive_frameworks)} cognitive frameworks")
        print(f"  - {len(self.intervention_strategies)} intervention strategies")
        print(f"  - {len(self.therapy_progress)} therapy progress records")

        # Pre-compute keywords for optimization
        self._precompute_knowledge_base_keywords()
    
    def _precompute_knowledge_base_keywords(self) -> None:
        """Pre-compute keywords for knowledge base items to speed up retrieval"""
        # Cognitive Frameworks
        for fw in self.cognitive_frameworks:
            # Normalize fields to lists if they are strings
            if isinstance(fw.get("automatic_thoughts"), str):
                fw["automatic_thoughts"] = [fw["automatic_thoughts"]]
            if isinstance(fw.get("compensatory_strategies"), str):
                fw["compensatory_strategies"] = [fw["compensatory_strategies"]]
            if isinstance(fw.get("cognitive_patterns"), str):
                fw["cognitive_patterns"] = [fw["cognitive_patterns"]]

            # Compute event keywords specifically
            fw["_event_keywords"] = self._extract_keywords(fw.get("event", ""))

            # Compute all keywords set
            text_parts = [str(fw.get("event", ""))]
            text_parts.extend(fw.get("automatic_thoughts", []))
            text_parts.extend(fw.get("compensatory_strategies", []))

            combined_text = " ".join(str(p) for p in text_parts)
            fw["_keywords"] = self._extract_keywords(combined_text)

            # Compute patterns set
            fw["_patterns_set"] = set(fw.get("cognitive_patterns", []))

        # Intervention Strategies
        for strategy in self.intervention_strategies:
            # Normalize rationale
            rationale = strategy.get("rationale", "")
            if isinstance(rationale, list):
                rationale = " ".join(rationale)

            theme = strategy.get("theme", "")
            combined_text = f"{theme} {rationale}"
            strategy["_keywords"] = self._extract_keywords(combined_text)
            strategy["_theme_keywords"] = self._extract_keywords(theme)

            # Target cognitive pattern
            target = strategy.get("target_cognitive_pattern")
            if target:
                strategy["_target_pattern_set"] = {target} if isinstance(target, str) else set(target)
            else:
                strategy["_target_pattern_set"] = set()

        # Therapy Progress
        for progress in self.therapy_progress:
            progress["_keywords"] = self._extract_keywords(progress.get("therapy_content", ""))
            progress["_stage_keywords"] = self._extract_keywords(progress.get("stage_name", ""))

            focus_areas = progress.get("focus_areas", [])
            progress["_focus_areas_set"] = set(f.lower() for f in focus_areas)

    def _extract_keywords(self, text: Any) -> Any:
        """Extract keywords set from text"""
        if not text:
            return set()
        if isinstance(text, list):
            text = " ".join(str(x) for x in text)
        if not isinstance(text, str):
            text = str(text)
        return set(w.lower() for w in text.split() if len(w) > 2)

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
        
        # Compute keywords once per request
        problem_keywords = self._extract_keywords(client_problem)
        # For compatibility with legacy behavior, also compute unfiltered tokens (split by space)
        # This is needed for focus_areas matching in therapy examples
        problem_tokens = set(client_problem.lower().split())

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
            problem_keywords,
            problem_tokens,
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
        problem_keywords: Any,
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
            
            # Match by automatic thoughts (event)
            if problem_keywords & framework.get("_event_keywords", set()):
                score += 0.25
            
            # Match by cognitive patterns
            if cognitive_patterns:
                framework_patterns = framework.get("_patterns_set", set())
                matched_patterns = set(cognitive_patterns) & framework_patterns
                if matched_patterns:
                    # Use length of client's patterns as denominator to preserve original behavior
                    total_patterns = len(cognitive_patterns)
                    if total_patterns > 0:
                        score += 0.25 * (len(matched_patterns) / total_patterns)
            
            # Match by keywords in problem (combined)
            if problem_keywords & framework.get("_keywords", set()):
                score += 0.2
            
            if score > 0:
                scores.append((idx, score, framework))
        
        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = [framework for _, score, framework in scores[:top_k]]
        
        # Store relevance scores
        for idx, score, _ in scores[:top_k]:
            relevance_scores[f"framework_{idx}"] = score
        
        return results
    
    def _retrieve_intervention_strategies(
        self,
        problem_keywords: Any,
        topic_keywords: Any,
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
            if cognitive_patterns:
                target_patterns = strategy.get("_target_pattern_set", set())
                if set(cognitive_patterns) & target_patterns:
                    score += 0.3
            
            # Match by theme/technique relevance to problem
            strategy_keywords = strategy.get("_keywords", set())
            if problem_keywords & strategy_keywords:
                score += 0.25
            
            # Match by problem category
            strategy_theme_keywords = strategy.get("_theme_keywords", set())
            if client_topic and (topic_keywords & strategy_theme_keywords):
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
        results = [strategy for _, score, strategy in scores[:top_k]]
        
        # Store relevance scores
        for idx, score, _ in scores[:top_k]:
            relevance_scores[f"strategy_{idx}"] = score
        
        return results
    
    def _retrieve_therapy_examples(
        self,
        problem_keywords: Any,
        problem_tokens: Any,
        topic_keywords: Any,
        therapy_stage: str,
        client_topic: Optional[str],
        top_k: int,
        relevance_scores: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Retrieve therapy progress examples from similar cases"""
        scores = []
        
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
            focus_areas_set = progress.get("_focus_areas_set", set())
            # Use problem_tokens (unfiltered) to match focus areas, preserving legacy behavior
            # where short words could match if present in focus_areas
            if problem_tokens & focus_areas_set:
                score += 0.4
            
            # Match by topic
            stage_keywords = progress.get("_stage_keywords", set())
            if client_topic and (topic_keywords & stage_keywords):
                score += 0.2
            
            # Match by therapy content
            content_keywords = progress.get("_keywords", set())
            if problem_keywords & content_keywords:
                score += 0.1
            
            if score > 0:
                scores.append((idx, score, progress))
        
        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = [progress for _, score, progress in scores[:top_k]]
        
        # Store relevance scores
        for idx, score, _ in scores[:top_k]:
            relevance_scores[f"example_{idx}"] = score
        
        return results
    
    def _text_similarity(self, text1: str, text2: str) -> bool:
        """Simple text similarity check based on keyword overlap"""
        # Deprecated: use pre-computed keywords instead
        k1 = self._extract_keywords(text1)
        k2 = self._extract_keywords(text2)
        return len(k1 & k2) > 0
    
    def _keyword_overlap(self, problem: str, framework: Dict[str, Any]) -> bool:
        """Check keyword overlap between problem and framework"""
        # Deprecated: use pre-computed keywords instead
        return len(self._extract_keywords(problem) & framework.get("_keywords", set())) > 0
    
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
