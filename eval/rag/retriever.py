"""
RAG Retriever for CBT Counselor Agent

Retrieves relevant cognitive frameworks and intervention strategies
based on client presentation and current therapy stage.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set
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
        self._keyword_cache: Dict[str, Dict[int, Any]] = {'frameworks': {}, 'strategies': {}, 'examples': {}}
        
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

        # Performance optimization: pre-compute keywords
        for idx, framework in enumerate(self.cognitive_frameworks):
            self._keyword_cache['frameworks'][idx] = {}
            self._keyword_cache['frameworks'][idx]['event'] = set(w for w in str(framework.get('event', '')).lower().split() if len(w) > 2)

            framework_text = " ".join([
                str(framework.get("event", "")),
                " ".join(framework.get("automatic_thoughts", [])),
                " ".join(framework.get("compensatory_strategies", [])),
            ]).lower()
            self._keyword_cache['frameworks'][idx]['combined'] = set(w for w in framework_text.split() if len(w) > 2)

        for idx, strategy in enumerate(self.intervention_strategies):
            self._keyword_cache['strategies'][idx] = {}
            self._keyword_cache['strategies'][idx]['theme'] = set(w for w in str(strategy.get('theme', '')).lower().split() if len(w) > 2)
            theme_rationale_text = f"{strategy.get('theme', '')} {strategy.get('rationale', '')}".lower()
            self._keyword_cache['strategies'][idx]['theme_rationale'] = set(w for w in theme_rationale_text.split() if len(w) > 2)

        for idx, progress in enumerate(self.therapy_progress):
            self._keyword_cache['examples'][idx] = {}
            self._keyword_cache['examples'][idx]['stage_name'] = set(w for w in str(progress.get('stage_name', '')).lower().split() if len(w) > 2)
            self._keyword_cache['examples'][idx]['therapy_content'] = set(w for w in str(progress.get('therapy_content', '')).lower().split() if len(w) > 2)
            focus_areas = progress.get("focus_areas", [])
            if focus_areas:
                self._keyword_cache['examples'][idx]['focus'] = set(f.lower() for f in focus_areas)
    
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
        
        # Retrieve cognitive frameworks
        frameworks = self._retrieve_cognitive_frameworks(
            client_problem,
            current_cognitive_patterns,
            client_topic,
            top_k,
            relevance_scores
        )
        
        # Retrieve intervention strategies
        strategies = self._retrieve_intervention_strategies(
            client_problem,
            current_cognitive_patterns,
            therapy_stage,
            client_topic,
            top_k,
            relevance_scores
        )
        
        # Retrieve therapy progress examples
        examples = self._retrieve_therapy_examples(
            client_problem,
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
        cognitive_patterns: Optional[List[str]],
        client_topic: Optional[str],
        top_k: int,
        relevance_scores: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant cognitive frameworks"""
        scores = []
        
        # Performance optimization: pre-tokenize client problem and topic to avoid redundant O(N) operations in the loop
        client_problem_keywords = set(w for w in client_problem.lower().split() if len(w) > 2)

        for idx, framework in enumerate(self.cognitive_frameworks):
            score = 0.0
            
            # Match by problem category/topic
            if client_topic and framework.get("problem_category") == client_topic:
                score += 0.3
            
            # Match by automatic thoughts
            event_keywords = self._keyword_cache['frameworks'].get(idx, {}).get('event', framework.get("event", ""))
            if self._text_similarity(client_problem_keywords, event_keywords):
                score += 0.25
            
            # Match by cognitive patterns
            if cognitive_patterns:
                framework_patterns = framework.get("cognitive_patterns", [])
                matched_patterns = set(cognitive_patterns) & set(framework_patterns)
                if matched_patterns:
                    score += 0.25 * (len(matched_patterns) / len(cognitive_patterns))
            
            # Match by keywords in problem
            if self._keyword_overlap(client_problem_keywords, framework, framework_idx=idx):
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
        client_problem: str,
        cognitive_patterns: Optional[List[str]],
        therapy_stage: str,
        client_topic: Optional[str],
        top_k: int,
        relevance_scores: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant intervention strategies"""
        scores = []
        
        # Performance optimization: pre-tokenize client problem and topic to avoid redundant O(N) operations in the loop
        client_problem_keywords = set(w for w in client_problem.lower().split() if len(w) > 2)
        client_topic_keywords = set(w for w in client_topic.lower().split() if len(w) > 2) if client_topic else set()

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
            
            # Match by theme/technique relevance to problem
            theme_rationale_keywords = self._keyword_cache['strategies'].get(idx, {}).get('theme_rationale', f"{strategy.get('theme', '')} {strategy.get('rationale', '')}")
            if self._text_similarity(client_problem_keywords, theme_rationale_keywords):
                score += 0.25
            
            # Match by problem category
            theme_keywords = self._keyword_cache['strategies'].get(idx, {}).get('theme', strategy.get("theme", ""))
            if client_topic and self._text_similarity(client_topic_keywords, theme_keywords):
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
        client_problem: str,
        therapy_stage: str,
        client_topic: Optional[str],
        top_k: int,
        relevance_scores: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Retrieve therapy progress examples from similar cases"""
        scores = []
        
        # Performance optimization: pre-tokenize client problem and topic to avoid redundant O(N) operations in the loop
        client_problem_keywords = set(w for w in client_problem.lower().split() if len(w) > 2)
        client_problem_keywords_all = set(client_problem.lower().split())
        client_topic_keywords = set(w for w in client_topic.lower().split() if len(w) > 2) if client_topic else set()

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
            if idx in self._keyword_cache['examples'] and 'focus' in self._keyword_cache['examples'][idx]:
                overlap = client_problem_keywords_all & self._keyword_cache['examples'][idx]['focus']
                if overlap:
                    score += 0.4
            else:
                focus_areas = progress.get("focus_areas", [])
                if focus_areas:
                    focus_keywords = set(f.lower() for f in focus_areas)
                    overlap = client_problem_keywords_all & focus_keywords
                    if overlap:
                        score += 0.4
            
            # Match by topic
            stage_name_keywords = self._keyword_cache['examples'].get(idx, {}).get('stage_name', progress.get("stage_name", ""))
            if client_topic and self._text_similarity(client_topic_keywords, stage_name_keywords):
                score += 0.2
            
            # Match by therapy content
            therapy_content_keywords = self._keyword_cache['examples'].get(idx, {}).get('therapy_content', progress.get("therapy_content", ""))
            if self._text_similarity(client_problem_keywords, therapy_content_keywords):
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
    
    def _text_similarity(self, text1: Union[str, Set[str]], text2: Union[str, Set[str]]) -> bool:
        """Simple text similarity check based on keyword overlap"""
        if not text1 or not text2:
            return False

        # Performance optimization: support pre-tokenized sets
        if isinstance(text1, set):
            keywords1 = text1
        else:
            if not isinstance(text1, str):
                text1 = str(text1)
            keywords1 = set(w for w in text1.lower().split() if len(w) > 2)

        if isinstance(text2, set):
            keywords2 = text2
        else:
            if not isinstance(text2, str):
                text2 = str(text2)
            keywords2 = set(w for w in text2.lower().split() if len(w) > 2)
        
        # Check for overlap
        overlap = keywords1 & keywords2
        return len(overlap) > 0
    
    def _keyword_overlap(self, problem: Union[str, Set[str]], framework: Dict[str, Any], framework_idx: Optional[int] = None) -> bool:
        """Check keyword overlap between problem and framework"""
        if isinstance(problem, set):
            problem_keywords = problem
        else:
            problem_keywords = set(w.lower() for w in problem.split() if len(w) > 2)

        # Performance optimization: use pre-computed framework keywords if available
        if framework_idx is not None and framework_idx in self._keyword_cache['frameworks']:
            framework_keywords = self._keyword_cache['frameworks'][framework_idx].get('combined', set())
        else:
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
