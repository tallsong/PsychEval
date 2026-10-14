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
            # Performance optimization: pre-compute framework keywords
            self._fw_event_keywords = []
            self._fw_overlap_keywords = []
            for fw in self.cognitive_frameworks:
                event = str(fw.get("event", ""))
                self._fw_event_keywords.append(set(w for w in event.lower().split() if len(w) > 2))

                framework_text = " ".join([
                    str(fw.get("event", "")),
                    " ".join(fw.get("automatic_thoughts", [])),
                    " ".join(fw.get("compensatory_strategies", [])),
                ]).lower()
                self._fw_overlap_keywords.append(set(w for w in framework_text.split() if len(w) > 2))
        
        strategies_file = self.kb_dir / "intervention_strategies.json"
        if strategies_file.exists():
            with open(strategies_file, 'r', encoding='utf-8') as f:
                self.intervention_strategies = json.load(f)
            # Performance optimization: pre-compute strategy keywords
            self._strategy_theme_rationale_keywords = []
            self._strategy_theme_keywords = []
            for s in self.intervention_strategies:
                theme_rationale = f"{s.get('theme', '')} {s.get('rationale', '')}"
                self._strategy_theme_rationale_keywords.append(set(w for w in theme_rationale.lower().split() if len(w) > 2))
                theme = str(s.get("theme", ""))
                self._strategy_theme_keywords.append(set(w for w in theme.lower().split() if len(w) > 2))
        
        progress_file = self.kb_dir / "therapy_progress.json"
        if progress_file.exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                self.therapy_progress = json.load(f)
            # Performance optimization: pre-compute progress keywords
            self._progress_stage_name_keywords = []
            self._progress_therapy_content_keywords = []
            self._progress_focus_keywords = []
            for p in self.therapy_progress:
                stage_name = str(p.get("stage_name", ""))
                self._progress_stage_name_keywords.append(set(w for w in stage_name.lower().split() if len(w) > 2))

                content = p.get("therapy_content", "")
                if isinstance(content, list):
                    content_str = " ".join(str(c) for c in content)
                else:
                    content_str = str(content)
                self._progress_therapy_content_keywords.append(set(w for w in content_str.lower().split() if len(w) > 2))

                focus_areas = p.get("focus_areas", [])
                self._progress_focus_keywords.append(set(f.lower() for f in focus_areas) if focus_areas else set())

        
        metadata_file = self.kb_dir / "case_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.case_metadata = {int(k): v for k, v in metadata.items()}
        
        print(f"Loaded knowledge base:")
        print(f"  - {len(self.cognitive_frameworks)} cognitive frameworks")
        print(f"  - {len(self.intervention_strategies)} intervention strategies")
        print(f"  - {len(self.therapy_progress)} therapy progress records")
    
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
        # Performance optimization: pre-tokenize queries
        problem_keywords = set(w for w in client_problem.lower().split() if len(w) > 2) if client_problem else set()

        scores = []
        
        for idx, framework in enumerate(self.cognitive_frameworks):
            score = 0.0
            
            # Match by problem category/topic
            if client_topic and framework.get("problem_category") == client_topic:
                score += 0.3
            
            # Match by automatic thoughts
            if self._text_similarity(problem_keywords, self._fw_event_keywords[idx]):
                score += 0.25
            
            # Match by cognitive patterns
            if cognitive_patterns:
                framework_patterns = framework.get("cognitive_patterns", [])
                matched_patterns = set(cognitive_patterns) & set(framework_patterns)
                if matched_patterns:
                    score += 0.25 * (len(matched_patterns) / len(cognitive_patterns))
            
            # Match by keywords in problem
            if self._keyword_overlap(problem_keywords, framework, self._fw_overlap_keywords[idx]):
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
        # Performance optimization: pre-tokenize queries
        problem_keywords = set(w for w in client_problem.lower().split() if len(w) > 2) if client_problem else set()
        topic_keywords = set(w for w in client_topic.lower().split() if len(w) > 2) if client_topic else set()

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
            
            # Match by theme/technique relevance to problem
            if self._text_similarity(problem_keywords, self._strategy_theme_rationale_keywords[idx]):
                score += 0.25
            
            # Match by problem category
            if client_topic and self._text_similarity(topic_keywords, self._strategy_theme_keywords[idx]):
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
        # Performance optimization: pre-tokenize queries
        problem_keywords = set(w for w in client_problem.lower().split() if len(w) > 2) if client_problem else set()
        topic_keywords = set(w for w in client_topic.lower().split() if len(w) > 2) if client_topic else set()
        problem_keywords_lower = set(client_problem.lower().split()) if client_problem else set()

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
            if self._progress_focus_keywords[idx]:
                overlap = problem_keywords_lower & self._progress_focus_keywords[idx]
                if overlap:
                    score += 0.4
            
            # Match by topic
            if client_topic and self._text_similarity(topic_keywords, self._progress_stage_name_keywords[idx]):
                score += 0.2
            
            # Match by therapy content
            if self._text_similarity(problem_keywords, self._progress_therapy_content_keywords[idx]):
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
    
    def _text_similarity(self, text1: Any, text2: Any) -> bool:
        """Simple text similarity check based on keyword overlap"""
        # Performance optimization: explicit fast path for pre-computed sets
        if type(text1) is set and type(text2) is set:
            return len(text1 & text2) > 0

        if not text1 or not text2:
            return False
        
        # Extract keywords for text1
        if isinstance(text1, set):
            keywords1 = text1
        else:
            if not isinstance(text1, str):
                text1 = str(text1)
            keywords1 = set(w for w in text1.lower().split() if len(w) > 2)

        # Extract keywords for text2
        if isinstance(text2, set):
            keywords2 = text2
        else:
            if not isinstance(text2, str):
                text2 = str(text2)
            keywords2 = set(w for w in text2.lower().split() if len(w) > 2)
        
        # Check for overlap
        overlap = keywords1 & keywords2
        return len(overlap) > 0
    
    def _keyword_overlap(self, problem: Any, framework: Dict[str, Any], precomputed_framework_keywords: Optional[set] = None) -> bool:
        """Check keyword overlap between problem and framework"""
        # Performance optimization: support pre-tokenized sets
        if isinstance(problem, set):
            problem_keywords = problem
        else:
            if not isinstance(problem, str):
                problem = str(problem)
            problem_keywords = set(w.lower() for w in problem.split() if len(w) > 2)
        
        if precomputed_framework_keywords is not None:
            framework_keywords = precomputed_framework_keywords
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
