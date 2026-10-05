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
                # Performance optimization: pre-compute keywords
                for fw in self.cognitive_frameworks:
                    fw['_event_keywords'] = set(w for w in str(fw.get("event", "")).lower().split() if len(w) > 2)
                    auto_thoughts = fw.get("automatic_thoughts", [])
                    comp_strategies = fw.get("compensatory_strategies", [])
                    auto_thoughts_str = " ".join(str(x) for x in auto_thoughts) if isinstance(auto_thoughts, list) else str(auto_thoughts)
                    comp_strategies_str = " ".join(str(x) for x in comp_strategies) if isinstance(comp_strategies, list) else str(comp_strategies)
                    fw_text = " ".join([str(fw.get("event", "")), auto_thoughts_str, comp_strategies_str]).lower()
                    fw['_framework_keywords'] = set(w for w in fw_text.split() if len(w) > 2)
        
        strategies_file = self.kb_dir / "intervention_strategies.json"
        if strategies_file.exists():
            with open(strategies_file, 'r', encoding='utf-8') as f:
                self.intervention_strategies = json.load(f)
                # Performance optimization: pre-compute keywords
                for st in self.intervention_strategies:
                    theme = str(st.get("theme", ""))
                    rationale = st.get("rationale", "")
                    rationale_str = " ".join(str(x) for x in rationale) if isinstance(rationale, list) else str(rationale)
                    st['_theme_rationale_keywords'] = set(w for w in f"{theme} {rationale_str}".lower().split() if len(w) > 2)
                    st['_theme_keywords'] = set(w for w in theme.lower().split() if len(w) > 2)
        
        progress_file = self.kb_dir / "therapy_progress.json"
        if progress_file.exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                self.therapy_progress = json.load(f)
                # Performance optimization: pre-compute keywords
                for pr in self.therapy_progress:
                    pr['_stage_name_keywords'] = set(w for w in str(pr.get("stage_name", "")).lower().split() if len(w) > 2)
                    content = pr.get("therapy_content", "")
                    content_str = " ".join(str(x) for x in content) if isinstance(content, list) else str(content)
                    pr['_content_keywords'] = set(w for w in content_str.lower().split() if len(w) > 2)
                    focus_areas = pr.get("focus_areas", [])
                    if isinstance(focus_areas, list):
                        pr['_focus_keywords'] = set(str(f).lower() for f in focus_areas)
                    elif focus_areas:
                        pr['_focus_keywords'] = set([str(focus_areas).lower()])
                    else:
                        pr['_focus_keywords'] = set()
        
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
        scores = []
        
        # Performance optimization: pre-compute query keywords
        client_problem_keywords = set(w for w in client_problem.lower().split() if len(w) > 2)

        for idx, framework in enumerate(self.cognitive_frameworks):
            score = 0.0
            
            # Match by problem category/topic
            if client_topic and framework.get("problem_category") == client_topic:
                score += 0.3
            
            # Match by automatic thoughts
            if self._text_similarity(client_problem_keywords, framework.get("_event_keywords", framework.get("event", ""))):
                score += 0.25
            
            # Match by cognitive patterns
            if cognitive_patterns:
                framework_patterns = framework.get("cognitive_patterns", [])
                if isinstance(framework_patterns, list):
                    matched_patterns = set(cognitive_patterns) & set(framework_patterns)
                else:
                    matched_patterns = set(cognitive_patterns) & set([framework_patterns])
                if matched_patterns:
                    score += 0.25 * (len(matched_patterns) / len(cognitive_patterns))
            
            # Match by keywords in problem
            if self._keyword_overlap(client_problem_keywords, framework):
                score += 0.2
            
            if score > 0:
                scores.append((idx, score, framework))
        
        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)

        # Bug Prevention / Architecture: use .copy() and remove internal keys
        results = []
        for _, score, fw in scores[:top_k]:
            fw_copy = fw.copy()
            fw_copy.pop('_event_keywords', None)
            fw_copy.pop('_framework_keywords', None)
            results.append(fw_copy)
        
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
        
        # Performance optimization: pre-compute query keywords
        client_problem_keywords = set(w for w in client_problem.lower().split() if len(w) > 2)
        if client_topic:
            client_topic_keywords = set(w for w in client_topic.lower().split() if len(w) > 2)
        else:
            client_topic_keywords = set()

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
            if self._text_similarity(
                client_problem_keywords,
                strategy.get('_theme_rationale_keywords', f"{strategy.get('theme', '')} {strategy.get('rationale', '')}")
            ):
                score += 0.25
            
            # Match by problem category
            if client_topic and self._text_similarity(client_topic_keywords, strategy.get('_theme_keywords', strategy.get("theme", ""))):
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

        # Bug Prevention / Architecture: use .copy() and remove internal keys
        results = []
        for _, score, strategy in scores[:top_k]:
            strategy_copy = strategy.copy()
            strategy_copy.pop('_theme_rationale_keywords', None)
            strategy_copy.pop('_theme_keywords', None)
            results.append(strategy_copy)
        
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
        
        stage_mapping = {
            "initial_conceptualization": 1,
            "core_intervention": 2,
            "consolidation": 3,
        }
        target_stage = stage_mapping.get(therapy_stage, 2)
        
        # Performance optimization: pre-compute query keywords
        client_problem_keywords = set(w for w in client_problem.lower().split() if len(w) > 2)
        client_problem_keywords_no_filter = set(client_problem.lower().split())
        if client_topic:
            client_topic_keywords = set(w for w in client_topic.lower().split() if len(w) > 2)
        else:
            client_topic_keywords = set()

        for idx, progress in enumerate(self.therapy_progress):
            score = 0.0
            
            # Match by stage
            progress_stage = progress.get("stage_number", 2)
            if abs(target_stage - progress_stage) <= 1:
                score += 0.3
            
            # Match by focus areas
            focus_areas = progress.get("focus_areas", [])
            if focus_areas:
                if '_focus_keywords' in progress:
                    focus_keywords = progress['_focus_keywords']
                else:
                    focus_keywords = set(str(f).lower() for f in focus_areas)
                overlap = client_problem_keywords_no_filter & focus_keywords
                if overlap:
                    score += 0.4
            
            # Match by topic
            if client_topic and self._text_similarity(client_topic_keywords, progress.get('_stage_name_keywords', progress.get("stage_name", ""))):
                score += 0.2
            
            # Match by therapy content
            if self._text_similarity(client_problem_keywords, progress.get('_content_keywords', progress.get("therapy_content", ""))):
                score += 0.1
            
            if score > 0:
                scores.append((idx, score, progress))
        
        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)

        # Bug Prevention / Architecture: use .copy() and remove internal keys
        results = []
        for _, score, progress in scores[:top_k]:
            progress_copy = progress.copy()
            progress_copy.pop('_stage_name_keywords', None)
            progress_copy.pop('_content_keywords', None)
            progress_copy.pop('_focus_keywords', None)
            results.append(progress_copy)
        
        # Store relevance scores
        for idx, score, _ in scores[:top_k]:
            relevance_scores[f"example_{idx}"] = score
        
        return results
    
    def _text_similarity(self, text1, text2) -> bool:
        """Simple text similarity check based on keyword overlap"""
        # Performance optimization: support pre-computed sets
        if isinstance(text1, set):
            keywords1 = text1
        else:
            if not text1: return False
            if not isinstance(text1, str): text1 = str(text1)
            keywords1 = set(w for w in text1.lower().split() if len(w) > 2)

        if isinstance(text2, set):
            keywords2 = text2
        else:
            if not text2: return False
            if not isinstance(text2, str): text2 = str(text2)
            keywords2 = set(w for w in text2.lower().split() if len(w) > 2)

        if not keywords1 or not keywords2:
            return False

        # Check for overlap
        overlap = keywords1 & keywords2
        return len(overlap) > 0
    
    def _keyword_overlap(self, problem, framework: Dict[str, Any]) -> bool:
        """Check keyword overlap between problem and framework"""
        if isinstance(problem, set):
            problem_keywords = problem
        else:
            problem_keywords = set(w.lower() for w in str(problem).split() if len(w) > 2)
        
        # Performance optimization: use pre-computed keywords if available
        if '_framework_keywords' in framework:
            framework_keywords = framework['_framework_keywords']
        else:
            auto_thoughts = framework.get("automatic_thoughts", [])
            comp_strategies = framework.get("compensatory_strategies", [])
            auto_thoughts_str = " ".join(str(x) for x in auto_thoughts) if isinstance(auto_thoughts, list) else str(auto_thoughts)
            comp_strategies_str = " ".join(str(x) for x in comp_strategies) if isinstance(comp_strategies, list) else str(comp_strategies)
            framework_text = " ".join([str(framework.get("event", "")), auto_thoughts_str, comp_strategies_str]).lower()
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
