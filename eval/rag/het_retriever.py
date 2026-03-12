"""
HET (Humanistic-Existential Therapy) RAG Retriever

Multi-dimensional retrieval system for HET cases.
Retrieves relevant self-concepts, existential themes, and client-centered strategies.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class RetrievalResult:
    """Result of RAG retrieval"""
    self_concepts: List[Dict]
    existential_themes: List[Dict]
    strategies: List[Dict]
    relevance_scores: Dict


class HETRetriever:
    """RAG retriever for HET knowledge base"""
    
    def __init__(self, knowledge_base_dir: str):
        self.kb_dir = Path(knowledge_base_dir)
        self.self_concepts = []
        self.existential_themes = []
        self.strategies = []
        
        self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> None:
        """Load all knowledge base files"""
        # Load self-concepts
        self_concepts_file = self.kb_dir / "het_self_concepts.json"
        if self_concepts_file.exists():
            with open(self_concepts_file, 'r', encoding='utf-8') as f:
                self.self_concepts = json.load(f)
        
        # Load existential themes
        existential_file = self.kb_dir / "het_existential_themes.json"
        if existential_file.exists():
            with open(existential_file, 'r', encoding='utf-8') as f:
                self.existential_themes = json.load(f)
        
        # Load strategies
        strategies_file = self.kb_dir / "het_client_centered_strategies.json"
        if strategies_file.exists():
            with open(strategies_file, 'r', encoding='utf-8') as f:
                self.strategies = json.load(f)

        # Precompute for self concepts
        for c in self.self_concepts:
            c["_current_self_perception_keywords"] = set(re.findall(r'\w+', c.get('current_self_perception', '').lower()))
            c["_growth_potential_keywords"] = set(re.findall(r'\w+', c.get('growth_potential', '').lower()))

        # Precompute for existential themes
        for t in self.existential_themes:
            manifestations = t.get('manifestations', [])
            t["_manifestation_keywords"] = [set(re.findall(r'\w+', m.lower())) for m in manifestations]

        # Precompute for strategies
        for s in self.strategies:
            s["_situation_keywords"] = set(re.findall(r'\w+', s.get('situation', '').lower()))
            s["_counselor_approach_keywords"] = set(re.findall(r'\w+', s.get('counselor_approach', '').lower()))

    def _clean_result(self, result: Dict) -> Dict:
        """Remove precomputed keys starting with _ and ending with _keywords"""
        return {k: v for k, v in result.items() if not (k.startswith('_') and k.endswith('_keywords'))}
    
    def retrieve(
        self,
        client_problem: str,
        self_perception: Optional[str] = None,
        existential_concern: Optional[str] = None,
        top_k: int = 3
    ) -> RetrievalResult:
        """
        Retrieve relevant HET knowledge.
        
        Args:
            client_problem: Client's presenting problem
            self_perception: Client's self-perception/identity issue
            existential_concern: Existential theme (meaning, authenticity, etc.)
            top_k: Number of top results to return per category
        """
        
        # Retrieve self-concept frameworks
        self_concept_results = self._retrieve_self_concepts(
            client_problem, self_perception, top_k
        )
        
        # Retrieve existential themes
        existential_results = self._retrieve_existential_themes(
            existential_concern or client_problem, top_k
        )
        
        # Retrieve strategies
        strategy_results = self._retrieve_strategies(
            client_problem, self_perception, top_k
        )
        
        return RetrievalResult(
            self_concepts=self_concept_results,
            existential_themes=existential_results,
            strategies=strategy_results,
            relevance_scores={
                'self_concepts': [r.get('relevance_score', 0) for r in self_concept_results],
                'existential_themes': [r.get('relevance_score', 0) for r in existential_results],
                'strategies': [r.get('relevance_score', 0) for r in strategy_results],
            }
        )
    
    def _retrieve_self_concepts(
        self,
        client_problem: str,
        self_perception: Optional[str],
        top_k: int
    ) -> List[Dict]:
        """Retrieve relevant self-concept frameworks"""
        query = " ".join(filter(None, [client_problem, self_perception]))
        query_words = set(re.findall(r'\w+', query.lower()))
        
        scored_results = []
        for concept in self.self_concepts:
            score = 0.0
            
            # Topic match
            current_perception_kws = concept.get("_current_self_perception_keywords")
            if current_perception_kws is not None:
                problem_sim = self._text_similarity(query_words, current_perception_kws)
            else:
                problem_sim = self._text_similarity(query, concept.get('current_self_perception', ''))
            score += problem_sim * 0.4
            
            # Growth potential match
            growth_potential_kws = concept.get("_growth_potential_keywords")
            if growth_potential_kws is not None:
                growth_sim = self._text_similarity(query_words, growth_potential_kws)
            else:
                growth_sim = self._text_similarity(query, concept.get('growth_potential', ''))
            score += growth_sim * 0.3
            
            # Incongruence relevance
            incongruence = concept.get('self_incongruence', [])
            if incongruence and any(kw in query for kw in ['矛盾', '冲突', '不一致']):
                score += 0.2
            
            concept_res = concept.copy()
            concept_res['relevance_score'] = score
            scored_results.append((score, concept_res))
        
        # Sort by score and return top-k
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [self._clean_result(r[1]) for r in scored_results[:top_k]]
    
    def _retrieve_existential_themes(
        self,
        existential_concern: str,
        top_k: int
    ) -> List[Dict]:
        """Retrieve relevant existential themes"""
        scored_results = []
        query_words = set(re.findall(r'\w+', existential_concern.lower()))
        
        theme_keywords = {
            '无意义': ['无意义', '意义'],
            '孤独': ['孤独', '隔离', '融入'],
            '真实性': ['真诚', '真实', '不真实'],
            '自由': ['选择', '自由', '责任'],
        }
        
        for theme in self.existential_themes:
            score = 0.0
            theme_type = theme.get('theme_type', '')
            
            # Theme match
            for kw_set, kws in theme_keywords.items():
                if any(kw in existential_concern for kw in kws):
                    if kw_set == theme_type:
                        score += 0.5
            
            # Manifestation match
            manifestation_kws_list = theme.get("_manifestation_keywords")
            if manifestation_kws_list is not None:
                for manif_kws in manifestation_kws_list:
                    if self._text_similarity(query_words, manif_kws) > 0.3:
                        score += 0.25
            else:
                manifestations = theme.get('manifestations', [])
                for manif in manifestations:
                    if self._text_similarity(existential_concern, manif) > 0.3:
                        score += 0.25
            
            theme_res = theme.copy()
            theme_res['relevance_score'] = score
            scored_results.append((score, theme_res))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [self._clean_result(r[1]) for r in scored_results[:top_k]]
    
    def _retrieve_strategies(
        self,
        client_problem: str,
        self_perception: Optional[str],
        top_k: int
    ) -> List[Dict]:
        """Retrieve relevant client-centered strategies"""
        query = " ".join(filter(None, [client_problem, self_perception]))
        query_words = set(re.findall(r'\w+', query.lower()))
        
        scored_results = []
        for strategy in self.strategies:
            score = 0.0
            
            # Situation match
            situation_kws = strategy.get("_situation_keywords")
            if situation_kws is not None:
                situation_sim = self._text_similarity(query_words, situation_kws)
            else:
                situation = strategy.get('situation', '')
                situation_sim = self._text_similarity(query, situation)
            score += situation_sim * 0.35
            
            # Strategy type match (prefer unconditional positive regard, empathy)
            strategy_type = strategy.get('strategy_type', '')
            if strategy_type in ['Empathic Understanding', 'Unconditional Positive Regard']:
                score += 0.15
            
            # Approach match
            approach_kws = strategy.get("_counselor_approach_keywords")
            if approach_kws is not None:
                approach_sim = self._text_similarity(query_words, approach_kws)
            else:
                approach = strategy.get('counselor_approach', '')
                approach_sim = self._text_similarity(query, approach)
            score += approach_sim * 0.25
            
            # Expected outcome (growth-oriented)
            outcome = strategy.get('expected_outcome', '')
            if any(kw in outcome for kw in ['自我', '理解', '成长', '认识']):
                score += 0.15
            
            strategy_res = strategy.copy()
            strategy_res['relevance_score'] = score
            scored_results.append((score, strategy_res))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [self._clean_result(r[1]) for r in scored_results[:top_k]]
    
    def _text_similarity(self, text_or_words1, text_or_words2) -> float:
        """Simple keyword overlap similarity"""
        if not text_or_words1 or not text_or_words2:
            return 0.0
        
        if isinstance(text_or_words1, str):
            words1 = set(re.findall(r'\w+', text_or_words1.lower()))
        else:
            words1 = text_or_words1

        if isinstance(text_or_words2, str):
            words2 = set(re.findall(r'\w+', text_or_words2.lower()))
        else:
            words2 = text_or_words2
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return overlap / total if total > 0 else 0.0
