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
        self._word_pattern = re.compile(r'\w+')
        
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

        # Precompute keyword sets for fast retrieval
        for concept in self.self_concepts:
            concept['_perception_keywords'] = set(self._word_pattern.findall(str(concept.get('current_self_perception', '')).lower()))
            concept['_growth_keywords'] = set(self._word_pattern.findall(str(concept.get('growth_potential', '')).lower()))

        for strategy in self.strategies:
            strategy['_situation_keywords'] = set(self._word_pattern.findall(str(strategy.get('situation', '')).lower()))
            strategy['_approach_keywords'] = set(self._word_pattern.findall(str(strategy.get('counselor_approach', '')).lower()))
    
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
        query_words = set(self._word_pattern.findall(query.lower()))
        
        scored_results = []
        for concept in self.self_concepts:
            score = 0.0
            
            # Topic match
            problem_sim = self._text_similarity(
                query_words,
                concept.get('_perception_keywords', concept.get('current_self_perception', ''))
            )
            score += problem_sim * 0.4
            
            # Growth potential match
            growth_sim = self._text_similarity(query_words, concept.get('_growth_keywords', concept.get('growth_potential', '')))
            score += growth_sim * 0.3
            
            # Incongruence relevance
            incongruence = concept.get('self_incongruence', [])
            if incongruence and any(kw in query for kw in ['矛盾', '冲突', '不一致']):
                score += 0.2
            
            c_copy = concept.copy()
            c_copy.pop('_perception_keywords', None)
            c_copy.pop('_growth_keywords', None)
            c_copy['relevance_score'] = score
            scored_results.append((score, c_copy))
        
        # Sort by score and return top-k
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in scored_results[:top_k]]
    
    def _retrieve_existential_themes(
        self,
        existential_concern: str,
        top_k: int
    ) -> List[Dict]:
        """Retrieve relevant existential themes"""
        scored_results = []
        
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
            manifestations = theme.get('manifestations', [])
            for manif in manifestations:
                if self._text_similarity(existential_concern, manif) > 0.3:
                    score += 0.25
            
            theme['relevance_score'] = score
            scored_results.append((score, theme))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in scored_results[:top_k]]
    
    def _retrieve_strategies(
        self,
        client_problem: str,
        self_perception: Optional[str],
        top_k: int
    ) -> List[Dict]:
        """Retrieve relevant client-centered strategies"""
        query = " ".join(filter(None, [client_problem, self_perception]))
        query_words = set(self._word_pattern.findall(query.lower()))
        
        scored_results = []
        for strategy in self.strategies:
            score = 0.0
            
            # Situation match
            situation_sim = self._text_similarity(query_words, strategy.get('_situation_keywords', strategy.get('situation', '')))
            score += situation_sim * 0.35
            
            # Strategy type match (prefer unconditional positive regard, empathy)
            strategy_type = strategy.get('strategy_type', '')
            if strategy_type in ['Empathic Understanding', 'Unconditional Positive Regard']:
                score += 0.15
            
            # Approach match
            approach_sim = self._text_similarity(query_words, strategy.get('_approach_keywords', strategy.get('counselor_approach', '')))
            score += approach_sim * 0.25
            
            # Expected outcome (growth-oriented)
            outcome = strategy.get('expected_outcome', '')
            if any(kw in outcome for kw in ['自我', '理解', '成长', '认识']):
                score += 0.15
            
            s_copy = strategy.copy()
            s_copy.pop('_situation_keywords', None)
            s_copy.pop('_approach_keywords', None)
            s_copy['relevance_score'] = score
            scored_results.append((score, s_copy))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in scored_results[:top_k]]
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple keyword overlap similarity"""
        if not text1 or not text2:
            return 0.0

        if isinstance(text1, set):
            words1 = text1
        else:
            words1 = set(self._word_pattern.findall(str(text1).lower()))

        if isinstance(text2, set):
            words2 = text2
        else:
            words2 = set(self._word_pattern.findall(str(text2).lower()))

        if not words1 or not words2:
            return 0.0

        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return overlap / total if total > 0 else 0.0
