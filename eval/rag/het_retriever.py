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

        # Pre-compute keyword sets for O(1) retrieval
        for concept in self.self_concepts:
            concept['_perception_keywords'] = set(re.findall(r'\w+', str(concept.get('current_self_perception', '')).lower()))
            concept['_growth_keywords'] = set(re.findall(r'\w+', str(concept.get('growth_potential', '')).lower()))

        for theme in self.existential_themes:
            manifestations = theme.get('manifestations', [])
            theme['_manifestation_keywords'] = [set(re.findall(r'\w+', str(m).lower())) for m in manifestations]

        for strategy in self.strategies:
            strategy['_situation_keywords'] = set(re.findall(r'\w+', str(strategy.get('situation', '')).lower()))
            strategy['_approach_keywords'] = set(re.findall(r'\w+', str(strategy.get('counselor_approach', '')).lower()))
    
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
        
        scored_results = []
        query_kws = set(re.findall(r'\w+', str(query).lower()))
        for concept in self.self_concepts:
            score = 0.0
            
            # Topic match
            perc_kws = concept.get('_perception_keywords', set())
            problem_sim = len(query_kws & perc_kws) / len(query_kws | perc_kws) if query_kws and perc_kws else 0.0
            score += problem_sim * 0.4
            
            # Growth potential match
            growth_kws = concept.get('_growth_keywords', set())
            growth_sim = len(query_kws & growth_kws) / len(query_kws | growth_kws) if query_kws and growth_kws else 0.0
            score += growth_sim * 0.3
            
            # Incongruence relevance
            incongruence = concept.get('self_incongruence', [])
            if incongruence and any(kw in query for kw in ['矛盾', '冲突', '不一致']):
                score += 0.2
            
            concept['relevance_score'] = score
            scored_results.append((score, concept))
        
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
        query_kws = set(re.findall(r'\w+', str(existential_concern).lower()))
        
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
            manifestation_kws_list = theme.get('_manifestation_keywords', [])
            for manif_kws in manifestation_kws_list:
                sim = len(query_kws & manif_kws) / len(query_kws | manif_kws) if query_kws and manif_kws else 0.0
                if sim > 0.3:
                    score += 0.25
            
            theme['relevance_score'] = score
            scored_results.append((score, theme))
        
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
        
        scored_results = []
        query_kws = set(re.findall(r'\w+', str(query).lower()))
        for strategy in self.strategies:
            score = 0.0
            
            # Situation match
            sit_kws = strategy.get('_situation_keywords', set())
            situation_sim = len(query_kws & sit_kws) / len(query_kws | sit_kws) if query_kws and sit_kws else 0.0
            score += situation_sim * 0.35
            
            # Strategy type match (prefer unconditional positive regard, empathy)
            strategy_type = strategy.get('strategy_type', '')
            if strategy_type in ['Empathic Understanding', 'Unconditional Positive Regard']:
                score += 0.15
            
            # Approach match
            app_kws = strategy.get('_approach_keywords', set())
            approach_sim = len(query_kws & app_kws) / len(query_kws | app_kws) if query_kws and app_kws else 0.0
            score += approach_sim * 0.25
            
            # Expected outcome (growth-oriented)
            outcome = strategy.get('expected_outcome', '')
            if any(kw in outcome for kw in ['自我', '理解', '成长', '认识']):
                score += 0.15
            
            strategy['relevance_score'] = score
            scored_results.append((score, strategy))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [self._clean_result(r[1]) for r in scored_results[:top_k]]
    
    def _clean_result(self, item: Dict) -> Dict:
        """Remove internal pre-computed keys before returning"""
        cleaned = item.copy()
        keys_to_remove = [k for k in cleaned.keys() if k.startswith('_') and k.endswith('_keywords')]
        for k in keys_to_remove:
            del cleaned[k]
        return cleaned

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple keyword overlap similarity"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return overlap / total if total > 0 else 0.0
