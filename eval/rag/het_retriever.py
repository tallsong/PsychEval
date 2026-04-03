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
    
    def _clean_result(self, item: Dict) -> Dict:
        cleaned = item.copy()
        keys_to_remove = [k for k in cleaned.keys() if k.startswith('_')]
        for k in keys_to_remove:
            del cleaned[k]
        return cleaned

    def _load_knowledge_base(self) -> None:
        """Load all knowledge base files"""
        # Load self-concepts
        self_concepts_file = self.kb_dir / "het_self_concepts.json"
        if self_concepts_file.exists():
            with open(self_concepts_file, 'r', encoding='utf-8') as f:
                self.self_concepts = json.load(f)
                for c in self.self_concepts:
                    csp = str(c.get('current_self_perception', ''))
                    c['_perception_keywords'] = set(re.findall(r'\w+', csp.lower()))
                    gp = str(c.get('growth_potential', ''))
                    c['_growth_keywords'] = set(re.findall(r'\w+', gp.lower()))
        
        # Load existential themes
        existential_file = self.kb_dir / "het_existential_themes.json"
        if existential_file.exists():
            with open(existential_file, 'r', encoding='utf-8') as f:
                self.existential_themes = json.load(f)
                for t in self.existential_themes:
                    t['_manifestation_keywords_list'] = []
                    for m in t.get('manifestations', []):
                        t['_manifestation_keywords_list'].append(set(re.findall(r'\w+', str(m).lower())))
        
        # Load strategies
        strategies_file = self.kb_dir / "het_client_centered_strategies.json"
        if strategies_file.exists():
            with open(strategies_file, 'r', encoding='utf-8') as f:
                self.strategies = json.load(f)
                for s in self.strategies:
                    sit = str(s.get('situation', ''))
                    s['_situation_keywords'] = set(re.findall(r'\w+', sit.lower()))
                    app = str(s.get('counselor_approach', ''))
                    s['_approach_keywords'] = set(re.findall(r'\w+', app.lower()))
    
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
            
            c_copy = self._clean_result(concept)
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
            if '_manifestation_keywords_list' in theme:
                for m_kws in theme['_manifestation_keywords_list']:
                    if self._text_similarity(query_words, m_kws) > 0.3:
                        score += 0.25
            else:
                manifestations = theme.get('manifestations', [])
                for manif in manifestations:
                    if self._text_similarity(query_words, manif) > 0.3:
                        score += 0.25
            
            t_copy = self._clean_result(theme)
            t_copy['relevance_score'] = score
            scored_results.append((score, t_copy))
        
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
        query_words = set(re.findall(r'\w+', query.lower()))
        
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
            
            s_copy = self._clean_result(strategy)
            s_copy['relevance_score'] = score
            scored_results.append((score, s_copy))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in scored_results[:top_k]]
    
    def _text_similarity(self, text1: str, text2) -> float:
        """Simple keyword overlap similarity"""
        if not text1 or not text2:
            return 0.0
        
        if isinstance(text1, set):
            words1 = text1
        else:
            words1 = set(re.findall(r'\w+', str(text1).lower()))

        if isinstance(text2, set):
            words2 = text2
        else:
            words2 = set(re.findall(r'\w+', str(text2).lower()))
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return overlap / total if total > 0 else 0.0
