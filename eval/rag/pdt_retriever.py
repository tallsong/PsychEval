"""
PDT (Psychodynamic Therapy) RAG Retriever

Multi-dimensional retrieval system for PDT cases.
Retrieves relevant core conflicts, object relations, unconscious patterns, and interventions.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class RetrievalResult:
    """Result of RAG retrieval for PDT"""
    core_conflicts: List[Dict]
    object_relations: List[Dict]
    unconscious_patterns: List[Dict]
    interventions: List[Dict]
    relevance_scores: Dict


class PDTRetriever:
    """RAG retriever for PDT knowledge base"""
    
    def __init__(self, knowledge_base_dir: str):
        self.kb_dir = Path(knowledge_base_dir)
        self.core_conflicts = []
        self.object_relations = []
        self.unconscious_patterns = []
        self.interventions = []
        
        self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> None:
        """Load all knowledge base files"""
        # Load core conflicts
        conflicts_file = self.kb_dir / "pdt_core_conflicts.json"
        if conflicts_file.exists():
            with open(conflicts_file, 'r', encoding='utf-8') as f:
                self.core_conflicts = json.load(f)
                for conflict in self.core_conflicts:
                    conflict['_wish_keywords'] = set(re.findall(r'\w+', conflict.get('wish', '').lower()))
                    conflict['_fear_keywords'] = set(re.findall(r'\w+', conflict.get('fear', '').lower()))
                    conflict['_behavior_keywords'] = [
                        set(re.findall(r'\w+', b.lower())) for b in conflict.get('behavioral_manifestations', [])
                    ]
        
        # Load object relations
        relations_file = self.kb_dir / "pdt_object_relations.json"
        if relations_file.exists():
            with open(relations_file, 'r', encoding='utf-8') as f:
                self.object_relations = json.load(f)
                for relation in self.object_relations:
                    relation['_self_rep_keywords'] = set(re.findall(r'\w+', relation.get('self_representation', '').lower()))
                    relation['_obj_rep_keywords'] = set(re.findall(r'\w+', relation.get('object_representation', '').lower()))
                    relation['_pattern_keywords'] = set(re.findall(r'\w+', relation.get('relational_pattern', '').lower()))
        
        # Load unconscious patterns
        patterns_file = self.kb_dir / "pdt_unconscious_patterns.json"
        if patterns_file.exists():
            with open(patterns_file, 'r', encoding='utf-8') as f:
                self.unconscious_patterns = json.load(f)
                for pattern in self.unconscious_patterns:
                    pattern['_manifestation_keywords'] = set(re.findall(r'\w+', pattern.get('current_manifestation', '').lower()))
        
        # Load interventions
        interventions_file = self.kb_dir / "pdt_psychodynamic_interventions.json"
        if interventions_file.exists():
            with open(interventions_file, 'r', encoding='utf-8') as f:
                self.interventions = json.load(f)
                for intervention in self.interventions:
                    intervention['_situation_keywords'] = set(re.findall(r'\w+', intervention.get('situation', '').lower()))

    def _clean_result(self, doc: Dict) -> Dict:
        """Return a copy of the document without internal _keywords fields"""
        cleaned = doc.copy()
        for k in list(cleaned.keys()):
            if k.startswith('_') and k.endswith('_keywords'):
                del cleaned[k]
        return cleaned
    
    def retrieve(
        self,
        client_problem: str,
        relational_patterns: Optional[List[str]] = None,
        defensive_behaviors: Optional[List[str]] = None,
        top_k: int = 3
    ) -> RetrievalResult:
        """
        Retrieve relevant PDT knowledge.
        
        Args:
            client_problem: Client's presenting problem/symptom
            relational_patterns: Identified relational patterns
            defensive_behaviors: Observed defense mechanisms
            top_k: Number of top results to return per category
        """
        
        # Retrieve core conflicts
        conflict_results = self._retrieve_core_conflicts(
            client_problem, relational_patterns or [], top_k
        )
        
        # Retrieve object relations
        relation_results = self._retrieve_object_relations(
            client_problem, relational_patterns or [], top_k
        )
        
        # Retrieve unconscious patterns
        pattern_results = self._retrieve_unconscious_patterns(
            client_problem, top_k
        )
        
        # Retrieve interventions
        intervention_results = self._retrieve_interventions(
            client_problem, defensive_behaviors or [], top_k
        )
        
        return RetrievalResult(
            core_conflicts=conflict_results,
            object_relations=relation_results,
            unconscious_patterns=pattern_results,
            interventions=intervention_results,
            relevance_scores={
                'core_conflicts': [r.get('relevance_score', 0) for r in conflict_results],
                'object_relations': [r.get('relevance_score', 0) for r in relation_results],
                'unconscious_patterns': [r.get('relevance_score', 0) for r in pattern_results],
                'interventions': [r.get('relevance_score', 0) for r in intervention_results],
            }
        )
    
    def _retrieve_core_conflicts(
        self,
        client_problem: str,
        relational_patterns: List[str],
        top_k: int
    ) -> List[Dict]:
        """Retrieve relevant core conflict patterns"""
        scored_results = []
        query_words = set(re.findall(r'\w+', client_problem.lower()))
        
        for conflict in self.core_conflicts:
            score = 0.0
            
            # Problem match (fear, wish)
            wish_sim = self._text_similarity(query_words, conflict.get('_wish_keywords', set()))
            fear_sim = self._text_similarity(query_words, conflict.get('_fear_keywords', set()))
            score += max(wish_sim, fear_sim) * 0.4
            
            # Behavioral manifestation match
            for behavior_words in conflict.get('_behavior_keywords', []):
                if self._text_similarity(query_words, behavior_words) > 0.2:
                    score += 0.15
            
            # Defense mechanism relevance
            defenses = conflict.get('defense_mechanisms', [])
            if defenses and any('防御' in p or '保护' in p for p in relational_patterns):
                score += 0.15
            
            conflict_copy = self._clean_result(conflict)
            conflict_copy['relevance_score'] = score
            scored_results.append((score, conflict_copy))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in scored_results[:top_k]]
    
    def _retrieve_object_relations(
        self,
        client_problem: str,
        relational_patterns: List[str],
        top_k: int
    ) -> List[Dict]:
        """Retrieve relevant object relations"""
        scored_results = []
        query_words = set(re.findall(r'\w+', client_problem.lower()))
        
        for relation in self.object_relations:
            score = 0.0
            
            # Self representation match
            self_sim = self._text_similarity(query_words, relation.get('_self_rep_keywords', set()))
            score += self_sim * 0.3
            
            # Object representation match (others)
            obj_sim = self._text_similarity(query_words, relation.get('_obj_rep_keywords', set()))
            score += obj_sim * 0.3
            
            # Linking affect relevance
            linking_affect = relation.get('linking_affect', '')
            if any(emotion in linking_affect for emotion in ['被抛弃', '失望', '怨恨', '空虚']):
                if any(emotion in client_problem for emotion in ['抛弃', '分离', '失望']):
                    score += 0.2
            
            # Relational pattern match
            if relational_patterns:
                for p in relational_patterns:
                    p_words = set(re.findall(r'\w+', p.lower()))
                    if self._text_similarity(p_words, relation.get('_pattern_keywords', set())) > 0.2:
                        score += 0.15
            
            relation_copy = self._clean_result(relation)
            relation_copy['relevance_score'] = score
            scored_results.append((score, relation_copy))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in scored_results[:top_k]]
    
    def _retrieve_unconscious_patterns(
        self,
        client_problem: str,
        top_k: int
    ) -> List[Dict]:
        """Retrieve relevant unconscious patterns"""
        scored_results = []
        query_words = set(re.findall(r'\w+', client_problem.lower()))
        
        pattern_keywords = {
            'Abandonment': ['离开', '抛弃', '分离', '空虚'],
            'Isolation': ['孤独', '隔离', '连接'],
            'Internal Emptiness': ['空虚', '无意义'],
            'Ambivalent': ['矛盾', '冲突', '爱恨'],
        }
        
        for pattern in self.unconscious_patterns:
            score = 0.0
            
            # Pattern theme match
            pattern_type = pattern.get('pattern_theme', '')
            for theme, keywords in pattern_keywords.items():
                if theme in pattern_type:
                    if any(kw in client_problem for kw in keywords):
                        score += 0.35
            
            # Current manifestation match
            manif_sim = self._text_similarity(query_words, pattern.get('_manifestation_keywords', set()))
            score += manif_sim * 0.3
            
            # Early origin relevance (developmental sensitivity)
            origin = pattern.get('early_origin', '')
            if any(kw in origin for kw in ['分离', '早期', '童年']):
                score += 0.15
            
            # Relational impact
            impact = pattern.get('relational_impact', '')
            if '关系' in impact:
                score += 0.15
            
            pattern_copy = self._clean_result(pattern)
            pattern_copy['relevance_score'] = score
            scored_results.append((score, pattern_copy))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in scored_results[:top_k]]
    
    def _retrieve_interventions(
        self,
        client_problem: str,
        defensive_behaviors: List[str],
        top_k: int
    ) -> List[Dict]:
        """Retrieve relevant psychodynamic interventions"""
        scored_results = []
        query_words = set(re.findall(r'\w+', client_problem.lower()))
        
        for intervention in self.interventions:
            score = 0.0
            
            # Situation match
            situation_sim = self._text_similarity(query_words, intervention.get('_situation_keywords', set()))
            score += situation_sim * 0.35
            
            # Intervention type appropriateness
            int_type = intervention.get('intervention_type', '')
            # Prefer interpretation and insight-focused for PDT
            if int_type in ['Interpretation', 'Connection Making']:
                score += 0.15
            
            # Targeted conflict relevance
            targeted = intervention.get('targeted_conflict', '')
            if any(kw in targeted for kw in ['无意识', '冲突', '防御']):
                score += 0.15
            
            # Therapist response depth
            response = intervention.get('therapist_response', '')
            if any(kw in response for kw in ['似乎', '可能', '潜在', '无意识']):
                score += 0.15
            
            intervention_copy = self._clean_result(intervention)
            intervention_copy['relevance_score'] = score
            scored_results.append((score, intervention_copy))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in scored_results[:top_k]]
    
    def _text_similarity(self, words1: set, words2: set) -> float:
        """Simple keyword overlap similarity"""
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return overlap / total if total > 0 else 0.0
