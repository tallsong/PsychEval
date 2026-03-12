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
        
        # Load object relations
        relations_file = self.kb_dir / "pdt_object_relations.json"
        if relations_file.exists():
            with open(relations_file, 'r', encoding='utf-8') as f:
                self.object_relations = json.load(f)
        
        # Load unconscious patterns
        patterns_file = self.kb_dir / "pdt_unconscious_patterns.json"
        if patterns_file.exists():
            with open(patterns_file, 'r', encoding='utf-8') as f:
                self.unconscious_patterns = json.load(f)
        
        # Load interventions
        interventions_file = self.kb_dir / "pdt_psychodynamic_interventions.json"
        if interventions_file.exists():
            with open(interventions_file, 'r', encoding='utf-8') as f:
                self.interventions = json.load(f)

        # Precompute for core conflicts
        for c in self.core_conflicts:
            c["_wish_keywords"] = set(re.findall(r'\w+', str(c.get('wish', '')).lower()))
            c["_fear_keywords"] = set(re.findall(r'\w+', str(c.get('fear', '')).lower()))
            behaviors = c.get('behavioral_manifestations', [])
            c["_behavior_keywords"] = [set(re.findall(r'\w+', str(b).lower())) for b in behaviors]

        # Precompute for object relations
        for r in self.object_relations:
            r["_self_rep_keywords"] = set(re.findall(r'\w+', str(r.get('self_representation', '')).lower()))
            r["_obj_rep_keywords"] = set(re.findall(r'\w+', str(r.get('object_representation', '')).lower()))
            r["_pattern_keywords"] = set(re.findall(r'\w+', str(r.get('relational_pattern', '')).lower()))

        # Precompute for unconscious patterns
        for p in self.unconscious_patterns:
            p["_manifestation_keywords"] = set(re.findall(r'\w+', str(p.get('current_manifestation', '')).lower()))

        # Precompute for interventions
        for i in self.interventions:
            i["_situation_keywords"] = set(re.findall(r'\w+', str(i.get('situation', '')).lower()))

    def _clean_result(self, result: Dict) -> Dict:
        """Remove precomputed keys starting with _ and ending with _keywords"""
        return {k: v for k, v in result.items() if not (k.startswith('_') and k.endswith('_keywords'))}
    
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
        client_problem_words = set(re.findall(r'\w+', str(client_problem).lower()))
        
        for conflict in self.core_conflicts:
            score = 0.0
            
            # Problem match (fear, wish)
            wish_kws = conflict.get("_wish_keywords")
            fear_kws = conflict.get("_fear_keywords")
            
            if wish_kws is not None:
                wish_sim = self._text_similarity(client_problem_words, wish_kws)
            else:
                wish_sim = self._text_similarity(client_problem, conflict.get('wish', ''))

            if fear_kws is not None:
                fear_sim = self._text_similarity(client_problem_words, fear_kws)
            else:
                fear_sim = self._text_similarity(client_problem, conflict.get('fear', ''))

            score += max(wish_sim, fear_sim) * 0.4
            
            # Behavioral manifestation match
            behavior_kws_list = conflict.get("_behavior_keywords")
            if behavior_kws_list is not None:
                for b_kws in behavior_kws_list:
                    if self._text_similarity(client_problem_words, b_kws) > 0.2:
                        score += 0.15
            else:
                behaviors = conflict.get('behavioral_manifestations', [])
                for behavior in behaviors:
                    if self._text_similarity(client_problem, behavior) > 0.2:
                        score += 0.15
            
            # Defense mechanism relevance
            defenses = conflict.get('defense_mechanisms', [])
            if defenses and any('防御' in p or '保护' in p for p in relational_patterns):
                score += 0.15
            
            conflict_res = conflict.copy()
            conflict_res['relevance_score'] = score
            scored_results.append((score, conflict_res))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [self._clean_result(r[1]) for r in scored_results[:top_k]]
    
    def _retrieve_object_relations(
        self,
        client_problem: str,
        relational_patterns: List[str],
        top_k: int
    ) -> List[Dict]:
        """Retrieve relevant object relations"""
        scored_results = []
        client_problem_words = set(re.findall(r'\w+', str(client_problem).lower()))
        
        for relation in self.object_relations:
            score = 0.0
            
            # Self representation match
            self_rep_kws = relation.get("_self_rep_keywords")
            if self_rep_kws is not None:
                self_sim = self._text_similarity(client_problem_words, self_rep_kws)
            else:
                self_sim = self._text_similarity(client_problem, relation.get('self_representation', ''))
            score += self_sim * 0.3
            
            # Object representation match (others)
            obj_rep_kws = relation.get("_obj_rep_keywords")
            if obj_rep_kws is not None:
                obj_sim = self._text_similarity(client_problem_words, obj_rep_kws)
            else:
                obj_sim = self._text_similarity(client_problem, relation.get('object_representation', ''))
            score += obj_sim * 0.3
            
            # Linking affect relevance
            linking_affect = relation.get('linking_affect', '')
            if any(emotion in linking_affect for emotion in ['被抛弃', '失望', '怨恨', '空虚']):
                if any(emotion in client_problem for emotion in ['抛弃', '分离', '失望']):
                    score += 0.2
            
            # Relational pattern match
            pattern_kws = relation.get("_pattern_keywords")
            if relational_patterns:
                for p in relational_patterns:
                    if pattern_kws is not None:
                        if self._text_similarity(p, pattern_kws) > 0.2:
                            score += 0.15
                    else:
                        if self._text_similarity(p, relation.get('relational_pattern', '')) > 0.2:
                            score += 0.15
            
            relation_res = relation.copy()
            relation_res['relevance_score'] = score
            scored_results.append((score, relation_res))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [self._clean_result(r[1]) for r in scored_results[:top_k]]
    
    def _retrieve_unconscious_patterns(
        self,
        client_problem: str,
        top_k: int
    ) -> List[Dict]:
        """Retrieve relevant unconscious patterns"""
        scored_results = []
        client_problem_words = set(re.findall(r'\w+', str(client_problem).lower()))
        
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
            manifestation_kws = pattern.get("_manifestation_keywords")
            if manifestation_kws is not None:
                manif_sim = self._text_similarity(client_problem_words, manifestation_kws)
            else:
                manif_sim = self._text_similarity(client_problem, pattern.get('current_manifestation', ''))
            score += manif_sim * 0.3
            
            # Early origin relevance (developmental sensitivity)
            origin = pattern.get('early_origin', '')
            if any(kw in origin for kw in ['分离', '早期', '童年']):
                score += 0.15
            
            # Relational impact
            impact = pattern.get('relational_impact', '')
            if '关系' in impact:
                score += 0.15
            
            pattern_res = pattern.copy()
            pattern_res['relevance_score'] = score
            scored_results.append((score, pattern_res))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [self._clean_result(r[1]) for r in scored_results[:top_k]]
    
    def _retrieve_interventions(
        self,
        client_problem: str,
        defensive_behaviors: List[str],
        top_k: int
    ) -> List[Dict]:
        """Retrieve relevant psychodynamic interventions"""
        scored_results = []
        client_problem_words = set(re.findall(r'\w+', str(client_problem).lower()))
        
        for intervention in self.interventions:
            score = 0.0
            
            # Situation match
            situation_kws = intervention.get("_situation_keywords")
            if situation_kws is not None:
                situation_sim = self._text_similarity(client_problem_words, situation_kws)
            else:
                situation_sim = self._text_similarity(client_problem, intervention.get('situation', ''))
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
            
            intervention_res = intervention.copy()
            intervention_res['relevance_score'] = score
            scored_results.append((score, intervention_res))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [self._clean_result(r[1]) for r in scored_results[:top_k]]
    
    def _text_similarity(self, text_or_words1, text_or_words2) -> float:
        """Simple keyword overlap similarity"""
        if not text_or_words1 or not text_or_words2:
            return 0.0
        
        if isinstance(text_or_words1, str) or isinstance(text_or_words1, list):
            if isinstance(text_or_words1, list):
                text_or_words1 = ' '.join(str(t) for t in text_or_words1)
            words1 = set(re.findall(r'\w+', str(text_or_words1).lower()))
        else:
            words1 = text_or_words1

        if isinstance(text_or_words2, str) or isinstance(text_or_words2, list):
            if isinstance(text_or_words2, list):
                text_or_words2 = ' '.join(str(t) for t in text_or_words2)
            words2 = set(re.findall(r'\w+', str(text_or_words2).lower()))
        else:
            words2 = text_or_words2
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return overlap / total if total > 0 else 0.0
