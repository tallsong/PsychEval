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

        # Precompute keywords for faster similarity calculation
        for cc in self.core_conflicts:
            wish = str(cc.get('wish', ''))
            cc['_wish_keywords'] = set(re.findall(r'\w+', wish.lower()))
            fear = str(cc.get('fear', ''))
            cc['_fear_keywords'] = set(re.findall(r'\w+', fear.lower()))
            cc['_behavior_keywords_list'] = []
            for behavior in cc.get('behavioral_manifestations', []):
                cc['_behavior_keywords_list'].append(set(re.findall(r'\w+', str(behavior).lower())))

        for or_ in self.object_relations:
            self_rep = str(or_.get('self_representation', ''))
            or_['_self_rep_keywords'] = set(re.findall(r'\w+', self_rep.lower()))
            obj_rep = str(or_.get('object_representation', ''))
            or_['_obj_rep_keywords'] = set(re.findall(r'\w+', obj_rep.lower()))
            pattern = str(or_.get('relational_pattern', ''))
            or_['_pattern_keywords'] = set(re.findall(r'\w+', pattern.lower()))

        for up in self.unconscious_patterns:
            manifestation = str(up.get('current_manifestation', ''))
            up['_manifestation_keywords'] = set(re.findall(r'\w+', manifestation.lower()))

        for intv in self.interventions:
            situation = str(intv.get('situation', ''))
            intv['_situation_keywords'] = set(re.findall(r'\w+', situation.lower()))

    def _clean_result(self, item: Dict) -> Dict:
        """Remove precomputed internal keys to prevent JSON serialization errors."""
        cleaned = item.copy()
        for key in list(cleaned.keys()):
            if key.startswith("_") and key.endswith("_keywords") or key.endswith("_keywords_list"):
                del cleaned[key]
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
            core_conflicts=[self._clean_result(r) for r in conflict_results],
            object_relations=[self._clean_result(r) for r in relation_results],
            unconscious_patterns=[self._clean_result(r) for r in pattern_results],
            interventions=[self._clean_result(r) for r in intervention_results],
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
        problem_words = set(re.findall(r'\w+', client_problem.lower()))
        scored_results = []
        
        for conflict in self.core_conflicts:
            score = 0.0
            
            # Problem match (fear, wish)
            wish_words = conflict.get('_wish_keywords', set())
            fear_words = conflict.get('_fear_keywords', set())
            
            wish_sim = len(problem_words & wish_words) / len(problem_words | wish_words) if (problem_words | wish_words) else 0.0
            fear_sim = len(problem_words & fear_words) / len(problem_words | fear_words) if (problem_words | fear_words) else 0.0
            score += max(wish_sim, fear_sim) * 0.4
            
            # Behavioral manifestation match
            behavior_keywords_list = conflict.get('_behavior_keywords_list', [])
            for behavior_words in behavior_keywords_list:
                sim = len(problem_words & behavior_words) / len(problem_words | behavior_words) if (problem_words | behavior_words) else 0.0
                if sim > 0.2:
                    score += 0.15
            
            # Defense mechanism relevance
            defenses = conflict.get('defense_mechanisms', [])
            if defenses and any('防御' in p or '保护' in p for p in relational_patterns):
                score += 0.15
            
            conflict_copy = conflict.copy()
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
        problem_words = set(re.findall(r'\w+', client_problem.lower()))
        scored_results = []
        
        for relation in self.object_relations:
            score = 0.0
            
            # Self representation match
            self_rep_words = relation.get('_self_rep_keywords', set())
            self_sim = len(problem_words & self_rep_words) / len(problem_words | self_rep_words) if (problem_words | self_rep_words) else 0.0
            score += self_sim * 0.3
            
            # Object representation match (others)
            obj_rep_words = relation.get('_obj_rep_keywords', set())
            obj_sim = len(problem_words & obj_rep_words) / len(problem_words | obj_rep_words) if (problem_words | obj_rep_words) else 0.0
            score += obj_sim * 0.3
            
            # Linking affect relevance
            linking_affect = relation.get('linking_affect', '')
            if any(emotion in linking_affect for emotion in ['被抛弃', '失望', '怨恨', '空虚']):
                if any(emotion in client_problem for emotion in ['抛弃', '分离', '失望']):
                    score += 0.2
            
            # Relational pattern match
            pattern_words = relation.get('_pattern_keywords', set())
            if relational_patterns:
                for p in relational_patterns:
                    p_words = set(re.findall(r'\w+', str(p).lower()))
                    sim = len(p_words & pattern_words) / len(p_words | pattern_words) if (p_words | pattern_words) else 0.0
                    if sim > 0.2:
                        score += 0.15
            
            relation_copy = relation.copy()
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
        problem_words = set(re.findall(r'\w+', client_problem.lower()))
        scored_results = []
        
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
            manif_words = pattern.get('_manifestation_keywords', set())
            manif_sim = len(problem_words & manif_words) / len(problem_words | manif_words) if (problem_words | manif_words) else 0.0
            score += manif_sim * 0.3
            
            # Early origin relevance (developmental sensitivity)
            origin = pattern.get('early_origin', '')
            if any(kw in origin for kw in ['分离', '早期', '童年']):
                score += 0.15
            
            # Relational impact
            impact = pattern.get('relational_impact', '')
            if '关系' in impact:
                score += 0.15
            
            pattern_copy = pattern.copy()
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
        problem_words = set(re.findall(r'\w+', client_problem.lower()))
        scored_results = []
        
        for intervention in self.interventions:
            score = 0.0
            
            # Situation match
            situation_words = intervention.get('_situation_keywords', set())
            situation_sim = len(problem_words & situation_words) / len(problem_words | situation_words) if (problem_words | situation_words) else 0.0
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
            
            intervention_copy = intervention.copy()
            intervention_copy['relevance_score'] = score
            scored_results.append((score, intervention_copy))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in scored_results[:top_k]]
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple keyword overlap similarity"""
        if not text1 or not text2:
            return 0.0
        
        if isinstance(text1, list):
            text1 = ' '.join(str(t) for t in text1)
        if isinstance(text2, list):
            text2 = ' '.join(str(t) for t in text2)
        
        words1 = set(re.findall(r'\w+', str(text1).lower()))
        words2 = set(re.findall(r'\w+', str(text2).lower()))
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return overlap / total if total > 0 else 0.0
