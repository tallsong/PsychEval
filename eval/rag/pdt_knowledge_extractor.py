"""
PDT (Psychodynamic Therapy) Knowledge Extractor

Extracts knowledge from PDT case files to build RAG knowledge base.
Focus areas:
- Core conflicts (wishes, fears, defenses)
- Object relations and internal representations
- Transference and countertransference patterns
- Defense mechanisms and their manifestations
- Unconscious patterns and symbolic meanings
- Psychodynamic interventions (interpretation, insight)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import re
import hashlib


@dataclass
class CoreConflict:
    """Core psychological conflicts in PDT framework"""
    case_id: int
    wish: str
    fear: str
    defense_mechanisms: List[str]
    behavioral_manifestations: List[str]
    extraction_hash: str = ""


@dataclass
class ObjectRelation:
    """Internal object representations and relational patterns"""
    case_id: int
    self_representation: str
    object_representation: str
    linking_affect: str
    relational_pattern: str
    transference_potential: str
    extraction_hash: str = ""


@dataclass
class UnconsciosPattern:
    """Unconscious patterns and repetitive behaviors"""
    case_id: int
    pattern_theme: str  # "abandonment", "rejection", "control", etc.
    early_origin: str
    current_manifestation: str
    relational_impact: str
    intervention_approach: str
    extraction_hash: str = ""


@dataclass
class PsychodynamicIntervention:
    """Psychodynamic therapy interventions and their rationales"""
    case_id: int
    intervention_type: str  # "interpretation", "confrontation", "insight", etc.
    situation: str
    therapist_response: str
    targeted_conflict: str
    goal: str
    extraction_hash: str = ""


class PDTKnowledgeExtractor:
    """Extract knowledge from PDT case JSON files"""
    
    def __init__(self, pdt_data_dir: str):
        self.data_dir = Path(pdt_data_dir)
        self.core_conflicts: List[CoreConflict] = []
        self.object_relations: List[ObjectRelation] = []
        self.unconscious_patterns: List[UnconsciosPattern] = []
        self.interventions: List[PsychodynamicIntervention] = []
    
    def extract_all(self) -> None:
        """Extract knowledge from all PDT case files"""
        json_files = sorted(self.data_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)
                
                case_id = case_data.get('client_id', 0)
                
                self._extract_core_conflicts(case_data, case_id)
                self._extract_object_relations(case_data, case_id)
                self._extract_unconscious_patterns(case_data, case_id)
                self._extract_interventions(case_data, case_id)
                
                print(f"  ✓ Case {case_id}: Extracted conflicts, object relations, patterns, interventions")
            except Exception as e:
                print(f"  ✗ Case {json_file.name}: {e}")
    
    def _extract_core_conflicts(self, case_data: Dict, case_id: int) -> None:
        """Extract core psychological conflicts"""
        client_info = case_data.get('client_info', {})
        core_conflict = client_info.get('core_conflict', {})
        
        if isinstance(core_conflict, dict):
            wish = core_conflict.get('wish', '')
            fear = core_conflict.get('fear', '')
            defense_goals = core_conflict.get('defense_goal', [])
            
            # Extract behavioral manifestations
            behavioral = client_info.get('behavioral_response_patterns', [])
            manifestations = []
            if isinstance(behavioral, list):
                for b in behavioral[:3]:
                    if isinstance(b, dict):
                        manifestations.append(b.get('pattern', ''))
            
            conflict = CoreConflict(
                case_id=case_id,
                wish=wish,
                fear=fear,
                defense_mechanisms=defense_goals if isinstance(defense_goals, list) else [],
                behavioral_manifestations=manifestations,
            )
            conflict.extraction_hash = self._hash_object(conflict)
            self.core_conflicts.append(conflict)
    
    def _extract_object_relations(self, case_data: Dict, case_id: int) -> None:
        """Extract object relations and internal representations"""
        client_info = case_data.get('client_info', {})
        object_rels = client_info.get('object_relations', [])
        
        if isinstance(object_rels, list):
            for obj_rel in object_rels:
                if isinstance(obj_rel, dict):
                    self_rep = obj_rel.get('self_representation', '')
                    obj_rep = obj_rel.get('object_representation', '')
                    affect = obj_rel.get('linking_affect', '')
                    
                    relation = ObjectRelation(
                        case_id=case_id,
                        self_representation=self_rep,
                        object_representation=obj_rep,
                        linking_affect=affect,
                        relational_pattern=self._infer_relational_pattern(self_rep, obj_rep),
                        transference_potential=self._assess_transference(obj_rep),
                    )
                    relation.extraction_hash = self._hash_object(relation)
                    self.object_relations.append(relation)
    
    def _extract_unconscious_patterns(self, case_data: Dict, case_id: int) -> None:
        """Extract unconscious patterns and early origins"""
        client_info = case_data.get('client_info', {})
        growth_exp = client_info.get('growth_experiences', [])
        
        main_problem = client_info.get('main_problem', '')
        
        # Identify pattern theme
        pattern_theme = self._identify_pattern_theme(main_problem, growth_exp)
        early_origin = self._extract_early_origin(growth_exp)
        
        pattern = UnconsciosPattern(
            case_id=case_id,
            pattern_theme=pattern_theme,
            early_origin=early_origin,
            current_manifestation=main_problem,
            relational_impact=self._assess_relational_impact(growth_exp),
            intervention_approach="探索早期根源与当前模式的联系，增进内部整合",
        )
        pattern.extraction_hash = self._hash_object(pattern)
        self.unconscious_patterns.append(pattern)
    
    def _extract_interventions(self, case_data: Dict, case_id: int) -> None:
        """Extract psychodynamic therapy interventions from global_plan"""
        global_plan = case_data.get('global_plan', [])
        
        if isinstance(global_plan, list):
            for stage in global_plan:
                if isinstance(stage, dict):
                    content = stage.get('content', {})
                    if isinstance(content, dict):
                        # Iterate through session contents
                        for session_key, session_data in content.items():
                            if isinstance(session_data, dict):
                                theme = session_data.get('theme', '')
                                case_material = session_data.get('case_material', [])
                                rationale = session_data.get('rationale', [])
                                
                                if case_material and rationale:
                                    intervention = PsychodynamicIntervention(
                                        case_id=case_id,
                                        intervention_type=self._classify_intervention_from_content(theme, rationale),
                                        situation='; '.join(case_material[:2]) if case_material else '',
                                        therapist_response=theme,
                                        targeted_conflict=self._extract_targeted_conflict(rationale),
                                        goal="通过揭示无意识冲突与防御机制，促进内部整合",
                                    )
                                    intervention.extraction_hash = self._hash_object(intervention)
                                    self.interventions.append(intervention)
    
    def _classify_intervention_from_content(self, theme: str, rationale: List[str]) -> str:
        """Classify intervention from content"""
        combined = theme + ' ' + ' '.join(rationale)
        
        if any(kw in combined for kw in ['解释', '意味着', '表明', '反映']):
            return "Interpretation"
        elif any(kw in combined for kw in ['冲突', '矛盾', '对抗']):
            return "Confrontation"
        elif any(kw in combined for kw in ['联系', '模式', '重复', '关联']):
            return "Connection Making"
        elif any(kw in combined for kw in ['防御', '保护', '机制']):
            return "Defense Analysis"
        else:
            return "Psychodynamic Facilitation"
    
    def _extract_targeted_conflict(self, rationale: List[str]) -> str:
        """Extract which conflict is being addressed"""
        rationale_str = ' '.join(rationale)
        
        if any(kw in rationale_str for kw in ['愿望', '欲望', '需要']):
            return "Underlying wish and need"
        elif any(kw in rationale_str for kw in ['害怕', '恐惧', '焦虑', '危险']):
            return "Underlying fear and anxiety"
        elif any(kw in rationale_str for kw in ['防御', '保护', '否认']):
            return "Defense mechanisms"
        elif any(kw in rationale_str for kw in ['无意识', '潜意识']):
            return "Unconscious conflict"
        else:
            return "Complex intrapsychic conflict"
    
    def _infer_relational_pattern(self, self_rep: str, obj_rep: str) -> str:
        """Infer relational pattern from representations"""
        if '被动' in self_rep and ('冷漠' in obj_rep or '离开' in obj_rep):
            return "Passive Submission to Potential Abandonment"
        elif '交给' in self_rep or '依赖' in self_rep:
            return "Dependent Attachment with Idealization"
        elif '抢先' in self_rep or '推开' in self_rep:
            return "Defensive Distancing and Preemptive Rejection"
        else:
            return "Complex Relational Ambivalence"
    
    def _assess_transference(self, object_rep: str) -> str:
        """Assess potential transference manifestations"""
        if '理想化' in object_rep or '完美' in object_rep:
            return "Likely positive transference with idealization risk"
        elif '坏' in object_rep or '不能满足' in object_rep:
            return "Likely negative transference with devaluation"
        elif '冷漠' in object_rep or '不负责任' in object_rep:
            return "Likely paternal/maternal transference patterns"
        else:
            return "Complex transference patterns requiring exploration"
    
    def _identify_pattern_theme(self, main_problem: str, growth_exp: List[str]) -> str:
        """Identify core unconscious pattern theme"""
        growth_str = ' '.join(growth_exp)
        
        if '离开' in growth_str or '分离' in growth_str or '抛弃' in main_problem:
            return "Abandonment and Separation Anxiety"
        elif '孤独' in main_problem or '孤独' in growth_str:
            return "Isolation and Disconnection"
        elif '空虚' in main_problem:
            return "Internal Emptiness and Void"
        elif '矛盾' in growth_str or '冲突' in growth_str:
            return "Ambivalent Internal Conflicts"
        else:
            return "Unresolved Developmental Issues"
    
    def _extract_early_origin(self, growth_exp: List[str]) -> str:
        """Extract early relational or developmental origin"""
        if growth_exp:
            return growth_exp[0][:300]
        return "Early relational experiences contributed to current patterns"
    
    def _assess_relational_impact(self, growth_exp: List[str]) -> str:
        """Assess impact on current relationships"""
        impact_indicators = []
        
        for exp in growth_exp:
            if '关系' in exp or '亲密' in exp:
                impact_indicators.append("Intimate relationship difficulties")
            if '信任' in exp or '依赖' in exp:
                impact_indicators.append("Trust and dependency issues")
            if '分离' in exp or '离开' in exp:
                impact_indicators.append("Separation anxiety patterns")
        
        return "; ".join(impact_indicators) if impact_indicators else "Complex relational impact patterns"
    
    def _classify_intervention(self, therapist_resp: str) -> str:
        """Classify psychodynamic intervention type"""
        if any(kw in therapist_resp for kw in ['似乎', '好像', '可能', '潜在']):
            return "Interpretation"
        elif any(kw in therapist_resp for kw in ['矛盾', '相反', '不一致']):
            return "Confrontation"
        elif any(kw in therapist_resp for kw in ['联系', '联想', '关联', '模式']):
            return "Connection Making"
        elif any(kw in therapist_resp for kw in ['感受', '经历', '体验']):
            return "Empathic Exploration"
        else:
            return "Psychodynamic Facilitation"
    
    def _identify_target_conflict(self, therapist_resp: str) -> str:
        """Identify which core conflict is being addressed"""
        if any(kw in therapist_resp for kw in ['需要', '渴望', '希望']):
            return "Underlying wish and need"
        elif any(kw in therapist_resp for kw in ['害怕', '恐惧', '焦虑']):
            return "Underlying fear and anxiety"
        elif any(kw in therapist_resp for kw in ['防御', '保护', '逃避']):
            return "Defense mechanisms"
        else:
            return "Complex intrapsychic conflict"
    
    def _hash_object(self, obj) -> str:
        """Generate hash for object"""
        data_str = json.dumps(asdict(obj), ensure_ascii=False, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def save_knowledge_base(self, output_dir: str) -> None:
        """Save extracted knowledge to JSON files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save core conflicts
        conflicts_list = [asdict(cc) for cc in self.core_conflicts]
        with open(output_path / "pdt_core_conflicts.json", 'w', encoding='utf-8') as f:
            json.dump(conflicts_list, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved {len(conflicts_list)} PDT core conflicts")
        
        # Save object relations
        relations_list = [asdict(or_) for or_ in self.object_relations]
        with open(output_path / "pdt_object_relations.json", 'w', encoding='utf-8') as f:
            json.dump(relations_list, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved {len(relations_list)} PDT object relations")
        
        # Save unconscious patterns
        patterns_list = [asdict(up) for up in self.unconscious_patterns]
        with open(output_path / "pdt_unconscious_patterns.json", 'w', encoding='utf-8') as f:
            json.dump(patterns_list, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved {len(patterns_list)} PDT unconscious patterns")
        
        # Save interventions
        interventions_list = [asdict(pi) for pi in self.interventions]
        with open(output_path / "pdt_psychodynamic_interventions.json", 'w', encoding='utf-8') as f:
            json.dump(interventions_list, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved {len(interventions_list)} PDT psychodynamic interventions")
