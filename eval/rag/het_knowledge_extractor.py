"""
HET (Humanistic-Existential Therapy) Knowledge Extractor

Extracts knowledge from HET case files to build RAG knowledge base.
Focus areas:
- Core conditions of worth and self-concept
- Existential themes (meaning, authenticity, freedom, responsibility)
- Client-centered exploration and self-discovery
- Congruence and incongruence manifestations
- Growth-oriented interventions
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import re
import hashlib


@dataclass
class SelfConceptFramework:
    """Self-concept and conditions of worth extracted from HET cases"""
    case_id: int
    conditions_of_worth: List[str]
    current_self_perception: str
    ideal_self: str
    self_incongruence: List[str]
    defensive_behaviors: List[str]
    growth_potential: str
    extraction_hash: str = ""


@dataclass
class ExistentialTheme:
    """Existential concerns from HET perspective"""
    case_id: int
    theme_type: str  # "meaning", "authenticity", "freedom", "isolation", "death", etc.
    manifestations: List[str]
    related_emotions: List[str]
    intervention_direction: str
    extraction_hash: str = ""


@dataclass
class ClientCenteredStrategy:
    """Client-centered and person-centered intervention strategies"""
    case_id: int
    strategy_type: str  # "unconditional_positive_regard", "congruence", "empathy", etc.
    situation: str
    counselor_approach: str
    rationale: str
    expected_outcome: str
    extraction_hash: str = ""


class HETKnowledgeExtractor:
    """Extract knowledge from HET case JSON files"""
    
    def __init__(self, cbt_data_dir: str):
        self.data_dir = Path(cbt_data_dir)
        self.self_concepts: List[SelfConceptFramework] = []
        self.existential_themes: List[ExistentialTheme] = []
        self.strategies: List[ClientCenteredStrategy] = []
    
    def extract_all(self) -> None:
        """Extract knowledge from all HET case files"""
        json_files = sorted(self.data_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)
                
                case_id = case_data.get('client_id', 0)
                
                self._extract_self_concept(case_data, case_id)
                self._extract_existential_themes(case_data, case_id)
                self._extract_strategies(case_data, case_id)
                
                print(f"  ✓ Case {case_id}: Extracted self-concept, existential themes, strategies")
            except Exception as e:
                print(f"  ✗ Case {json_file.name}: {e}")
    
    def _extract_self_concept(self, case_data: Dict, case_id: int) -> None:
        """Extract self-concept and conditions of worth"""
        client_info = case_data.get('client_info', {})
        growth_exp = client_info.get('growth_experiences', [])
        
        conditions_of_worth = []
        if growth_exp:
            for exp in growth_exp:
                if any(kw in exp for kw in ['要求', '期待', '认可', '肯定', '应该', '必须']):
                    conditions_of_worth.append(exp)
        
        current_self = client_info.get('main_problem', '')
        ideal_self = ""
        
        # Extract incongruence indicators
        incongruence = []
        if 'language_features' in client_info.get('static_traits', {}):
            lang = client_info['static_traits']['language_features']
            if any(kw in lang for kw in ['矛盾', '冲突', '不一致']):
                incongruence.append(lang)
        
        framework = SelfConceptFramework(
            case_id=case_id,
            conditions_of_worth=conditions_of_worth[:5],
            current_self_perception=current_self,
            ideal_self=ideal_self,
            self_incongruence=incongruence,
            defensive_behaviors=self._extract_defensive_behaviors(case_data),
            growth_potential=self._extract_growth_potential(case_data),
        )
        framework.extraction_hash = self._hash_object(framework)
        self.self_concepts.append(framework)
    
    def _extract_existential_themes(self, case_data: Dict, case_id: int) -> None:
        """Extract existential concerns"""
        client_info = case_data.get('client_info', {})
        existential_topics = client_info.get('existentialism_topic', [])
        
        if isinstance(existential_topics, list):
            for topic in existential_topics:
                if isinstance(topic, dict):
                    theme = ExistentialTheme(
                        case_id=case_id,
                        theme_type=topic.get('theme', ''),
                        manifestations=topic.get('manifestations', []),
                        related_emotions=self._extract_emotions(topic),
                        intervention_direction=self._generate_intervention_direction(topic),
                    )
                    theme.extraction_hash = self._hash_object(theme)
                    self.existential_themes.append(theme)
    
    def _extract_strategies(self, case_data: Dict, case_id: int) -> None:
        """Extract client-centered intervention strategies from global_plan"""
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
                                    strategy = ClientCenteredStrategy(
                                        case_id=case_id,
                                        strategy_type=self._classify_strategy_from_content(theme, case_material),
                                        situation='; '.join(case_material[:2]) if case_material else '',
                                        counselor_approach=theme,
                                        rationale='; '.join(rationale) if rationale else '',
                                        expected_outcome="通过积极倾听与情感反映，帮助来访者增进自我觉察",
                                    )
                                    strategy.extraction_hash = self._hash_object(strategy)
                                    self.strategies.append(strategy)
    
    def _classify_strategy_from_content(self, theme: str, case_material: List[str]) -> str:
        """Classify strategy from global_plan content"""
        combined_text = theme + ' ' + ' '.join(case_material)
        
        if any(kw in combined_text for kw in ['倾听', '反映', '理解', '认可']):
            return "Empathic Understanding"
        elif any(kw in combined_text for kw in ['接纳', '肯定', '正向']):
            return "Unconditional Positive Regard"
        elif any(kw in combined_text for kw in ['探索', '觉察', '反思', '意义']):
            return "Existential Exploration"
        elif any(kw in combined_text for kw in ['建立', '联盟', '框架', '安全']):
            return "Therapeutic Alliance"
        else:
            return "Client-Centered Facilitation"
    
    def _extract_defensive_behaviors(self, case_data: Dict) -> List[str]:
        """Extract defense mechanisms or coping behaviors"""
        behaviors = []
        client_info = case_data.get('client_info', {})
        
        # Look for protective behaviors in language features
        lang_features = client_info.get('static_traits', {}).get('language_features', '')
        if '回避' in lang_features or '躲闪' in lang_features:
            behaviors.append("Avoidance")
        if '防御' in lang_features or '保护' in lang_features:
            behaviors.append("Self-protection")
        
        return behaviors
    
    def _extract_growth_potential(self, case_data: Dict) -> str:
        """Extract growth-oriented potential and self-actualization direction"""
        client_info = case_data.get('client_info', {})
        main_problem = client_info.get('main_problem', '')
        
        # Invert problem to growth direction
        if '缺乏意义' in main_problem:
            return "探索人生意义，建立自我价值感"
        elif '孤独' in main_problem:
            return "建立真诚的人际关系和连接"
        elif '焦虑' in main_problem:
            return "增进自我认识，减少内在冲突"
        else:
            return "自我实现和个人成长"
    
    def _extract_emotions(self, topic: Dict) -> List[str]:
        """Extract emotions from theme outcomes"""
        outcomes = topic.get('outcomes', [])
        emotions = []
        
        emotion_keywords = {
            '低落': 'Depression',
            '焦虑': 'Anxiety',
            '孤独': 'Isolation',
            '绝望': 'Hopelessness',
            '空虚': 'Emptiness',
            '困惑': 'Confusion',
        }
        
        for outcome in outcomes:
            for kw, emotion in emotion_keywords.items():
                if kw in outcome:
                    emotions.append(emotion)
        
        return emotions
    
    def _generate_intervention_direction(self, topic: Dict) -> str:
        """Generate intervention direction based on theme"""
        theme = topic.get('theme', '')
        
        if '无意义' in theme:
            return "帮助来访者探索生活的意义和个人价值"
        elif '孤独' in theme:
            return "促进真诚的人际连接和归属感"
        elif '真实性' in theme or '真实' in theme:
            return "引导来访者发现和表现真实的自我"
        elif '自由' in theme:
            return "帮助来访者承认和行使个人选择权"
        else:
            return "通过自我探索和反思实现个人成长"
    
    def _classify_strategy(self, counselor_resp: str) -> str:
        """Classify intervention strategy type"""
        if any(kw in counselor_resp for kw in ['感受', '感到', '体验', '理解']):
            return "Empathic Understanding"
        elif any(kw in counselor_resp for kw in ['反映', '回应', '倾听']):
            return "Reflection"
        elif any(kw in counselor_resp for kw in ['接纳', '肯定', '认可']):
            return "Unconditional Positive Regard"
        elif any(kw in counselor_resp for kw in ['想法', '感受', '想象', '探索']):
            return "Existential Exploration"
        else:
            return "Client-Centered Facilitation"
    
    def _extract_rationale(self, counselor_resp: str) -> str:
        """Extract therapeutic rationale from counselor response"""
        if len(counselor_resp) > 100:
            return counselor_resp[:200]
        return "通过提供真诚的理解和接纳，帮助来访者增进自我认识"
    
    def _hash_object(self, obj) -> str:
        """Generate hash for object"""
        data_str = json.dumps(asdict(obj), ensure_ascii=False, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def save_knowledge_base(self, output_dir: str) -> None:
        """Save extracted knowledge to JSON files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save self-concepts
        self_concepts_list = [asdict(sc) for sc in self.self_concepts]
        with open(output_path / "het_self_concepts.json", 'w', encoding='utf-8') as f:
            json.dump(self_concepts_list, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved {len(self_concepts_list)} HET self-concept frameworks")
        
        # Save existential themes
        existential_list = [asdict(et) for et in self.existential_themes]
        with open(output_path / "het_existential_themes.json", 'w', encoding='utf-8') as f:
            json.dump(existential_list, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved {len(existential_list)} HET existential themes")
        
        # Save strategies
        strategies_list = [asdict(st) for st in self.strategies]
        with open(output_path / "het_client_centered_strategies.json", 'w', encoding='utf-8') as f:
            json.dump(strategies_list, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved {len(strategies_list)} HET client-centered strategies")
