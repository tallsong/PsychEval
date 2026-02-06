"""
CBT Knowledge Extractor

Extract structured knowledge from CBT JSON cases:
- Cognitive frameworks (special situations, ABC models)
- Intervention strategies (techniques per stage)
- Therapy progress (session-level goals and outcomes)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class CognitiveFramework:
    """Represents a cognitive framework from special situation"""
    case_id: int
    situation_index: int
    event: str
    automatic_thoughts: List[str]
    conditional_assumptions: List[str]
    core_beliefs: List[str]
    cognitive_patterns: List[str]
    compensatory_strategies: List[str]
    problem_category: str
    framework_id: str  # hash-based unique identifier


@dataclass
class InterventionStrategy:
    """Represents a therapeutic intervention strategy"""
    case_id: int
    stage_number: int
    stage_name: str
    session_number: int
    theme: str
    technique: str
    rationale: str
    case_material: str  # Specific homework/exercise
    target_cognitive_pattern: Optional[str] = None
    expected_outcome: Optional[str] = None


@dataclass
class TherapyProgress:
    """Represents session-level progress information"""
    case_id: int
    stage_number: int
    stage_name: str
    session_number: int
    theme: str
    objectives: str
    therapy_content: str
    focus_areas: List[str]


class CBTKnowledgeExtractor:
    """
    Extracts structured CBT knowledge from JSON case files.
    
    Knowledge types:
    1. Cognitive Frameworks - ABC models and cognitive patterns
    2. Intervention Strategies - Specific techniques per stage/session
    3. Therapy Progress - Session goals and outcomes
    """
    
    def __init__(self, cbt_data_dir: str):
        """
        Initialize extractor with CBT data directory
        
        Args:
            cbt_data_dir: Path to data/cbt directory containing JSON files
        """
        self.data_dir = Path(cbt_data_dir)
        self.cognitive_frameworks: List[CognitiveFramework] = []
        self.intervention_strategies: List[InterventionStrategy] = []
        self.therapy_progress: List[TherapyProgress] = []
        self.case_metadata: Dict[int, Dict[str, Any]] = {}
        
    def extract_all(self) -> None:
        """Extract all knowledge from CBT case files"""
        json_files = sorted(self.data_dir.glob("*.json"), key=lambda x: int(x.stem))
        
        for json_file in json_files:
            case_id = int(json_file.stem)
            print(f"Processing case {case_id}...")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)
                
                # Extract metadata
                self._extract_metadata(case_id, case_data)
                
                # Extract cognitive frameworks
                self._extract_cognitive_frameworks(case_id, case_data)
                
                # Extract intervention strategies
                self._extract_intervention_strategies(case_id, case_data)
                
                # Extract therapy progress
                self._extract_therapy_progress(case_id, case_data)
                
            except Exception as e:
                print(f"Error processing case {case_id}: {e}")
                continue
        
        print(f"\nExtraction complete:")
        print(f"  - Cognitive Frameworks: {len(self.cognitive_frameworks)}")
        print(f"  - Intervention Strategies: {len(self.intervention_strategies)}")
        print(f"  - Therapy Progress Records: {len(self.therapy_progress)}")
    
    def _extract_metadata(self, case_id: int, case_data: Dict[str, Any]) -> None:
        """Extract client metadata"""
        client_info = case_data.get("client_info", {})
        self.case_metadata[case_id] = {
            "case_id": case_id,
            "main_problem": client_info.get("main_problem", ""),
            "topic": client_info.get("topic", ""),
            "age": client_info.get("static_traits", {}).get("age", ""),
            "gender": client_info.get("static_traits", {}).get("gender", ""),
            "occupation": client_info.get("static_traits", {}).get("occupation", ""),
            "core_beliefs": client_info.get("core_beliefs", []),
            "core_demands": client_info.get("core_demands", ""),
        }
    
    def _extract_cognitive_frameworks(self, case_id: int, case_data: Dict[str, Any]) -> None:
        """Extract cognitive frameworks from special situations"""
        client_info = case_data.get("client_info", {})
        special_situations = client_info.get("special_situations", [])
        
        for idx, situation in enumerate(special_situations):
            # Generate unique ID based on content
            content = str(situation)
            framework_id = hashlib.md5(f"{case_id}_{idx}_{content}".encode()).hexdigest()[:16]
            
            framework = CognitiveFramework(
                case_id=case_id,
                situation_index=idx,
                event=situation.get("event", ""),
                automatic_thoughts=situation.get("automatic_thoughts", []),
                conditional_assumptions=situation.get("conditional_assumptions", []),
                core_beliefs=situation.get("core_beliefs", []),
                cognitive_patterns=situation.get("cognitive_pattern", []),
                compensatory_strategies=situation.get("compensatory_strategies", []),
                problem_category=client_info.get("topic", ""),
                framework_id=framework_id,
            )
            self.cognitive_frameworks.append(framework)
    
    def _extract_intervention_strategies(self, case_id: int, case_data: Dict[str, Any]) -> None:
        """Extract intervention strategies from global plan"""
        global_plan = case_data.get("global_plan", [])
        
        for stage in global_plan:
            stage_number = stage.get("stage_number", 0)
            stage_name = stage.get("stage_name", "")
            content_dict = stage.get("content", {})
            
            # content is a dict with keys like '第1次_session_content', '第2次_session_content'
            if isinstance(content_dict, dict):
                for session_key, session_content in content_dict.items():
                    # Extract session number from key (e.g., '第1次' -> 1)
                    session_number = self._extract_session_number(session_key)
                    
                    if isinstance(session_content, dict):
                        theme = session_content.get("theme", "")
                        rationale = session_content.get("rationale", "")
                        case_material = session_content.get("case_material", "")
                        
                        # Extract technique from theme or rationale
                        technique = self._extract_technique(theme, rationale)
                        target_pattern = self._extract_cognitive_pattern(theme, rationale)
                        
                        strategy = InterventionStrategy(
                            case_id=case_id,
                            stage_number=stage_number,
                            stage_name=stage_name,
                            session_number=session_number,
                            theme=theme,
                            technique=technique,
                            rationale=rationale,
                            case_material=case_material,
                            target_cognitive_pattern=target_pattern,
                            expected_outcome=session_content.get("expected_outcome", None),
                        )
                        self.intervention_strategies.append(strategy)
    
    def _extract_therapy_progress(self, case_id: int, case_data: Dict[str, Any]) -> None:
        """Extract therapy progress information"""
        global_plan = case_data.get("global_plan", [])
        
        for stage in global_plan:
            stage_number = stage.get("stage_number", 0)
            stage_name = stage.get("stage_name", "")
            content_dict = stage.get("content", {})
            
            # content is a dict with keys like '第1次_session_content'
            if isinstance(content_dict, dict):
                for session_key, session_content in content_dict.items():
                    session_number = self._extract_session_number(session_key)
                    
                    if isinstance(session_content, dict):
                        theme = session_content.get("theme", "")
                        dialogue = ""  # Not available in this structure
                        
                        # Extract focus areas from theme
                        focus_areas = self._extract_focus_areas(f"{theme} {session_content.get('rationale', '')}")
                        
                        progress = TherapyProgress(
                            case_id=case_id,
                            stage_number=stage_number,
                            stage_name=stage_name,
                            session_number=session_number,
                            theme=theme,
                            objectives=theme,
                            therapy_content=session_content.get("case_material", "")[:200],
                            focus_areas=focus_areas,
                        )
                        self.therapy_progress.append(progress)
    
    def _extract_session_number(self, session_key: str) -> int:
        """Extract session number from key like '第1次_session_content'"""
        import re
        match = re.search(r'第(\d+)次', session_key)
        if match:
            return int(match.group(1))
        return 0
    
    def _extract_technique(self, theme: str, rationale: str) -> str:
        """Extract CBT technique name from theme and rationale"""
        techniques = [
            "guided_discovery", "socratic_questioning", "behavioral_experiment",
            "cognitive_restructuring", "thought_record", "exposure", "relaxation",
            "problem_solving", "assertiveness", "activity_scheduling", "homework"
        ]
        
        combined = f"{theme} {rationale}".lower()
        for technique in techniques:
            if technique.replace("_", " ") in combined:
                return technique
        
        return "general_intervention"
    
    def _extract_cognitive_pattern(self, theme: str, rationale: str) -> Optional[str]:
        """Extract target cognitive pattern"""
        patterns = [
            "All-or-Nothing Thinking", "Overgeneralization", "Mental Filter",
            "Disqualifying the Positive", "Jumping to Conclusions", "Magnification",
            "Emotional Reasoning", "Should Statements", "Labeling", "Personalization",
            "Comparing and Despairing", "Fortune Telling"
        ]
        
        combined = f"{theme} {rationale}"
        for pattern in patterns:
            if pattern in combined or pattern.replace("-", " ").lower() in combined.lower():
                return pattern
        
        return None
    
    def _extract_focus_areas(self, dialogue: str) -> List[str]:
        """Extract focus areas from dialogue"""
        focus_keywords = [
            "feeling", "thought", "belief", "behavior", "emotion", "anxiety",
            "depression", "relationship", "work", "family", "decision"
        ]
        
        focus_areas = []
        dialogue_lower = dialogue.lower()
        for keyword in focus_keywords:
            if keyword in dialogue_lower:
                focus_areas.append(keyword)
        
        return list(set(focus_areas))[:5]  # Return up to 5 unique focus areas
    
    def save_knowledge_base(self, output_dir: str) -> None:
        """Save extracted knowledge to JSON files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save cognitive frameworks
        frameworks_file = output_path / "cognitive_frameworks.json"
        with open(frameworks_file, 'w', encoding='utf-8') as f:
            json.dump(
                [asdict(fw) for fw in self.cognitive_frameworks],
                f, ensure_ascii=False, indent=2
            )
        print(f"Saved {len(self.cognitive_frameworks)} frameworks to {frameworks_file}")
        
        # Save intervention strategies
        strategies_file = output_path / "intervention_strategies.json"
        with open(strategies_file, 'w', encoding='utf-8') as f:
            json.dump(
                [asdict(s) for s in self.intervention_strategies],
                f, ensure_ascii=False, indent=2
            )
        print(f"Saved {len(self.intervention_strategies)} strategies to {strategies_file}")
        
        # Save therapy progress
        progress_file = output_path / "therapy_progress.json"
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(
                [asdict(p) for p in self.therapy_progress],
                f, ensure_ascii=False, indent=2
            )
        print(f"Saved {len(self.therapy_progress)} progress records to {progress_file}")
        
        # Save metadata
        metadata_file = output_path / "case_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.case_metadata, f, ensure_ascii=False, indent=2)
        print(f"Saved metadata for {len(self.case_metadata)} cases to {metadata_file}")
    
    def get_cognitive_frameworks(self) -> List[CognitiveFramework]:
        """Get all extracted cognitive frameworks"""
        return self.cognitive_frameworks
    
    def get_intervention_strategies(self) -> List[InterventionStrategy]:
        """Get all extracted intervention strategies"""
        return self.intervention_strategies
    
    def get_therapy_progress(self) -> List[TherapyProgress]:
        """Get all extracted therapy progress records"""
        return self.therapy_progress
    
    def get_case_metadata(self, case_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata for specific case"""
        return self.case_metadata.get(case_id)
