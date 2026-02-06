"""
PDT (Psychodynamic Therapy) Counselor Agent

Implements a psychodynamic therapist using RAG retrieval.
Focuses on: core conflicts, unconscious patterns, object relations, transference insights.
"""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime

try:
    from .pdt_retriever import PDTRetriever
except ImportError:
    from pdt_retriever import PDTRetriever


@dataclass
class ClientState:
    """Client state tracked across PDT sessions"""
    case_id: int
    client_name: str
    presenting_problem: str
    topic: str
    session_number: int
    identified_conflicts: List[str]
    defense_mechanisms: List[str]
    transference_patterns: List[str]
    insights: List[str]
    therapeutic_progress: str


@dataclass
class PDTSessionContext:
    """Per-session context for PDT therapy"""
    session_id: str
    dialogue_history: List[Dict]
    retrieved_core_conflicts: List[Dict]
    retrieved_object_relations: List[Dict]
    retrieved_patterns: List[Dict]
    retrieved_interventions: List[Dict]
    transference_observations: List[str]
    notes: str
    session_summary: str


class PDTSessionMemory:
    """Manage PDT client state and session memory"""
    
    def __init__(self):
        self.client_state: Optional[ClientState] = None
        self.session_context: Optional[PDTSessionContext] = None
    
    def initialize_client(
        self,
        case_id: int,
        client_name: str,
        presenting_problem: str,
        topic: str
    ) -> ClientState:
        """Initialize new client"""
        self.client_state = ClientState(
            case_id=case_id,
            client_name=client_name,
            presenting_problem=presenting_problem,
            topic=topic,
            session_number=0,
            identified_conflicts=[],
            defense_mechanisms=[],
            transference_patterns=[],
            insights=[],
            therapeutic_progress="初始评估阶段"
        )
        return self.client_state
    
    def start_new_session(self) -> PDTSessionContext:
        """Create new session context"""
        if not self.client_state:
            raise ValueError("Client not initialized")
        
        self.client_state.session_number += 1
        session_id = f"{self.client_state.case_id}_pdt_session_{self.client_state.session_number}"
        
        self.session_context = PDTSessionContext(
            session_id=session_id,
            dialogue_history=[],
            retrieved_core_conflicts=[],
            retrieved_object_relations=[],
            retrieved_patterns=[],
            retrieved_interventions=[],
            transference_observations=[],
            notes="",
            session_summary=""
        )
        return self.session_context
    
    def add_dialogue(self, speaker: str, content: str) -> None:
        """Record dialogue turn"""
        if not self.session_context:
            raise ValueError("Session not started")
        
        self.session_context.dialogue_history.append({
            "timestamp": datetime.now().isoformat(),
            "speaker": speaker,
            "content": content
        })
    
    def update_identified_conflicts(self, conflicts: List[str]) -> None:
        """Update identified core conflicts"""
        if self.client_state:
            self.client_state.identified_conflicts.extend(conflicts)
            self.client_state.identified_conflicts = list(
                set(self.client_state.identified_conflicts)
            )
    
    def record_defense_mechanism(self, defense: str) -> None:
        """Record observed defense mechanism"""
        if self.client_state and defense not in self.client_state.defense_mechanisms:
            self.client_state.defense_mechanisms.append(defense)
    
    def record_transference(self, transference_obs: str) -> None:
        """Record transference observation"""
        if self.session_context:
            self.session_context.transference_observations.append(transference_obs)
    
    def add_insight(self, insight: str) -> None:
        """Record client insight or breakthrough"""
        if self.client_state:
            self.client_state.insights.append(insight)
    
    def update_progress(self, progress: str) -> None:
        """Update therapeutic progress stage"""
        if self.client_state:
            self.client_state.therapeutic_progress = progress
    
    def to_dict(self) -> Dict:
        """Serialize to dict"""
        return {
            'client_state': asdict(self.client_state) if self.client_state else None,
            'session_context': {
                'session_id': self.session_context.session_id if self.session_context else None,
                'dialogue_history': self.session_context.dialogue_history if self.session_context else [],
                'transference_observations': self.session_context.transference_observations if self.session_context else [],
            }
        }
    
    def save(self, filepath: str) -> None:
        """Save session memory to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class PDTCounselorAgent:
    """PDT counselor agent integrating RAG retrieval"""
    
    def __init__(self, retriever: PDTRetriever):
        self.retriever = retriever
        self.session_memory: Optional[PDTSessionMemory] = None
    
    def initialize_client(
        self,
        case_id: int,
        client_name: str,
        presenting_problem: str,
        topic: str
    ) -> PDTSessionMemory:
        """Initialize new client"""
        self.session_memory = PDTSessionMemory()
        self.session_memory.initialize_client(
            case_id, client_name, presenting_problem, topic
        )
        return self.session_memory
    
    def start_session(self, session_num: int) -> str:
        """Generate session opening"""
        if not self.session_memory:
            raise ValueError("Client not initialized")
        
        self.session_memory.start_new_session()
        client_state = self.session_memory.client_state
        
        if session_num == 1:
            opening = f"""欢迎，{client_state.client_name}。
我很高兴有机会和你一起工作。我想告诉你，在这里，我们会深入探讨你的内心世界——
你的想法、感受、关系模式，以及那些可能你自己还没有完全意识到的东西。

你提到{client_state.presenting_problem}。
这些困境往往与我们更深层的心理冲突有关。我的工作是帮助你发现这些模式，理解它们的根源。

让我们从你现在最困扰你的事情开始吧。"""
        else:
            opening = f"""欢迎回来。上次我们探讨了你在{client_state.topic}中的一些模式。
今天，我想继续深入。在过去的一周里，有什么新的感受或体验吗？"""
        
        self.session_memory.add_dialogue("therapist", opening)
        return opening
    
    def process_client_input(
        self,
        client_input: str,
        relational_patterns: Optional[List[str]] = None,
        defensive_behaviors: Optional[List[str]] = None
    ) -> Dict:
        """Process client input and generate psychodynamic response"""
        if not self.session_memory:
            raise ValueError("Session not initialized")
        
        client_state = self.session_memory.client_state
        
        # Record client input
        self.session_memory.add_dialogue("client", client_input)
        
        # Retrieve relevant PDT knowledge
        retrieved = self.retriever.retrieve(
            client_problem=client_input,
            relational_patterns=relational_patterns or [],
            defensive_behaviors=defensive_behaviors or [],
            top_k=2
        )
        
        # Store retrieved knowledge
        self.session_memory.session_context.retrieved_core_conflicts = retrieved.core_conflicts
        self.session_memory.session_context.retrieved_object_relations = retrieved.object_relations
        self.session_memory.session_context.retrieved_patterns = retrieved.unconscious_patterns
        self.session_memory.session_context.retrieved_interventions = retrieved.interventions
        
        # Generate psychodynamic response
        therapist_response = self._generate_response(client_input, retrieved)
        
        # Record response
        self.session_memory.add_dialogue("therapist", therapist_response)
        
        # Update memory with identified patterns
        if retrieved.core_conflicts:
            conflict = retrieved.core_conflicts[0]
            self.session_memory.update_identified_conflicts([
                conflict.get('wish', ''),
                conflict.get('fear', '')
            ])
        
        return {
            "therapist_response": therapist_response,
            "core_conflicts": retrieved.core_conflicts,
            "object_relations": retrieved.object_relations,
            "unconscious_patterns": retrieved.unconscious_patterns,
            "interventions": retrieved.interventions,
        }
    
    def _generate_response(self, client_input: str, retrieved) -> str:
        """Generate psychodynamic interpretation"""
        response_parts = []
        
        # Opening acknowledgment with depth
        response_parts.append("我听到你说的...")
        
        # Identify potential pattern or conflict
        if retrieved.unconscious_patterns:
            pattern = retrieved.unconscious_patterns[0]
            pattern_theme = pattern.get('pattern_theme', '')
            response_parts.append(
                f"\n这让我想到，也许这与你深层的{pattern_theme}的担忧有关。"
            )
        
        # Explore object relations
        if retrieved.object_relations:
            obj_rel = retrieved.object_relations[0]
            self_rep = obj_rel.get('self_representation', '')
            response_parts.append(
                f"\n我注意到你的描述中，似乎有这样一个想法：{self_rep}"
            )
            response_parts.append(
                "\n这个自我认知是如何形成的呢？你能回想起什么时候开始有这样的感受吗？"
            )
        
        # Core conflict exploration
        if retrieved.core_conflicts:
            conflict = retrieved.core_conflicts[0]
            wish = conflict.get('wish', '')
            fear = conflict.get('fear', '')
            
            if wish and fear:
                response_parts.append(
                    f"\n我有一个想法：也许在你心中，既有对{wish}的渴望，"
                    f"同时也有对{fear}的恐惧。这两种力量在拉扯。"
                )
        
        # Interpretation
        if retrieved.interventions:
            intervention = retrieved.interventions[0]
            int_type = intervention.get('intervention_type', '')
            
            if int_type == 'Interpretation':
                response_parts.append(
                    "\n让我大胆地说出我的观察：这个行为模式似乎是一种保护机制，"
                    "保护你免受更深层的伤害。但它也阻止了你得到真正想要的东西。"
                )
        
        # Invitation for reflection
        response_parts.append(
            "\n你对这个想法有什么反应？这是否触及了什么？"
        )
        
        return "".join(response_parts)
    
    def complete_session(self) -> str:
        """Generate session summary with key insights"""
        if not self.session_memory or not self.session_memory.session_context:
            raise ValueError("Session not completed")
        
        memory = self.session_memory
        summary = f"""
=== 第 {memory.client_state.session_number} 次治疗总结 ===

呈现问题：{memory.client_state.presenting_problem}

识别的核心冲突：
{'; '.join(memory.client_state.identified_conflicts[:2]) if memory.client_state.identified_conflicts else '继续探索中'}

观察到的防御机制：
{'; '.join(memory.client_state.defense_mechanisms[:2]) if memory.client_state.defense_mechanisms else '需进一步明确'}

本次突破点：
- 对潜在的无意识模式的初步认识
- 对早期关系经历与当前模式的联系的思考

移情观察：
{'; '.join(memory.session_context.transference_observations) if memory.session_context.transference_observations else '待后续发展'}

治疗进程：{memory.client_state.therapeutic_progress}

下次目标：
1. 继续深化对核心冲突的理解
2. 探索这个模式如何影响你的亲密关系
3. 考虑潜在的早期根源

这是一个持续的过程。我们会一起逐步揭开这些层面。
"""
        
        memory.session_context.session_summary = summary
        return summary
    
    def save_session(self, output_dir: str) -> None:
        """Save session to file"""
        if not self.session_memory:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        case_id = self.session_memory.client_state.case_id
        session_num = self.session_memory.client_state.session_number
        
        filepath = output_path / f"pdt_case_{case_id}_session_{session_num}.json"
        self.session_memory.save(str(filepath))
