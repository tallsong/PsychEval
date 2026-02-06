"""
Session Memory and Client State Tracking

Maintains session-level context for CBT therapy continuity across multiple sessions.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class ClientState:
    """Current state of a client during therapy"""
    case_id: int
    client_name: str
    main_problem: str
    topic: str
    current_stage: str  # initial_conceptualization, core_intervention, consolidation
    current_session: int = 1
    identified_cognitive_patterns: List[str] = field(default_factory=list)
    identified_beliefs: List[str] = field(default_factory=list)
    progress_indicators: Dict[str, Any] = field(default_factory=dict)
    session_goals: List[str] = field(default_factory=list)
    completed_homework: List[str] = field(default_factory=list)
    key_insights: List[str] = field(default_factory=list)


@dataclass
class SessionContext:
    """Context for a single session"""
    session_id: str
    case_id: int
    session_number: int
    timestamp: str
    dialogue_history: List[Dict[str, str]] = field(default_factory=list)  # [{role, content}]
    retrieved_frameworks: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_strategies: List[Dict[str, Any]] = field(default_factory=list)
    session_notes: str = ""
    client_response_quality: Optional[str] = None  # positive, neutral, challenging


class CBTSessionMemory:
    """Manages session-level memory and continuity"""
    
    def __init__(self, case_id: int, client_name: str = ""):
        self.case_id = case_id
        self.client_state = ClientState(
            case_id=case_id,
            client_name=client_name,
            main_problem="",
            topic="",
            current_stage="initial_conceptualization",
        )
        self.sessions: List[SessionContext] = []
        self.current_session: Optional[SessionContext] = None
    
    def initialize_client(
        self,
        main_problem: str,
        topic: str,
        core_beliefs: List[str] = None,
    ) -> None:
        """Initialize client state from assessment"""
        self.client_state.main_problem = main_problem
        self.client_state.topic = topic
        if core_beliefs:
            self.client_state.identified_beliefs = core_beliefs
    
    def start_new_session(self, session_number: int) -> SessionContext:
        """Create new session context"""
        session_id = f"case{self.case_id}_s{session_number}_{datetime.now().isoformat()}"
        
        self.current_session = SessionContext(
            session_id=session_id,
            case_id=self.case_id,
            session_number=session_number,
            timestamp=datetime.now().isoformat(),
        )
        
        self.client_state.current_session = session_number
        self.sessions.append(self.current_session)
        return self.current_session
    
    def add_dialogue(self, role: str, content: str) -> None:
        """Add turn to current session dialogue"""
        if self.current_session is None:
            raise RuntimeError("No active session. Call start_new_session first.")
        
        self.current_session.dialogue_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
    
    def add_retrieved_knowledge(
        self,
        frameworks: List[Dict[str, Any]],
        strategies: List[Dict[str, Any]],
    ) -> None:
        """Record retrieved knowledge for session"""
        if self.current_session is None:
            raise RuntimeError("No active session.")
        
        self.current_session.retrieved_frameworks = frameworks
        self.current_session.retrieved_strategies = strategies
    
    def update_identified_patterns(self, patterns: List[str]) -> None:
        """Update identified cognitive patterns"""
        for pattern in patterns:
            if pattern not in self.client_state.identified_cognitive_patterns:
                self.client_state.identified_cognitive_patterns.append(pattern)
    
    def add_insight(self, insight: str) -> None:
        """Record client insight or breakthrough"""
        if insight not in self.client_state.key_insights:
            self.client_state.key_insights.append(insight)
    
    def add_homework(self, homework: str) -> None:
        """Record homework assigned"""
        self.client_state.session_goals.append(homework)
    
    def complete_homework(self, homework: str) -> None:
        """Mark homework as completed"""
        self.client_state.completed_homework.append(homework)
    
    def update_stage(self, new_stage: str) -> None:
        """Update current therapy stage"""
        self.client_state.current_stage = new_stage
    
    def set_session_notes(self, notes: str) -> None:
        """Set session summary notes"""
        if self.current_session:
            self.current_session.session_notes = notes
    
    def set_client_response_quality(self, quality: str) -> None:
        """Set assessment of client's engagement/response"""
        if self.current_session:
            self.current_session.client_response_quality = quality
    
    def get_dialogue_history(self, max_turns: int = None) -> str:
        """Get formatted dialogue history for context window"""
        if not self.current_session:
            return ""
        
        dialogue = self.current_session.dialogue_history
        if max_turns:
            dialogue = dialogue[-max_turns:]
        
        formatted = []
        for turn in dialogue:
            role = turn["role"].upper()
            content = turn["content"]
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    def get_client_summary(self) -> str:
        """Get summary of client state"""
        state = self.client_state
        summary = f"""
Client Summary:
- Name: {state.client_name}
- Main Problem: {state.main_problem}
- Topic: {state.topic}
- Current Stage: {state.current_stage}
- Session: {state.current_session}
- Identified Patterns: {", ".join(state.identified_cognitive_patterns) if state.identified_cognitive_patterns else "None yet"}
- Core Beliefs: {", ".join(state.identified_beliefs) if state.identified_beliefs else "Unknown"}
- Key Insights: {len(state.key_insights)} recorded
- Homework Completed: {len(state.completed_homework)}/{len(state.session_goals)}
"""
        return summary.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "case_id": self.case_id,
            "client_state": {
                "case_id": self.client_state.case_id,
                "client_name": self.client_state.client_name,
                "main_problem": self.client_state.main_problem,
                "topic": self.client_state.topic,
                "current_stage": self.client_state.current_stage,
                "current_session": self.client_state.current_session,
                "identified_cognitive_patterns": self.client_state.identified_cognitive_patterns,
                "identified_beliefs": self.client_state.identified_beliefs,
                "key_insights": self.client_state.key_insights,
                "completed_homework": self.client_state.completed_homework,
            },
            "sessions": [
                {
                    "session_id": s.session_id,
                    "session_number": s.session_number,
                    "timestamp": s.timestamp,
                    "dialogue_history": s.dialogue_history,
                    "session_notes": s.session_notes,
                    "client_response_quality": s.client_response_quality,
                }
                for s in self.sessions
            ],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CBTSessionMemory":
        """Load from dictionary"""
        memory = cls(data["case_id"])
        state_data = data["client_state"]
        memory.client_state = ClientState(
            case_id=state_data["case_id"],
            client_name=state_data["client_name"],
            main_problem=state_data["main_problem"],
            topic=state_data["topic"],
            current_stage=state_data["current_stage"],
            current_session=state_data["current_session"],
            identified_cognitive_patterns=state_data["identified_cognitive_patterns"],
            identified_beliefs=state_data["identified_beliefs"],
            key_insights=state_data["key_insights"],
            completed_homework=state_data["completed_homework"],
        )
        return memory
    
    def save(self, filepath: str) -> None:
        """Save session memory to JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "CBTSessionMemory":
        """Load session memory from JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
