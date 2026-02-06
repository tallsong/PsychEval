"""
HET (Humanistic-Existential Therapy) Counselor Agent

Implements a humanistic-existential counselor using RAG retrieval.
Focuses on: unconditional positive regard, authenticity, self-exploration, existential meaning.
"""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime

try:
    from .het_retriever import HETRetriever
except ImportError:
    from het_retriever import HETRetriever


@dataclass
class ClientState:
    """Client state tracked across HET sessions"""
    case_id: int
    client_name: str
    presenting_problem: str
    self_perception_theme: str
    session_number: int
    identified_incongruences: List[str]
    growth_directions: List[str]
    insights: List[str]
    homework_assignments: List[str]


@dataclass
class HETSessionContext:
    """Per-session context for HET therapy"""
    session_id: str
    dialogue_history: List[Dict]
    retrieved_self_concepts: List[Dict]
    retrieved_existential_themes: List[Dict]
    retrieved_strategies: List[Dict]
    notes: str
    session_summary: str


class HETSessionMemory:
    """Manage HET client state and session memory"""
    
    def __init__(self):
        self.client_state: Optional[ClientState] = None
        self.session_context: Optional[HETSessionContext] = None
    
    def initialize_client(
        self,
        case_id: int,
        client_name: str,
        presenting_problem: str,
        self_perception_theme: str
    ) -> ClientState:
        """Initialize new client"""
        self.client_state = ClientState(
            case_id=case_id,
            client_name=client_name,
            presenting_problem=presenting_problem,
            self_perception_theme=self_perception_theme,
            session_number=0,
            identified_incongruences=[],
            growth_directions=[],
            insights=[],
            homework_assignments=[]
        )
        return self.client_state
    
    def start_new_session(self) -> HETSessionContext:
        """Create new session context"""
        if not self.client_state:
            raise ValueError("Client not initialized")
        
        self.client_state.session_number += 1
        session_id = f"{self.client_state.case_id}_session_{self.client_state.session_number}"
        
        self.session_context = HETSessionContext(
            session_id=session_id,
            dialogue_history=[],
            retrieved_self_concepts=[],
            retrieved_existential_themes=[],
            retrieved_strategies=[],
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
    
    def update_incongruences(self, incongruences: List[str]) -> None:
        """Update identified self-incongruences"""
        if self.client_state:
            self.client_state.identified_incongruences.extend(incongruences)
            # Deduplicate
            self.client_state.identified_incongruences = list(
                set(self.client_state.identified_incongruences)
            )
    
    def add_insight(self, insight: str) -> None:
        """Record client insight"""
        if self.client_state:
            self.client_state.insights.append(insight)
    
    def to_dict(self) -> Dict:
        """Serialize to dict"""
        return {
            'client_state': asdict(self.client_state) if self.client_state else None,
            'session_context': {
                'session_id': self.session_context.session_id if self.session_context else None,
                'dialogue_history': self.session_context.dialogue_history if self.session_context else [],
            }
        }
    
    def save(self, filepath: str) -> None:
        """Save session memory to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class HETCounselorAgent:
    """HET counselor agent integrating RAG retrieval"""
    
    def __init__(self, retriever: HETRetriever):
        self.retriever = retriever
        self.session_memory: Optional[HETSessionMemory] = None
    
    def initialize_client(
        self,
        case_id: int,
        client_name: str,
        presenting_problem: str,
        self_perception_theme: str
    ) -> HETSessionMemory:
        """Initialize new client"""
        self.session_memory = HETSessionMemory()
        self.session_memory.initialize_client(
            case_id, client_name, presenting_problem, self_perception_theme
        )
        return self.session_memory
    
    def start_session(self, session_num: int) -> str:
        """Generate session opening"""
        if not self.session_memory:
            raise ValueError("Client not initialized")
        
        self.session_memory.start_new_session()
        client_state = self.session_memory.client_state
        
        if session_num == 1:
            opening = f"""欢迎，{client_state.client_name}。我很高兴能和你相识。
我想首先让你知道，这是一个安全、保密的空间，我们可以自由地探讨你现在的感受和想法。

你提到{client_state.presenting_problem}。我想听听你的故事，但没有任何压力。
随时可以跳过你不想讨论的话题。我的角色是倾听、理解，并和你一起探索。

你觉得现在从哪里开始会比较好呢？"""
        else:
            opening = f"""欢迎回来，{client_state.client_name}。
上次我们讨论了{client_state.presenting_problem}。
今天你想继续探索什么呢？有什么新的想法或感受吗？"""
        
        self.session_memory.add_dialogue("counselor", opening)
        return opening
    
    def process_client_input(
        self,
        client_input: str,
        identified_themes: Optional[List[str]] = None
    ) -> Dict:
        """Process client input and generate response"""
        if not self.session_memory:
            raise ValueError("Session not initialized")
        
        client_state = self.session_memory.client_state
        
        # Record client input
        self.session_memory.add_dialogue("client", client_input)
        
        # Retrieve relevant knowledge
        retrieved = self.retriever.retrieve(
            client_problem=client_input,
            self_perception=client_state.self_perception_theme,
            existential_concern=identified_themes[0] if identified_themes else None,
            top_k=2
        )
        
        # Store retrieved knowledge
        self.session_memory.session_context.retrieved_self_concepts = retrieved.self_concepts
        self.session_memory.session_context.retrieved_existential_themes = retrieved.existential_themes
        self.session_memory.session_context.retrieved_strategies = retrieved.strategies
        
        # Generate response
        counselor_response = self._generate_response(client_input, retrieved)
        
        # Record response
        self.session_memory.add_dialogue("counselor", counselor_response)
        
        return {
            "counselor_response": counselor_response,
            "retrieved_strategies": retrieved.strategies,
            "existential_themes": retrieved.existential_themes,
        }
    
    def _generate_response(self, client_input: str, retrieved) -> str:
        """Generate humanistic-existential response"""
        strategies = retrieved.strategies
        themes = retrieved.existential_themes
        concepts = retrieved.self_concepts
        
        # Build response with empathy and existential exploration
        response_parts = []
        
        # Empathic reflection
        if '担心' in client_input or '害怕' in client_input:
            response_parts.append("我听出你内心有些担忧...")
        elif '感到' in client_input or '觉得' in client_input:
            response_parts.append("你在分享这些感受，我很感谢你的信任...")
        else:
            response_parts.append("我在用心倾听你的分享...")
        
        # Existential exploration
        if themes:
            theme = themes[0]
            theme_type = theme.get('theme_type', '')
            response_parts.append(
                f"\n看起来{theme_type}的议题在你心中很重要。这是很多人都会面对的人生课题。"
            )
        
        # Strategy suggestion (client-centered)
        if strategies:
            strategy_type = strategies[0].get('strategy_type', '')
            if strategy_type == 'Empathic Understanding':
                response_parts.append(
                    "\n也许我们可以一起深入地探索这个想法，看看它背后有什么含义..."
                )
            elif strategy_type == 'Unconditional Positive Regard':
                response_parts.append(
                    "\n无论你的感受是什么，我都接纳你。让我们一起看看这背后有什么故事。"
                )
        
        # Growth-oriented closure
        response_parts.append(
            "\n你觉得这个想法是否值得继续探索呢？"
        )
        
        return "".join(response_parts)
    
    def complete_session(self) -> str:
        """Generate session summary and homework"""
        if not self.session_memory or not self.session_memory.session_context:
            raise ValueError("Session not completed")
        
        memory = self.session_memory
        summary = f"""
=== 第 {memory.client_state.session_number} 次会谈总结 ===

核心议题：
- {memory.client_state.presenting_problem}

识别的自我不一致：
- {'; '.join(memory.client_state.identified_incongruences[:3]) if memory.client_state.identified_incongruences else '待深化探索'}

本次学到的：
- 进一步理解了自我与理想自我之间的差距
- 增进了对生活意义的思考

推荐的家庭作业：
1. 每天留10分钟反思：\"今天我是否为自己活着？\"
2. 列出3-5个你觉得有意义的活动，这周至少做一次
3. 记录一个你感到最真实、最自我的时刻

下次见面时，我们可以继续深化这些探索。
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
        
        filepath = output_path / f"het_case_{case_id}_session_{session_num}.json"
        self.session_memory.save(str(filepath))
