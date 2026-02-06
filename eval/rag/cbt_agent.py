"""
CBT Counselor Agent

AI counselor that uses RAG to provide CBT-informed therapeutic responses.
Integrates cognitive frameworks, intervention strategies, and session continuity.
"""

from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path

from .retriever import CBTRetriever, RetrievalResult
from .session_memory import CBTSessionMemory
from .knowledge_extractor import CBTKnowledgeExtractor


class CBTCounselorAgent:
    """
    CBT-informed counselor agent with RAG support.
    
    Workflow:
    1. Initialize with client information
    2. For each client turn:
        - Retrieve relevant cognitive frameworks and strategies
        - Inject into system prompt
        - Generate therapeutic response
        - Update session memory
    3. Track progress across sessions
    """
    
    def __init__(
        self,
        retriever: CBTRetriever,
        llm_client: Optional[Any] = None,
        model_name: str = "deepseek-chat",
    ):
        """
        Initialize CBT counselor agent
        
        Args:
            retriever: RAGRetriever instance with loaded knowledge base
            llm_client: LLM client for generating responses (optional for mock mode)
            model_name: Name of LLM model to use
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.model_name = model_name
        self.session_memory: Optional[CBTSessionMemory] = None
        
        self.system_prompt_template = """你是一位经验丰富的认知行为治疗(CBT)咨询师。

你的角色：
1. 帮助来访者识别自动化思维和核心信念
2. 使用苏格拉底式提问引导自我发现
3. 安排行为实验或思维记录作业
4. 跟踪治疗进度并调整策略

治疗阶段：{therapy_stage}
当前会话：第 {session_number} 次

【来访者信息】
{client_summary}

【参考认知框架】
{cognitive_frameworks}

【推荐干预策略】
{intervention_strategies}

【类似案例进展】
{therapy_examples}

指导原则：
- 在初期概念化阶段，重点识别和理解认知模式
- 在核心干预阶段，直接挑战和重组不适应性思维
- 在巩固阶段，加强新技能并制定防复发计划
- 始终保持同理心和非评判态度
- 与来访者进行协作，共同制定目标

请根据上述信息和来访者的反应，提供具有CBT指导的治疗回应。
"""
    
    def initialize_client(
        self,
        case_id: int,
        client_name: str,
        main_problem: str,
        topic: str,
        core_beliefs: Optional[List[str]] = None,
    ) -> CBTSessionMemory:
        """
        Initialize therapy with a new client
        
        Args:
            case_id: Unique case identifier
            client_name: Client's name
            main_problem: Description of main presenting problem
            topic: Therapy topic (e.g., "职业发展", "情绪管理")
            core_beliefs: Initial core beliefs identified
        
        Returns:
            SessionMemory instance for this client
        """
        self.session_memory = CBTSessionMemory(case_id, client_name)
        self.session_memory.initialize_client(
            main_problem=main_problem,
            topic=topic,
            core_beliefs=core_beliefs,
        )
        
        return self.session_memory
    
    def start_session(self, session_number: int) -> str:
        """
        Start a new therapy session
        
        Args:
            session_number: Sequential session number
        
        Returns:
            Opening greeting from counselor
        """
        if not self.session_memory:
            raise RuntimeError("Client not initialized. Call initialize_client first.")
        
        self.session_memory.start_new_session(session_number)
        
        # Generate opening based on session number and stage
        if session_number == 1:
            opening = self._generate_initial_greeting()
        else:
            opening = self._generate_session_greeting(session_number)
        
        return opening
    
    def _generate_initial_greeting(self) -> str:
        """Generate opening for first session"""
        state = self.session_memory.client_state
        return f"""你好 {state.client_name}，欢迎来到我们的咨询。我很高兴认识你。

在我们开始之前，我想了解更多关于你的情况。你提到的主要困扰是：{state.main_problem}

让我们花一些时间仔细探讨这个问题，这样我可以更好地帮助你。首先，你能告诉我这个困扰什么时候开始的吗？有什么特殊的事件引发了这一切吗？
"""
    
    def _generate_session_greeting(self, session_number: int) -> str:
        """Generate greeting for continuation session"""
        state = self.session_memory.client_state
        homework_status = f"完成了{len(state.completed_homework)}个作业" if state.completed_homework else "还没完成作业"
        
        return f"""欢迎回来 {state.client_name}。这是我们的第 {session_number} 次会话。

上次我们讨论了{state.main_problem}。我很高兴看到你一直在参与这个过程。你{homework_status}。

今天你想从哪里开始？有什么新的想法或观察吗？
"""
    
    def process_client_input(
        self,
        client_message: str,
        identified_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process client input and generate CBT-informed response
        
        Args:
            client_message: Client's utterance
            identified_patterns: Cognitive patterns identified in this turn
        
        Returns:
            Dict containing response, retrieved knowledge, and metadata
        """
        if not self.session_memory:
            raise RuntimeError("Client not initialized.")
        
        # Add client message to dialogue
        self.session_memory.add_dialogue("client", client_message)
        
        # Update identified patterns
        if identified_patterns:
            self.session_memory.update_identified_patterns(identified_patterns)
        
        # Retrieve relevant knowledge
        retrieval_result = self._retrieve_knowledge(
            client_message,
            identified_patterns,
        )
        
        # Generate response
        counselor_response = self._generate_response(
            client_message,
            retrieval_result,
        )
        
        # Add counselor response to dialogue
        self.session_memory.add_dialogue("counselor", counselor_response)
        
        # Record retrieved knowledge
        self.session_memory.add_retrieved_knowledge(
            retrieval_result.cognitive_frameworks,
            retrieval_result.intervention_strategies,
        )
        
        return {
            "client_message": client_message,
            "counselor_response": counselor_response,
            "identified_patterns": identified_patterns or [],
            "retrieved_frameworks": retrieval_result.cognitive_frameworks,
            "retrieved_strategies": retrieval_result.intervention_strategies,
            "relevance_scores": retrieval_result.relevance_scores,
        }
    
    def _retrieve_knowledge(
        self,
        client_message: str,
        identified_patterns: Optional[List[str]] = None,
    ) -> RetrievalResult:
        """Retrieve relevant CBT knowledge for client message"""
        state = self.session_memory.client_state
        
        return self.retriever.retrieve(
            client_problem=client_message,
            current_cognitive_patterns=identified_patterns or state.identified_cognitive_patterns,
            therapy_stage=state.current_stage,
            client_topic=state.topic,
            top_k=3,
        )
    
    def _generate_response(
        self,
        client_message: str,
        retrieval_result: RetrievalResult,
    ) -> str:
        """
        Generate therapeutic response using RAG context
        
        If LLM client is available, uses it. Otherwise, generates template response.
        """
        # Prepare context
        context = self._prepare_context(retrieval_result)
        
        # If LLM client available, use it
        if self.llm_client:
            return self._generate_with_llm(client_message, context)
        else:
            # Generate template-based response for demo
            return self._generate_template_response(client_message, context)
    
    def _prepare_context(self, retrieval_result: RetrievalResult) -> str:
        """Prepare context from retrieval results"""
        state = self.session_memory.client_state
        
        # Format cognitive frameworks
        frameworks_text = "识别的认知框架：\n"
        for fw in retrieval_result.cognitive_frameworks[:2]:
            frameworks_text += f"\n- 事件: {fw.get('event', '')}\n"
            frameworks_text += f"  自动化思维: {', '.join(fw.get('automatic_thoughts', []))}\n"
            frameworks_text += f"  认知模式: {', '.join(fw.get('cognitive_patterns', []))}\n"
        
        # Format intervention strategies
        strategies_text = "推荐的干预策略：\n"
        for s in retrieval_result.intervention_strategies[:2]:
            strategies_text += f"\n- {s.get('theme', '')}\n"
            strategies_text += f"  技术: {s.get('technique', '')}\n"
            strategies_text += f"  理由: {s.get('rationale', '')}\n"
        
        return self.system_prompt_template.format(
            therapy_stage=state.current_stage,
            session_number=state.current_session,
            client_summary=state.main_problem,
            cognitive_frameworks=frameworks_text,
            intervention_strategies=strategies_text,
            therapy_examples="(参考案例信息)",
        )
    
    def _generate_with_llm(
        self,
        client_message: str,
        context: str,
    ) -> str:
        """Generate response using LLM client"""
        # This would integrate with actual LLM API
        # Placeholder for now
        return self._generate_template_response(client_message, context)
    
    def _generate_template_response(
        self,
        client_message: str,
        context: str,
    ) -> str:
        """Generate template-based response"""
        state = self.session_memory.client_state
        
        # Simple template-based response
        response_templates = [
            f"我理解你的感受。让我们更深入地看这个想法：""{client_message.split()[0]}...""。你能具体描述一下是什么让你产生了这个想法吗？",
            f"这很有趣。我注意到你提到了'{client_message.split()[-1]}'。这个词对你意味着什么？",
            f"我们来做一个思维记录练习。首先，能具体描述一下当时的情境吗？",
            f"你能帮我理解更多关于这个信念吗？你什么时候开始这样认为的？",
            f"这听起来很真实。但让我们用一些证据来测试这个想法。有没有时候这个想法并不成立？",
        ]
        
        import random
        template = random.choice(response_templates)
        
        # Add homework suggestion based on stage
        if state.current_session == 1:
            homework = "\n\n我想为下次会面分配一个简单的任务。请在一周内记录三个这样的自动化思维，以及当时的情境。这会帮我们更好地了解你的思维模式。"
            template += homework
        
        return template
    
    def complete_session(self, session_notes: str = "") -> Dict[str, Any]:
        """
        Complete current session and summarize
        
        Args:
            session_notes: Therapist notes on session
        
        Returns:
            Session summary and progress indicators
        """
        if not self.session_memory or not self.session_memory.current_session:
            raise RuntimeError("No active session.")
        
        self.session_memory.set_session_notes(session_notes)
        
        # Generate summary
        summary = {
            "session_number": self.session_memory.client_state.current_session,
            "dialogue_turns": len(self.session_memory.current_session.dialogue_history),
            "identified_patterns": self.session_memory.client_state.identified_cognitive_patterns,
            "key_insights": self.session_memory.client_state.key_insights,
            "homework_assigned": self.session_memory.client_state.session_goals[-1:],
            "notes": session_notes,
        }
        
        return summary
    
    def get_session_summary(self) -> str:
        """Get formatted session summary"""
        if not self.session_memory:
            return ""
        
        return self.session_memory.get_client_summary()
    
    def save_session_memory(self, filepath: str) -> None:
        """Save session memory to file"""
        if not self.session_memory:
            raise RuntimeError("No session memory to save.")
        
        self.session_memory.save(filepath)
    
    def load_session_memory(self, filepath: str) -> CBTSessionMemory:
        """Load session memory from file"""
        self.session_memory = CBTSessionMemory.load(filepath)
        return self.session_memory
