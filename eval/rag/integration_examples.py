"""
Integration Guide: RAG System with Existing PsychEval Framework

This script shows how to integrate the CBT Counselor RAG system with
PsychEval's existing evaluation infrastructure.
"""

# Example 1: Basic Integration with CTRS Evaluation

def example_1_basic_integration():
    """
    Run a CBT counseling session and evaluate with CTRS
    """
    from eval.rag import CBTCounselorAgent, RAGRetriever
    from eval.methods.counselor_ctrs import CTRSEvaluator
    from eval.utils.llm_api import GPT5ChatClient
    
    # Initialize RAG system
    retriever = RAGRetriever("eval/rag/knowledge_base")
    
    # Initialize counselor agent
    # Note: llm_client is optional - can use template responses without it
    counselor = CBTCounselorAgent(retriever)
    
    # Initialize client
    memory = counselor.initialize_client(
        case_id=1,
        client_name="李明",
        main_problem="工作焦虑，担心新工作会失败",
        topic="职业发展",
        core_beliefs=["只有被留下才能算是成功"]
    )
    
    # Start session
    opening = counselor.start_session(1)
    print(f"Opening: {opening}\n")
    
    # Client interaction
    client_input = "是的，我很担心。新工作涉及很多我没有经验的技能。"
    result = counselor.process_client_input(
        client_input,
        identified_patterns=["Catastrophizing"]
    )
    
    print(f"Client: {client_input}")
    print(f"Counselor: {result['counselor_response']}\n")
    print(f"Retrieved {len(result['retrieved_frameworks'])} frameworks")
    print(f"Relevance scores: {result['relevance_scores']}\n")
    
    # Complete session
    summary = counselor.complete_session("Good engagement, identified key patterns")
    
    # Evaluate with CTRS
    # Note: Requires GPT5ChatClient API key
    try:
        evaluator = CTRSEvaluator(
            llm_client=GPT5ChatClient(api_key="your_api_key")
        )
        dialogue = counselor.session_memory.get_dialogue_history()
        scores = evaluator.evaluate(dialogue)
        print(f"CTRS Evaluation: {scores}")
    except Exception as e:
        print(f"Evaluation skipped (no API key): {e}")


# Example 2: Multi-Session Management

def example_2_multi_session():
    """
    Manage multiple sessions with persistent memory
    """
    from eval.rag import CBTCounselorAgent, RAGRetriever
    from pathlib import Path
    
    retriever = RAGRetriever("eval/rag/knowledge_base")
    counselor = CBTCounselorAgent(retriever)
    
    # Initialize client
    memory = counselor.initialize_client(
        case_id=2,
        client_name="王芳",
        main_problem="社交焦虑症，害怕在公众场合说话",
        topic="情绪管理"
    )
    
    # Session 1: Initial Conceptualization
    print("=" * 60)
    print("SESSION 1: Initial Conceptualization")
    print("=" * 60)
    
    counselor.start_session(1)
    
    response1 = counselor.process_client_input(
        "每次要在会议上发言时，我就开始想象所有人都在评判我",
        identified_patterns=["Mind Reading"]
    )
    print(f"Counselor: {response1['counselor_response'][:200]}...")
    
    response2 = counselor.process_client_input(
        "是的，我会想他们认为我很笨，或者表现不好"
    )
    
    summary1 = counselor.complete_session(
        "Identified mind reading pattern, assigned thought record homework"
    )
    
    # Save session memory
    memory_file = Path("sessions/case_2_session_1.json")
    memory_file.parent.mkdir(exist_ok=True)
    counselor.save_session_memory(str(memory_file))
    print(f"\nSession 1 memory saved to {memory_file}")
    
    # Session 2: Core Intervention (one week later)
    print("\n" + "=" * 60)
    print("SESSION 2: Core Intervention")
    print("=" * 60)
    
    counselor.session_memory.update_stage("core_intervention")
    counselor.start_session(2)
    
    response3 = counselor.process_client_input(
        "上周我按你说的做了。虽然很难，但我在团队会议上发表了看法。"
        "有趣的是，没人表现出不同意。有人还说我的想法很好。"
    )
    
    print(f"Counselor: {response3['counselor_response'][:200]}...")
    
    # Record insight
    counselor.session_memory.add_insight(
        "客户成功挑战了'人们会评判我'的预测。现实反馈推翻了灾难化想象。"
    )
    
    summary2 = counselor.complete_session(
        "Successful behavioral experiment, reality testing challenge catastrophic predictions"
    )
    
    # Save updated memory
    counselor.save_session_memory(str(memory_file))
    
    # Print progress summary
    print("\n" + "=" * 60)
    print("PROGRESS SUMMARY")
    print("=" * 60)
    print(counselor.get_session_summary())


# Example 3: Batch Processing Multiple Cases

def example_3_batch_processing():
    """
    Process multiple cases in batch mode for evaluation studies
    """
    from eval.rag import CBTCounselorAgent, RAGRetriever
    from pathlib import Path
    import json
    
    retriever = RAGRetriever("eval/rag/knowledge_base")
    
    # Define test cases
    test_cases = [
        {
            "case_id": 101,
            "name": "小李",
            "problem": "工作压力大，经常失眠",
            "topic": "压力管理",
            "beliefs": ["我必须完美完成所有工作"]
        },
        {
            "case_id": 102,
            "name": "小王",
            "problem": "与同事关系紧张",
            "topic": "人际关系",
            "beliefs": ["别人不喜欢我"]
        },
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\nProcessing case {case['case_id']}: {case['name']}")
        
        counselor = CBTCounselorAgent(retriever)
        memory = counselor.initialize_client(
            case_id=case["case_id"],
            client_name=case["name"],
            main_problem=case["problem"],
            topic=case["topic"],
            core_beliefs=case["beliefs"]
        )
        
        # Simulate a session
        counselor.start_session(1)
        
        test_input = f"是的，{case['problem']}一直困扰着我。"
        result = counselor.process_client_input(test_input)
        
        case_result = {
            "case_id": case["case_id"],
            "name": case["name"],
            "input": test_input,
            "response_length": len(result["counselor_response"]),
            "frameworks_retrieved": len(result["retrieved_frameworks"]),
            "strategies_retrieved": len(result["retrieved_strategies"]),
            "relevance_scores": result["relevance_scores"]
        }
        results.append(case_result)
        
        print(f"  ✓ Retrieved {len(result['retrieved_frameworks'])} frameworks")
        print(f"  ✓ Retrieved {len(result['retrieved_strategies'])} strategies")
    
    # Save batch results
    with open("batch_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Processed {len(results)} cases")
    print("  Results saved to batch_results.json")


# Example 4: Custom Retrieval Strategy

def example_4_custom_retrieval():
    """
    Use advanced retrieval features for specific scenarios
    """
    from eval.rag import RAGRetriever
    
    retriever = RAGRetriever("eval/rag/knowledge_base")
    
    print("=" * 60)
    print("SCENARIO 1: Retrieve for specific cognitive pattern")
    print("=" * 60)
    
    # Get all frameworks for a specific pattern
    catastrophizing_cases = retriever.get_framework_by_pattern("Catastrophizing")
    print(f"Found {len(catastrophizing_cases)} cases with catastrophizing pattern")
    
    # Show first example
    if catastrophizing_cases:
        example = catastrophizing_cases[0]
        print(f"\nExample case ID: {example['case_id']}")
        print(f"Event: {example['event'][:100]}")
        print(f"Automatic thoughts: {example['automatic_thoughts']}")
    
    print("\n" + "=" * 60)
    print("SCENARIO 2: Retrieve strategies for specific stage")
    print("=" * 60)
    
    # Get strategies for consolidation stage
    consolidation_strategies = retriever.get_strategies_by_stage("巩固与结束")
    print(f"Found {len(consolidation_strategies)} consolidation strategies")
    
    # Show first example
    if consolidation_strategies:
        strategy = consolidation_strategies[0]
        print(f"\nTheme: {strategy['theme']}")
        print(f"Technique: {strategy['technique']}")
        print(f"Rationale: {strategy['rationale'][:100]}")


# Example 5: Integration with Manager Framework

def example_5_manager_integration():
    """
    Integrate RAG with PsychEval's evaluation manager
    """
    from eval.rag import CBTCounselorAgent, RAGRetriever
    from eval.manager.base_manager import BaseManager
    import json
    
    class RAGCounselorManager(BaseManager):
        """
        Manager that uses RAG system for dialogue generation
        """
        
        def __init__(self, knowledge_base_dir: str):
            self.retriever = RAGRetriever(knowledge_base_dir)
            self.counselors = {}
        
        def create_counselor_session(self, case_id: int, case_data: dict):
            """Create a RAG-based counselor session"""
            counselor = CBTCounselorAgent(self.retriever)
            
            client_info = case_data.get("client_info", {})
            memory = counselor.initialize_client(
                case_id=case_id,
                client_name=client_info.get("static_traits", {}).get("name", ""),
                main_problem=client_info.get("main_problem", ""),
                topic=client_info.get("topic", ""),
                core_beliefs=client_info.get("core_beliefs", [])
            )
            
            self.counselors[case_id] = counselor
            return counselor
        
        def run_session(self, case_id: int, session_num: int):
            """Run a session with dialogue"""
            counselor = self.counselors.get(case_id)
            if not counselor:
                return None
            
            # Start session
            opening = counselor.start_session(session_num)
            
            # Simulate multiple turns of dialogue
            # In real usage, this would be interactive
            
            return {
                "opening": opening,
                "session_memory": counselor.session_memory.to_dict()
            }
    
    # Usage
    manager = RAGCounselorManager("eval/rag/knowledge_base")
    
    # Create test case
    test_case = {
        "client_info": {
            "static_traits": {"name": "Test User"},
            "main_problem": "Test problem",
            "topic": "职业发展",
            "core_beliefs": ["Test belief"]
        }
    }
    
    counselor = manager.create_counselor_session(999, test_case)
    result = manager.run_session(999, 1)
    
    print("Manager Integration Test:")
    print(f"Opening: {result['opening'][:100]}...")


# Example 6: Load and Resume Session

def example_6_resume_session():
    """
    Load a previous session and continue from where it left off
    """
    from eval.rag import CBTCounselorAgent, RAGRetriever
    from eval.rag.session_memory import SessionMemory
    from pathlib import Path
    
    # Assume we have a saved session
    session_file = Path("sessions/case_2_session_1.json")
    
    if not session_file.exists():
        print("Session file not found. Create one first with example_2_multi_session()")
        return
    
    # Load session memory
    memory = SessionMemory.load(str(session_file))
    
    # Create new agent
    retriever = RAGRetriever("eval/rag/knowledge_base")
    counselor = CBTCounselorAgent(retriever)
    counselor.session_memory = memory
    
    # Continue from previous session
    print(f"Loaded session for {memory.client_state.client_name}")
    print(f"Previous session: {memory.client_state.current_session}")
    print(f"Identified patterns: {', '.join(memory.client_state.identified_cognitive_patterns)}")
    
    # Start new session building on previous
    counselor.session_memory.update_stage("core_intervention")
    opening = counselor.start_session(2)
    
    print(f"\nNew session opening:\n{opening}")


if __name__ == "__main__":
    import sys
    
    examples = {
        "1": ("Basic Integration", example_1_basic_integration),
        "2": ("Multi-Session Management", example_2_multi_session),
        "3": ("Batch Processing", example_3_batch_processing),
        "4": ("Custom Retrieval", example_4_custom_retrieval),
        "5": ("Manager Integration", example_5_manager_integration),
        "6": ("Resume Session", example_6_resume_session),
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in examples:
        example_num = sys.argv[1]
        print(f"\nRunning Example {example_num}: {examples[example_num][0]}")
        print("=" * 70)
        examples[example_num][1]()
    else:
        print("CBT Counselor RAG - Integration Examples")
        print("=" * 70)
        print("\nUsage: python eval/rag/integration_examples.py [1-6]\n")
        for num, (name, _) in examples.items():
            print(f"  {num}. {name}")
        
        print("\nExample: python eval/rag/integration_examples.py 2")
        print("         (Runs the multi-session management example)")
