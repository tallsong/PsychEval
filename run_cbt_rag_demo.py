"""
CBT Counselor RAG System - Complete Workflow Demo

This script demonstrates the end-to-end workflow for:
1. Extracting CBT knowledge from case JSON files
2. Building RAG knowledge base
3. Initializing counselor agent
4. Running multi-session therapy simulations
5. Evaluating counselor responses
"""

import sys
import json
from pathlib import Path
from typing import Optional, List

from eval.rag import (
    CBTKnowledgeExtractor,
    RAGRetriever,
    CBTCounselorAgent,
)
from eval.rag.session_memory import SessionMemory


def step1_extract_knowledge(
    cbt_data_dir: str,
    output_dir: str,
) -> None:
    """
    Step 1: Extract CBT knowledge from JSON cases
    
    Extracts three types of knowledge:
    - Cognitive frameworks (ABC models, patterns)
    - Intervention strategies (techniques, homework)
    - Therapy progress (session goals, outcomes)
    """
    print("\n" + "="*80)
    print("STEP 1: EXTRACT CBT KNOWLEDGE FROM CASE FILES")
    print("="*80)
    
    extractor = CBTKnowledgeExtractor(cbt_data_dir)
    extractor.extract_all()
    extractor.save_knowledge_base(output_dir)
    
    print("\n✓ Knowledge extraction complete!")
    return


def step2_initialize_rag(knowledge_base_dir: str) -> RAGRetriever:
    """
    Step 2: Initialize RAG retriever with knowledge base
    """
    print("\n" + "="*80)
    print("STEP 2: INITIALIZE RAG RETRIEVER")
    print("="*80)
    
    retriever = RAGRetriever(knowledge_base_dir)
    print("\n✓ RAG retriever initialized!")
    return retriever


def step3_demonstrate_retrieval(
    retriever: RAGRetriever,
) -> None:
    """
    Step 3: Demonstrate RAG retrieval with sample scenarios
    """
    print("\n" + "="*80)
    print("STEP 3: DEMONSTRATE RETRIEVAL SYSTEM")
    print("="*80)
    
    # Scenario 1: Anxiety about job change
    print("\n【Scenario 1】来访者困扰：工作转换中的焦虑")
    print("-" * 60)
    
    result = retriever.retrieve(
        client_problem="我对换工作感到非常焦虑，总是想'如果我失败了怎么办？'这让我无法专注",
        current_cognitive_patterns=["Catastrophizing", "Fortune Telling"],
        therapy_stage="initial_conceptualization",
        client_topic="职业发展",
        top_k=2,
    )
    
    print(f"\nRetrieved {len(result.cognitive_frameworks)} cognitive frameworks:")
    for i, fw in enumerate(result.cognitive_frameworks[:2], 1):
        print(f"\n  Framework {i}:")
        print(f"    Event: {fw.get('event', '')[:100]}")
        print(f"    Patterns: {', '.join(fw.get('cognitive_patterns', [])[:3])}")
    
    print(f"\nRetrieved {len(result.intervention_strategies)} intervention strategies:")
    for i, s in enumerate(result.intervention_strategies[:2], 1):
        print(f"\n  Strategy {i}:")
        print(f"    Theme: {s.get('theme', '')}")
        print(f"    Technique: {s.get('technique', '')}")
    
    # Scenario 2: Relationship problem
    print("\n\n【Scenario 2】来访者困扰：关系问题")
    print("-" * 60)
    
    result = retriever.retrieve(
        client_problem="我总是觉得伴侣不爱我，每次他晚点回家我都开始想象最坏的情况",
        current_cognitive_patterns=["Mind Reading", "Personalization"],
        therapy_stage="core_intervention",
        client_topic="人际关系",
        top_k=2,
    )
    
    print(f"\nRetrieved {len(result.cognitive_frameworks)} cognitive frameworks")
    print(f"Relevance Scores: {result.relevance_scores}")
    
    print("\n✓ Retrieval demonstration complete!")


def step4_single_session_demo(retriever: RAGRetriever) -> None:
    """
    Step 4: Demonstrate single session with CBT counselor agent
    """
    print("\n" + "="*80)
    print("STEP 4: SINGLE SESSION DEMO")
    print("="*80)
    
    # Initialize counselor
    counselor = CBTCounselorAgent(retriever)
    
    # Initialize client
    print("\n【Initializing Client】")
    print("-" * 60)
    
    memory = counselor.initialize_client(
        case_id=1,
        client_name="李明",
        main_problem="对工作转换感到焦虑，担心自己无法适应新角色",
        topic="职业发展",
        core_beliefs=["我必须完美", "如果我失败了会很可怕"],
    )
    
    print(f"Client: {memory.client_state.client_name}")
    print(f"Problem: {memory.client_state.main_problem}")
    print(f"Topic: {memory.client_state.topic}")
    
    # Start first session
    print("\n【Starting Session 1】")
    print("-" * 60)
    
    opening = counselor.start_session(session_number=1)
    print(f"\nCounselor: {opening}")
    
    # Client input 1
    print("\n【Client Input 1】")
    print("-" * 60)
    
    client_input1 = "是的，我很担心。新工作涉及很多我没有经验的技能。我不断地想象失败的场景。"
    print(f"Client: {client_input1}")
    
    result = counselor.process_client_input(
        client_input1,
        identified_patterns=["Catastrophizing", "Overgeneralization"],
    )
    
    print(f"\nCounselor: {result['counselor_response'][:300]}...")
    print(f"\nIdentified Patterns: {result['identified_patterns']}")
    print(f"Retrieved Frameworks: {len(result['retrieved_frameworks'])}")
    
    # Client input 2
    print("\n【Client Input 2】")
    print("-" * 60)
    
    client_input2 = "是的，我确实在脑子里反复演练最坏的情况。有一次我甚至想象自己在第一个月就被解雇了。"
    print(f"Client: {client_input2}")
    
    result = counselor.process_client_input(
        client_input2,
        identified_patterns=["Fortune Telling"],
    )
    
    print(f"\nCounselor: {result['counselor_response'][:300]}...")
    
    # Complete session
    print("\n【Session Summary】")
    print("-" * 60)
    
    summary = counselor.complete_session(
        session_notes="Successfully identified catastrophizing pattern. Client engaged well. Assigned thought record homework."
    )
    
    print(f"Session: {summary['session_number']}")
    print(f"Dialogue Turns: {summary['dialogue_turns']}")
    print(f"Identified Patterns: {', '.join(summary['identified_patterns'])}")
    print(f"Key Insights: {len(summary['key_insights'])}")
    print(f"Notes: {summary['notes']}")
    
    print("\n✓ Single session demo complete!")
    return counselor


def step5_multisession_demo(retriever: RAGRetriever) -> None:
    """
    Step 5: Demonstrate multi-session continuity
    """
    print("\n" + "="*80)
    print("STEP 5: MULTI-SESSION CONTINUITY DEMO")
    print("="*80)
    
    counselor = CBTCounselorAgent(retriever)
    
    # Initialize client
    counselor.initialize_client(
        case_id=101,
        client_name="王芳",
        main_problem="社交焦虑症，害怕在公众场合说话",
        topic="情绪管理",
        core_beliefs=["别人会批评我", "我很笨"],
    )
    
    print(f"\nClient: {counselor.session_memory.client_state.client_name}")
    print(f"Topic: {counselor.session_memory.client_state.topic}")
    
    # Session 1
    print("\n【Session 1】Initial Conceptualization")
    print("-" * 60)
    
    counselor.start_session(1)
    counselor.session_memory.update_identified_patterns(["Social Anxiety", "Mind Reading"])
    
    result = counselor.process_client_input(
        "是的，每次我要在会议上发言时，我就开始想象所有人都在评判我",
        identified_patterns=["Mind Reading"],
    )
    print(f"Counselor: {result['counselor_response'][:200]}...")
    
    session1_summary = counselor.complete_session("First session focused on problem formulation")
    
    # Session 2
    print("\n【Session 2】Core Intervention (one week later)")
    print("-" * 60)
    
    counselor.start_session(2)
    counselor.session_memory.update_stage("core_intervention")
    
    result = counselor.process_client_input(
        "上周我按你说的做了。虽然很难，但我在团队会议上发表了看法。没人表现出不同意",
        identified_patterns=[],
    )
    
    print(f"Counselor: {result['counselor_response'][:200]}...")
    counselor.session_memory.add_insight("Client successfully challenged catastrophic predictions in real situation")
    
    session2_summary = counselor.complete_session("Successful behavioral experiment, challenged negative predictions")
    
    # Progress summary
    print("\n【Progress Summary Across Sessions】")
    print("-" * 60)
    
    print(counselor.get_session_summary())
    
    print("\n✓ Multi-session demo complete!")


def step6_evaluation_framework(retriever: RAGRetriever) -> None:
    """
    Step 6: Demonstrate CTRS evaluation framework integration
    """
    print("\n" + "="*80)
    print("STEP 6: EVALUATION FRAMEWORK INTEGRATION")
    print("="*80)
    
    print("\nThe system integrates with existing CTRS (Cognitive Therapy Rating Scale)")
    print("evaluation methods to assess:")
    print("  - Agenda setting and structure")
    print("  - Focus on key cognitions or behaviors")
    print("  - Guided discovery technique")
    print("  - Strategy effectiveness")
    print("  - Collaboration with client")
    print("  - Understanding of conceptualization")
    
    print("\nEvaluation workflow:")
    print("1. Extract dialogue from session")
    print("2. Use CTRS evaluation methods (in eval/methods/)")
    print("3. Generate evaluation report")
    print("4. Iterate on counselor prompts if needed")
    
    print("\n✓ Evaluation framework overview complete!")


def main():
    """Run complete CBT Counselor RAG system workflow"""
    
    print("\n" + "="*80)
    print("CBT COUNSELOR AGENT WITH RAG SYSTEM")
    print("Complete Implementation Workflow")
    print("="*80)
    
    # Configure paths
    project_root = Path(__file__).parent
    cbt_data_dir = project_root / "data" / "cbt"
    kb_output_dir = project_root / "eval" / "rag" / "knowledge_base"
    
    # Check data directory
    if not cbt_data_dir.exists():
        print(f"\nError: CBT data directory not found: {cbt_data_dir}")
        return
    
    print(f"\nProject Root: {project_root}")
    print(f"CBT Data Dir: {cbt_data_dir}")
    print(f"KB Output Dir: {kb_output_dir}")
    
    # Run workflow steps
    try:
        # Step 1: Extract knowledge
        step1_extract_knowledge(str(cbt_data_dir), str(kb_output_dir))
        
        # Step 2: Initialize RAG
        retriever = step2_initialize_rag(str(kb_output_dir))
        
        # Step 3: Demonstrate retrieval
        step3_demonstrate_retrieval(retriever)
        
        # Step 4: Single session demo
        step4_single_session_demo(retriever)
        
        # Step 5: Multi-session demo
        step5_multisession_demo(retriever)
        
        # Step 6: Evaluation framework
        step6_evaluation_framework(retriever)
        
        print("\n" + "="*80)
        print("✓ COMPLETE WORKFLOW EXECUTED SUCCESSFULLY")
        print("="*80)
        
        print("\nNext Steps:")
        print("1. Integrate with GPT5ChatClient for actual LLM generation")
        print("2. Run CTRS evaluation on generated responses")
        print("3. Store session memories for continuity")
        print("4. Fine-tune retrieval thresholds based on evaluation results")
        
    except Exception as e:
        print(f"\n✗ Error in workflow: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
