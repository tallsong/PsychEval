"""
Unified Three-RAG System Demo: CBT, HET, and PDT Counselor Agents

Demonstrates RAG systems for three different therapeutic modalities in a single file:
- CBT: Cognitive-Behavioral Therapy
- HET: Humanistic-Existential Therapy  
- PDT: Psychodynamic Therapy

Each system demonstrates knowledge extraction, RAG retrieval, and counselor agent workflow.
"""

import sys
from pathlib import Path

# Add eval/rag to path
sys.path.insert(0, str(Path(__file__).parent / "eval" / "rag"))

from eval.rag import (
    CBTKnowledgeExtractor,
    CBTRetriever,
    CBTCounselorAgent,
)
from het_knowledge_extractor import HETKnowledgeEretrievextractor
from het_retriever import HETRetriever
from het_counselor_agent import HETCounselorAgent

from pdt_knowledge_extractor import PDTKnowledgeExtractor
from pdt_retriever import PDTRetriever
from pdt_counselor_agent import PDTCounselorAgent


# ============================================================================
# CBT RAG SYSTEM
# ============================================================================

def demo_cbt_rag():
    """Demonstrate CBT RAG System"""
    print("\n" + "="*80)
    print("CBT (Cognitive-Behavioral Therapy) RAG SYSTEM DEMO")
    print("="*80)
    
    project_root = Path(__file__).parent
    cbt_data_dir = project_root / "data" / "cbt"
    kb_output_dir = project_root / "eval" / "rag" / "knowledge_base"
    
    # Step 1: Extract Knowledge
    print("\n▶ Step 1: Extracting CBT Knowledge from Case Files")
    print("-" * 60)
    try:
        extractor = CBTKnowledgeExtractor(str(cbt_data_dir))
        extractor.extract_all()
        extractor.save_knowledge_base(str(kb_output_dir))
        print("✓ Knowledge extraction complete")
    except Exception as e:
        print(f"⚠ Knowledge extraction skipped: {e}")
    
    # Step 2: Initialize RAG Retriever
    print("\n▶ Step 2: Initializing CBT RAG Retriever")
    print("-" * 60)
    try:
        retriever = CBTRetriever(str(kb_output_dir))
        print("✓ RAG retriever initialized")
    except Exception as e:
        print(f"✗ Failed to initialize retriever: {e}")
        return
    
    # Step 3: Demonstrate Retrieval
    print("\n▶ Step 3: CBT RAG Retrieval Example")
    print("-" * 60)
    try:
        result = retriever.retrieve(
            client_problem="我对换工作感到非常焦虑，总是想'如果我失败了怎么办？'",
            # agent来识别
            current_cognitive_patterns=["Catastrophizing", "Fortune Telling"],
            # defined by agent
            therapy_stage="initial_conceptualization",
            # defined by agent
            client_topic="职业发展",
            top_k=2,
        )
        
        print(f"\nRetrieved {len(result.cognitive_frameworks)} cognitive frameworks")
        print(f"Retrieved {len(result.intervention_strategies)} intervention strategies")
        print(f"Relevance scores: {result.relevance_scores}")
    except Exception as e:
        print(f"⚠ Retrieval demo skipped: {e}")
    
    # Step 4: Single Session Demo
    print("\n▶ Step 4: CBT Counselor Agent Single Session")
    print("-" * 60)
    try:
        counselor = CBTCounselorAgent(retriever)
        
        # Initialize client
        memory = counselor.initialize_client(
            case_id=1,
            client_name="李明",
            main_problem="对工作转换感到焦虑",
            topic="职业发展",
            core_beliefs=["我必须完美", "如果我失败了会很可怕"],
        )
        print(f"Client: {memory.client_state.client_name}")
        print(f"Problem: {memory.client_state.main_problem[:50]}...")
        
        # Start session
        opening = counselor.start_session(session_number=1)
        print(f"\n咨询师开场:\n{opening[:150]}...\n")
        
        # Process client input
        client_input = "我的父母总是对我有很高的要求，我现在做的工作虽然稳定，但我一点也不开心。"
        print(f"来访者:\n{client_input}\n")
        
        result = counselor.process_client_input(
            client_input,
            identified_patterns=["Perfectionism", "Conditional Self-Worth"],
        )
        print(f"咨询师:\n{result.get('counselor_response', '')[:150]}...")
        print("✓ CBT session completed")
        
    except Exception as e:
        print(f"⚠ Session demo skipped: {e}")


# ============================================================================
# HET RAG SYSTEM
# ============================================================================

def demo_het_rag():
    """Demonstrate HET (Humanistic-Existential Therapy) RAG System"""
    print("\n" + "="*80)
    print("HET (Humanistic-Existential Therapy) RAG SYSTEM DEMO")
    print("="*80)
    
    project_root = Path(__file__).parent
    het_data_dir = project_root / "data" / "het"
    kb_output_dir = project_root / "eval" / "rag" / "knowledge_base"
    
    # Step 1: Extract Knowledge
    print("\n▶ Step 1: Extracting HET Knowledge from Case Files")
    print("-" * 60)
    try:
        extractor = HETKnowledgeExtractor(str(het_data_dir))
        extractor.extract_all()
        extractor.save_knowledge_base(str(kb_output_dir))
        print("✓ HET knowledge extraction complete")
    except Exception as e:
        print(f"⚠ Knowledge extraction skipped: {e}")
    
    # Step 2: Initialize RAG Retriever
    print("\n▶ Step 2: Initializing HET RAG Retriever")
    print("-" * 60)
    try:
        retriever = HETRetriever(str(kb_output_dir))
        print("✓ HET RAG retriever initialized")
    except Exception as e:
        print(f"✗ Failed to initialize HET retriever: {e}")
        return
    
    # Step 3: Demonstrate Retrieval
    print("\n▶ Step 3: HET RAG Retrieval Example")
    print("-" * 60)
    try:
        result = retriever.retrieve(
            client_problem="工作压力大，感到生活缺乏意义",
            self_perception=None,
            existential_concern="无意义 真实性",
            top_k=2,
        )

        print(f"\nRetrieved {len(result.self_concepts)} self-concepts")
        print(f"Retrieved {len(result.existential_themes)} existential themes")
        print(f"Retrieved {len(result.strategies)} therapeutic strategies")
    except Exception as e:
        print(f"⚠ Retrieval demo skipped: {e}")
    
    # Step 4: Single Session Demo 
    print("\n▶ Step 4: HET Counselor Agent Single Session")
    print("-" * 60)
    try:
        counselor = HETCounselorAgent(retriever)
        
        # Initialize client
        memory = counselor.initialize_client(
            case_id=1,
            client_name="王莉",
            presenting_problem="工作压力大，感到生活缺乏意义",
            self_perception_theme="自我认知与理想自我的冲突"
        )
        print(f"Client: {memory.client_state.client_name}")
        print(f"Problem: {memory.client_state.presenting_problem}")
        
        # Start session
        opening = counselor.start_session(1)
        print(f"\n咨询师开场:\n{opening[:150]}...\n")
        
        # Process client input
        client_input = "我的父母总是对我有很高的要求，我现在做的工作虽然稳定，但我一点也不开心。我甚至不知道自己真正想要什么..."
        print(f"来访者:\n{client_input}\n")
        
        result = counselor.process_client_input(
            client_input,
            identified_themes=["无意义", "真实性"]
        )
        print(f"咨询师:\n{result.get('counselor_response', '')[:150]}...")
        print("✓ HET session completed")
        
    except Exception as e:
        print(f"⚠ Session demo skipped: {e}")


# ============================================================================
# PDT RAG SYSTEM
# ============================================================================

def demo_pdt_rag():
    """Demonstrate PDT (Psychodynamic Therapy) RAG System"""
    print("\n" + "="*80)
    print("PDT (Psychodynamic Therapy) RAG SYSTEM DEMO")
    print("="*80)
    
    project_root = Path(__file__).parent
    pdt_data_dir = project_root / "data" / "pdt"
    kb_output_dir = project_root / "eval" / "rag" / "knowledge_base"
    
    # Step 1: Extract Knowledge
    print("\n▶ Step 1: Extracting PDT Knowledge from Case Files")
    print("-" * 60)
    try:
        extractor = PDTKnowledgeExtractor(str(pdt_data_dir))
        extractor.extract_all()
        extractor.save_knowledge_base(str(kb_output_dir))
        print("✓ PDT knowledge extraction complete")
    except Exception as e:
        print(f"⚠ Knowledge extraction skipped: {e}")
    
    # Step 2: Initialize RAG Retriever
    print("\n▶ Step 2: Initializing PDT RAG Retriever")
    print("-" * 60)
    try:
        retriever = PDTRetriever(str(kb_output_dir))
        print("✓ PDT RAG retriever initialized")
    except Exception as e:
        print(f"✗ Failed to initialize PDT retriever: {e}")
        return
    
    # Step 3: Demonstrate Retrieval
    print("\n▶ Step 3: PDT RAG Retrieval Example")
    print("-" * 60)
    try:
        result = retriever.retrieve(
            client_problem="反复失败的亲密关系",
            relational_patterns=None,
            defensive_behaviors=["Self-sabotage"],
            top_k=2,
        )

        print(f"\nRetrieved {len(result.core_conflicts)} core conflicts")
        print(f"Retrieved {len(result.object_relations)} object relations")
        print(f"Retrieved {len(result.interventions)} therapeutic interventions")
    except Exception as e:
        print(f"⚠ Retrieval demo skipped: {e}")
    
    # Step 4: Single Session Demo
    print("\n▶ Step 4: PDT Counselor Agent Single Session")
    print("-" * 60)
    try:
        counselor = PDTCounselorAgent(retriever)
        
        # Initialize client
        memory = counselor.initialize_client(
            case_id=1,
            client_name="张涛",
            presenting_problem="反复的失败关系与被拒绝的恐惧",
            topic="早期缺失与亲密恐惧"
        )
        print(f"Client: {memory.client_state.client_name}")
        print(f"Problem: {memory.client_state.presenting_problem}")
        
        # Start session
        opening = counselor.start_session(1)
        print(f"\n治疗师开场:\n{opening[:150]}...\n")
        
        # Process client input
        client_input = "我从小就缺少父亲的陪伴。他总是工作很忙。现在我每次亲密关系都很短暂，好像我在自我破坏..."
        print(f"来访者:\n{client_input}\n")
        
        result = counselor.process_client_input(
            client_input,
            relational_patterns=["Passive Submission"],
            defensive_behaviors=["Self-sabotage"]
        )
        print(f"治疗师:\n{result.get('therapist_response', '')[:150]}...")
        print("✓ PDT session completed")
        
    except Exception as e:
        print(f"⚠ Session demo skipped: {e}")


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main():
    """Run complete three-RAG system demonstration"""
    print("\n" + "="*80)
    print("╔═════════════════════════════════════════════════════════════════════════╗")
    print("║           THREE-RAG SYSTEM UNIFIED DEMONSTRATION                        ║")
    print("║     CBT + HET + PDT Counselor Agents in a Single Demo File              ║")
    print("╚═════════════════════════════════════════════════════════════════════════╝")
    print("="*80)
    
    # Run all three RAG systems
    demo_cbt_rag()
    demo_het_rag()
    demo_pdt_rag()
    
    # Summary
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nThree RAG Systems Demonstrated:")
    print("  ✓ CBT (Cognitive-Behavioral Therapy)")
    print("  ✓ HET (Humanistic-Existential Therapy)")
    print("  ✓ PDT (Psychodynamic Therapy)")
    print("\nEach system includes:")
    print("  • Knowledge extraction from case files")
    print("  • RAG retriever initialization and retrieval demo")
    print("  • Counselor agent single session demonstration")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
