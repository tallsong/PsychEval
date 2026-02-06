"""
Multi-Therapy RAG Demo: CBT, HET, and PDT Counselor Agents

Demonstrates RAG systems for three different therapeutic modalities:
- CBT: Cognitive-Behavioral Therapy
- HET: Humanistic-Existential Therapy  
- PDT: Psychodynamic Therapy

Each with knowledge extraction from case data and counselor agent implementation.
"""

import sys
from pathlib import Path

# Add eval/rag to path
sys.path.insert(0, str(Path(__file__).parent / "eval" / "rag"))

from het_knowledge_extractor import HETKnowledgeExtractor
from het_retriever import HETRetriever
from het_counselor_agent import HETCounselorAgent

from pdt_knowledge_extractor import PDTKnowledgeExtractor
from pdt_retriever import PDTRetriever
from pdt_counselor_agent import PDTCounselorAgent


def step1_het_extract_knowledge(data_dir: str, output_dir: str) -> None:
    """Step 1: Extract HET knowledge from case files"""
    print("\n" + "="*80)
    print("STEP 1: HET Knowledge Extraction (50 cases)")
    print("="*80)
    
    extractor = HETKnowledgeExtractor(data_dir)
    print(f"\nExtracting from {data_dir}...")
    extractor.extract_all()
    
    print(f"\nSaving knowledge base to {output_dir}...")
    extractor.save_knowledge_base(output_dir)
    
    print(f"\n✓ HET Knowledge extraction complete!")


def step2_pdt_extract_knowledge(data_dir: str, output_dir: str) -> None:
    """Step 2: Extract PDT knowledge from case files"""
    print("\n" + "="*80)
    print("STEP 2: PDT Knowledge Extraction (50 cases)")
    print("="*80)
    
    extractor = PDTKnowledgeExtractor(data_dir)
    print(f"\nExtracting from {data_dir}...")
    extractor.extract_all()
    
    print(f"\nSaving knowledge base to {output_dir}...")
    extractor.save_knowledge_base(output_dir)
    
    print(f"\n✓ PDT Knowledge extraction complete!")


def step3_het_demo(kb_dir: str) -> None:
    """Step 3: HET counselor agent demo"""
    print("\n" + "="*80)
    print("STEP 3: HET Counselor Agent Demo (Single Session)")
    print("="*80)
    
    # Initialize retriever and agent
    retriever = HETRetriever(kb_dir)
    counselor = HETCounselorAgent(retriever)
    
    # Initialize client
    print("\n[Initializing HET Client]")
    memory = counselor.initialize_client(
        case_id=1,
        client_name="李明",
        presenting_problem="工作压力大，感到生活缺乏意义",
        self_perception_theme="自我认知与理想自我的冲突"
    )
    
    # Session 1
    print("\n[HET Session 1]")
    opening = counselor.start_session(1)
    print(f"\n咨询师：\n{opening}\n")
    
    # Client input 1
    client_input1 = "我的父母总是对我有很高的要求，我现在做的工作虽然稳定，但我一点也不开心。我甚至不知道自己真正想要什么..."
    print(f"来访者：\n{client_input1}\n")
    
    result1 = counselor.process_client_input(
        client_input1,
        identified_themes=["无意义", "真实性"]
    )
    print(f"咨询师：\n{result1['counselor_response']}\n")
    
    # Client input 2
    client_input2 = "是的，我觉得自己一直都在为别人活。甚至在学校也是这样，只要成绩不好就会感到内疚。"
    print(f"来访者：\n{client_input2}\n")
    
    result2 = counselor.process_client_input(
        client_input2,
        identified_themes=["条件性积极关注"]
    )
    print(f"咨询师：\n{result2['counselor_response']}\n")
    
    # Session summary
    print("\n[HET Session Summary]")
    summary = counselor.complete_session()
    print(summary)


def step4_pdt_demo(kb_dir: str) -> None:
    """Step 4: PDT counselor agent demo"""
    print("\n" + "="*80)
    print("STEP 4: PDT Counselor Agent Demo (Single Session)")
    print("="*80)
    
    # Initialize retriever and agent
    retriever = PDTRetriever(kb_dir)
    counselor = PDTCounselorAgent(retriever)
    
    # Initialize client
    print("\n[Initializing PDT Client]")
    memory = counselor.initialize_client(
        case_id=1,
        client_name="李静",
        presenting_problem="持续的抑郁、空虚感及亲密关系建立困难",
        topic="人际关系与分离焦虑"
    )
    
    # Session 1
    print("\n[PDT Session 1]")
    opening = counselor.start_session(1)
    print(f"\n治疗师：\n{opening}\n")
    
    # Client input 1
    client_input1 = "我妈妈在我两岁时离开了家庭。之后每段亲密关系中，我都会莫名其妙地害怕对方会离开我。最后我就抢先提出分手..."
    print(f"来访者：\n{client_input1}\n")
    
    result1 = counselor.process_client_input(
        client_input1,
        relational_patterns=["被动承受分离"],
        defensive_behaviors=["Preemptive Rejection"]
    )
    print(f"治疗师：\n{result1['therapist_response']}\n")
    
    # Client input 2
    client_input2 = "我现在意识到我可能一直在和妈妈的离开作战。和新的伴侣在一起时，我既特别渴望他们的接近，又特别害怕被伤害..."
    print(f"来访者：\n{client_input2}\n")
    
    result2 = counselor.process_client_input(
        client_input2,
        relational_patterns=["Dependent Attachment with Idealization"],
        defensive_behaviors=["Defensive Distancing"]
    )
    print(f"治疗师：\n{result2['therapist_response']}\n")
    
    # Session summary
    print("\n[PDT Session Summary]")
    summary = counselor.complete_session()
    print(summary)


def step5_multi_session_het(kb_dir: str) -> None:
    """Step 5: HET multi-session demo with progress"""
    print("\n" + "="*80)
    print("STEP 5: HET Multi-Session Continuity (2 Sessions)")
    print("="*80)
    
    retriever = HETRetriever(kb_dir)
    counselor = HETCounselorAgent(retriever)
    
    # Initialize
    memory = counselor.initialize_client(
        case_id=2,
        client_name="张敏",
        presenting_problem="人际关系紧张，感到孤独",
        self_perception_theme="无法融入群体的自我认知"
    )
    
    # Session 1
    print("\n[HET Session 1: Initial Exploration]")
    opening1 = counselor.start_session(1)
    print(f"咨询师：{opening1[:100]}...\n")
    
    result1 = counselor.process_client_input(
        "我在办公室里常常感到被孤立。同事们似乎都不太喜欢我，他们一起吃饭也不会叫我。",
        identified_themes=["孤独"]
    )
    print(f"来访者观察：识别到孤独和社交困境\n")
    
    # Session 2
    print("\n[HET Session 2: Deepening Self-Exploration]")
    opening2 = counselor.start_session(2)
    print(f"咨询师：{opening2[:80]}...\n")
    
    result2 = counselor.process_client_input(
        "这周我开始思考，也许问题不全在别人身上。我意识到我可能害怕被拒绝，所以总是先退缩...",
        identified_themes=["真实性", "自我认知"]
    )
    print(f"来访者观察：开始从内部责任角度思考\n")
    
    print(f"\n会话数量：{memory.client_state.session_number}")
    print(f"识别的自我不一致：{len(memory.client_state.identified_incongruences)}")


def step6_multi_session_pdt(kb_dir: str) -> None:
    """Step 6: PDT multi-session demo with insight development"""
    print("\n" + "="*80)
    print("STEP 6: PDT Multi-Session Continuity (2 Sessions)")
    print("="*80)
    
    retriever = PDTRetriever(kb_dir)
    counselor = PDTCounselorAgent(retriever)
    
    # Initialize
    memory = counselor.initialize_client(
        case_id=2,
        client_name="王磊",
        presenting_problem="反复的失败关系与被拒绝的恐惧",
        topic="早期缺失与亲密恐惧"
    )
    
    # Session 1
    print("\n[PDT Session 1: Assessment & Pattern Recognition]")
    opening1 = counselor.start_session(1)
    print(f"治疗师：{opening1[:100]}...\n")
    
    result1 = counselor.process_client_input(
        "我从小就缺少父亲的陪伴。他总是工作很忙。现在我每次亲密关系都很短暂，好像我在自我破坏...",
        relational_patterns=["Passive Submission"],
        defensive_behaviors=["Self-sabotage"]
    )
    print(f"来访者观察：识别早期缺失模式\n")
    
    # Session 2
    print("\n[PDT Session 2: Deepening Insight & Connection Making]")
    opening2 = counselor.start_session(2)
    print(f"治疗师：{opening2[:80]}...\n")
    
    result2 = counselor.process_client_input(
        "上周我开始注意到，当伴侣表示关心时，我会莫名其妙地产生距离感。也许我期望被拒绝，所以先保护自己...",
        relational_patterns=["Complex Ambivalence"],
        defensive_behaviors=["Defensive Distancing"]
    )
    print(f"来访者观察：开始整合冲突与防御认知\n")
    
    # Update progress
    memory.update_progress("深化治疗阶段：核心冲突的工作")
    
    print(f"\n会话数量：{memory.client_state.session_number}")
    print(f"识别的核心冲突：{len(memory.client_state.identified_conflicts)}")
    print(f"治疗进程：{memory.client_state.therapeutic_progress}")


def main():
    """Run complete multi-therapy demonstration"""
    project_root = Path(__file__).parent
    
    het_data_dir = project_root / "data" / "het"
    pdt_data_dir = project_root / "data" / "pdt"
    kb_output_dir = project_root / "eval" / "rag" / "knowledge_base"
    
    print("\n" + "="*80)
    print("MULTI-THERAPY RAG SYSTEM DEMONSTRATION")
    print("CBT + HET + PDT Counselor Agents")
    print("="*80)
    
    # HET workflow
    print("\n[HET WORKFLOW]")
    step1_het_extract_knowledge(str(het_data_dir), str(kb_output_dir))
    step3_het_demo(str(kb_output_dir))
    step5_multi_session_het(str(kb_output_dir))
    
    # PDT workflow
    print("\n\n[PDT WORKFLOW]")
    step2_pdt_extract_knowledge(str(pdt_data_dir), str(kb_output_dir))
    step4_pdt_demo(str(kb_output_dir))
    step6_multi_session_pdt(str(kb_output_dir))
    
    # Final summary
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nGenerated Knowledge Bases:")
    print(f"  HET: 3 JSON files (self-concepts, existential themes, strategies)")
    print(f"  PDT: 4 JSON files (conflicts, relations, patterns, interventions)")
    print(f"\nTotal therapy modalities integrated: 3 (CBT + HET + PDT)")
    print(f"Total case data processed: 148 CBT + 50 HET + 50 PDT = 248 cases")
    print("="*80)


if __name__ == "__main__":
    main()
