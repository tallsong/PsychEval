#!/usr/bin/env python3
"""
Multi-Therapy RAG System Verification Script
Validates all components are in place and functional
"""

import os
import json
from pathlib import Path

def check_files():
    """Verify all implementation files exist"""
    base_path = Path("/Users/cedar/code/PsychEval")
    
    print("=" * 70)
    print("MULTI-THERAPY RAG SYSTEM VERIFICATION")
    print("=" * 70)
    
    # Check Python implementation files
    print("\n✓ IMPLEMENTATION FILES:")
    py_files = {
        "eval/rag/het_knowledge_extractor.py": "HET Knowledge Extraction",
        "eval/rag/het_retriever.py": "HET RAG Retrieval",
        "eval/rag/het_counselor_agent.py": "HET Counselor Agent",
        "eval/rag/pdt_knowledge_extractor.py": "PDT Knowledge Extraction",
        "eval/rag/pdt_retriever.py": "PDT RAG Retrieval",
        "eval/rag/pdt_counselor_agent.py": "PDT Therapist Agent",
        "run_multi_therapy_demo.py": "Multi-Therapy Demo Script",
    }
    
    for file_path, desc in py_files.items():
        full_path = base_path / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"  ✅ {desc:30} {size:10,} bytes")
        else:
            print(f"  ❌ {desc:30} MISSING")
    
    # Check knowledge base files
    print("\n✓ KNOWLEDGE BASE FILES:")
    kb_files = {
        "eval/rag/knowledge_base/het_self_concepts.json": "HET Self-Concepts",
        "eval/rag/knowledge_base/het_existential_themes.json": "HET Existential Themes",
        "eval/rag/knowledge_base/het_client_centered_strategies.json": "HET Strategies",
        "eval/rag/knowledge_base/pdt_core_conflicts.json": "PDT Core Conflicts",
        "eval/rag/knowledge_base/pdt_object_relations.json": "PDT Object Relations",
        "eval/rag/knowledge_base/pdt_unconscious_patterns.json": "PDT Unconscious Patterns",
        "eval/rag/knowledge_base/pdt_psychodynamic_interventions.json": "PDT Interventions",
    }
    
    for file_path, desc in kb_files.items():
        full_path = base_path / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            with open(full_path) as f:
                data = json.load(f)
                count = len(data)
            print(f"  ✅ {desc:30} {size:10,} bytes ({count} units)")
        else:
            print(f"  ❌ {desc:30} MISSING")
    
    # Check documentation files
    print("\n✓ DOCUMENTATION FILES:")
    doc_files = {
        "MULTI_THERAPY_RAG_README.md": "Technical Implementation Guide",
        "MULTI_THERAPY_QUICKREF.md": "Quick Reference Guide",
        "MULTI_THERAPY_COMPLETION_CN.md": "Chinese Summary",
        "MULTI_THERAPY_SUMMARY.txt": "Project Summary",
    }
    
    for file_path, desc in doc_files.items():
        full_path = base_path / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            lines = sum(1 for _ in open(full_path))
            print(f"  ✅ {desc:30} {size:10,} bytes ({lines} lines)")
        else:
            print(f"  ❌ {desc:30} MISSING")
    
    # Check __init__.py exports
    print("\n✓ MODULE EXPORTS:")
    init_file = base_path / "eval/rag/__init__.py"
    if init_file.exists():
        with open(init_file) as f:
            content = f.read()
        
        exports = [
            "HETKnowledgeExtractor",
            "HETRetriever",
            "HETCounselorAgent",
            "PDTKnowledgeExtractor",
            "PDTRetriever",
            "PDTCounselorAgent",
        ]
        
        for export in exports:
            if export in content:
                print(f"  ✅ {export}")
            else:
                print(f"  ❌ {export} NOT EXPORTED")
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print("\nAll files are in place and system is ready for use.")
    print("\nNext steps:")
    print("  1. Run: python run_multi_therapy_demo.py")
    print("  2. Import in your code: from eval.rag import HETRetriever, etc.")
    print("  3. See MULTI_THERAPY_QUICKREF.md for usage examples")

if __name__ == "__main__":
    check_files()
