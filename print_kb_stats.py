#!/usr/bin/env python3
"""Print multi-therapy RAG knowledge base statistics"""

import json
from pathlib import Path

kb_dir = Path('eval/rag/knowledge_base')

print('='*70)
print('MULTI-THERAPY RAG SYSTEM - KNOWLEDGE BASE SUMMARY')
print('='*70)

# CBT
print('\nCBT (148 cases):')
files = ['cognitive_frameworks.json', 'intervention_strategies.json', 'therapy_progress.json', 'case_metadata.json']
cbt_total = 0
for f in files:
    fp = kb_dir / f
    if fp.exists():
        with open(fp) as file:
            data = json.load(file)
            size = fp.stat().st_size / 1024
            count = len(data)
            cbt_total += count
            print(f'  {f:<40} {count:>4} units {size:>8.1f} KB')

print(f'  {"CBT Total:":<40} {cbt_total:>4} units')

# HET
print('\nHET (50 cases):')
files = ['het_self_concepts.json', 'het_existential_themes.json', 'het_client_centered_strategies.json']
het_total = 0
for f in files:
    fp = kb_dir / f
    if fp.exists():
        with open(fp) as file:
            data = json.load(file)
            size = fp.stat().st_size / 1024
            count = len(data)
            het_total += count
            print(f'  {f:<40} {count:>4} units {size:>8.1f} KB')

print(f'  {"HET Total:":<40} {het_total:>4} units')

# PDT
print('\nPDT (50 cases):')
files = ['pdt_core_conflicts.json', 'pdt_object_relations.json', 'pdt_unconscious_patterns.json', 'pdt_psychodynamic_interventions.json']
pdt_total = 0
for f in files:
    fp = kb_dir / f
    if fp.exists():
        with open(fp) as file:
            data = json.load(file)
            size = fp.stat().st_size / 1024
            count = len(data)
            pdt_total += count
            print(f'  {f:<40} {count:>4} units {size:>8.1f} KB')

print(f'  {"PDT Total:":<40} {pdt_total:>4} units')

# Grand total
grand_total = cbt_total + het_total + pdt_total
total_size = sum(f.stat().st_size for f in kb_dir.glob('*.json')) / 1024 / 1024

print('\n' + '='*70)
print(f'GRAND TOTAL: {grand_total} knowledge units across 3 therapies')
print(f'TOTAL SIZE:  {total_size:.1f} MB')
print(f'CASES:       248 (148 CBT + 50 HET + 50 PDT)')
print('='*70)
