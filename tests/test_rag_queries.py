import json
from test_rag_query import query_rag


def test_cbt_returns_frameworks():
    out = query_rag('cbt', '我对换工作感到非常焦虑', top_k=1)
    assert out['modality'] == 'cbt'
    assert 'cognitive_frameworks' in out
    assert 'intervention_strategies' in out


def test_het_returns_self_concepts():
    out = query_rag('het', '感觉生活缺乏意义', top_k=1)
    assert out['modality'] == 'het'
    assert 'self_concepts' in out
    assert 'existential_themes' in out


def test_pdt_returns_conflicts():
    out = query_rag('pdt', '反复失败的亲密关系', top_k=1)
    assert out['modality'] == 'pdt'
    assert 'core_conflicts' in out
    assert 'object_relations' in out
