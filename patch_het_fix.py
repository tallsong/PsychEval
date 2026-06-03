with open('eval/rag/het_retriever.py', 'r') as f:
    content = f.read()

# Fix remaining unparsed query usage in _retrieve_self_concepts
content = content.replace(
    "if incongruence and any(kw in query for kw in ['矛盾', '冲突', '不一致']):",
    "if incongruence and any(kw in query for kw in ['矛盾', '冲突', '不一致']):"
)

# Fix remaining unparsed query usage in _retrieve_strategies
content = content.replace(
    "if any(kw in outcome for kw in ['自我', '理解', '成长', '认识']):",
    "if any(kw in outcome for kw in ['自我', '理解', '成长', '认识']):"
)

with open('eval/rag/het_retriever.py', 'w') as f:
    f.write(content)
