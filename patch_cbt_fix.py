with open('eval/rag/retriever.py', 'r') as f:
    content = f.read()

# Fix unparsed client_problem usage in other methods if applicable, but we only touched cognitive_frameworks per the plan.

with open('eval/rag/retriever.py', 'w') as f:
    f.write(content)
