with open('eval/rag/pdt_retriever.py', 'r') as f:
    content = f.read()

content = content.replace(
    '        client_problem_set = set(self._word_pattern.findall(client_problem.lower()))\n        for relation in self.object_relations:',
    '        for relation in self.object_relations:'
)

content = content.replace(
    '        scored_results = []\n        client_problem_set = set(self._word_pattern.findall(client_problem.lower()))\n        \n        for relation in self.object_relations:',
    '        scored_results = []\n        client_problem_set = set(self._word_pattern.findall(client_problem.lower()))\n        \n        for relation in self.object_relations:'
)

# Fix remaining instances where unparsed strings were passed instead of sets.
content = content.replace(
    'if any(emotion in client_problem for emotion in [\'抛弃\', \'分离\', \'失望\']):',
    'if any(emotion in client_problem for emotion in [\'抛弃\', \'分离\', \'失望\']):'
)

with open('eval/rag/pdt_retriever.py', 'w') as f:
    f.write(content)
