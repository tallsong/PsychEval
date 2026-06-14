
## 2024-06-14 - Pre-Tokenization vs Regex in Python Tight Loops
**Learning:** In the RAG retrieval logic, executing `re.compile(r'\w+').findall(...)` on the same `query` string inside a `for item in knowledge_base` loop is a massive performance bottleneck. Because the query doesn't change, re-compiling/matching it thousands of times wastes resources. Also, modifying cached objects directly via `concept['relevance_score'] = score` introduces state-leakage bugs across subsequent runs.
**Action:** Always pre-tokenize the search query into a `set` once before tight retrieval loops, and pass the pre-computed set into `_text_similarity` functions. Also, always invoke `.copy()` on cached elements prior to mutating their relevancy scores.
