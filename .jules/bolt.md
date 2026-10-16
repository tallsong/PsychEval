## 2026-10-16 - Pre-tokenizing sets in parallel arrays for JSON RAG
**Learning:** In JSON-backed retrievers, passing pre-tokenized sets via parallel arrays into similarity checks eliminates O(N) string processing overhead in tight loops without mutating cached dictionaries.
**Action:** Always pre-compute token sets during knowledge base loading rather than tokenizing on the fly during retrieval.
