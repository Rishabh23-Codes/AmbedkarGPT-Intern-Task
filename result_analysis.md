# 📊 Results Analysis – AmbedkarGPT Evaluation (Assignment 2)

This document contains the complete analysis of the RAG evaluation performed on the **AmbedkarGPT** system using the 25-question test dataset and the 6-document Ambedkar corpus.  
The evaluation compares different chunk sizes and measures retrieval accuracy, answer quality, and semantic similarity.

---

# 🔵 1. Overview of Evaluation
The evaluation measures:
- Retrieval correctness  
- Faithfulness of generated answers  
- Relevance to ground-truth  
- Similarity scores  
- Effects of chunk size  

The system was tested using **three chunking strategies**:
1. **Small chunks** (200–300 chars)  
2. **Medium chunks** (500–600 chars)  
3. **Large chunks** (800–1000 chars)

---

# 🔵 2. Retrieval Performance Summary

### ✔ Hit Rate  
- **Small chunks:** Highest hit rate  
- **Medium chunks:** Good but slightly lower  
- **Large chunks:** Weak, due to large contextual mixing

Small chunks identified relevant text more frequently because they contain focused information.

### ✔ Precision@K  
- **Small chunks:** Best precision  
- **Medium chunks:** Balanced precision  
- **Large chunks:** Lowest precision  

Large chunks often included irrelevant extra text → lowered precision.

### ✔ Mean Reciprocal Rank (MRR)
- **Small chunks:** Highest MRR  
- **Medium chunks:** Moderate  
- **Large chunks:** Lowest  

MRR shows that the correct text appears earlier in results when chunks are small.

---

# 🔵 3. Answer Quality Metrics

### ✔ Faithfulness (RAGAS)
- **Medium chunks:** Highest faithfulness  
- **Small chunks:** Good but sometimes lacked enough context  
- **Large chunks:** Weak; too much irrelevant context introduced hallucination

Medium-sized chunks provided the right balance of context → grounded answers.

### ✔ Answer Relevance (RAGAS)
- **Medium chunks:** Best  
- **Small chunks:** Lower due to incomplete context  
- **Large chunks:** Lowest relevance  

Smaller chunks retrieve correct lines, but sometimes miss supporting sentences → lowers relevance.

### ✔ ROUGE-L Score
- Best in **medium chunks**
- Worst in **large chunks**

ROUGE-L highlights that medium chunks align well with ground truth text.

---

# 🔵 4. Semantic Similarity Metrics

### ✔ BLEU Score
- **Medium chunks:** Highest  
- **Small chunks:** Moderate  
- **Large chunks:** Lowest  

BLEU reflects phrase-level similarity; large chunks caused more paraphrasing → lower scores.

### ✔ Cosine Similarity
- **Medium chunks:** Best  
- **Small chunks:** Good  
- **Large chunks:** Poor  

Large chunks produced answers with mixed topics → weak vector similarity.

---

# 🔵 **Model Limitation Note (Important Update)**  
- The assignment originally recommended **Mistral-7B**, but it could not run on the available laptop due to hardware limitations.  
-  Therefore, all evaluation results were generated using **llama3.2:1b** (a lightweight 1-billion-parameter model).  

- *This smaller model produces simpler answers, lower semantic similarity, and weaker RAGAS scores.  
This explains why metrics such as answer relevance, BLEU, ROUGE-L, and cosine similarity are lower compared to results expected from a 7B model.*  

---

# 🔵 5. Chunk Size Comparison (Overall)

| Metric Category       | Best Performer | Reason |
|-----------------------|----------------|--------|
| Retrieval Accuracy    | Small chunks   | Precise, focused context |
| Faithfulness          | Medium chunks  | Enough context to stay grounded |
| Answer Relevance     | Medium chunks  | Balanced context + precision |
| Semantic Similarity   | Medium chunks  | Stable answer patterns |
| Hallucination Rate    | Small chunks   | Less noise in retrieved text |
| Context Coverage      | Large chunks   | More text per chunk |

### 🟩 **Final Verdict**  
**Medium chunks (500–600 characters) deliver the most stable and overall best performance.**

---

# 🔵 6. Common Failure Cases

### ❌ 1. Retrieval misses due to semantic drift  
Large chunks blend multiple topics → similarity search becomes inaccurate.

### ❌ 2. Missing context in small chunks  
Some answers require multi-sentence context → small chunks provide incomplete information.

### ❌ 3. LLM over-explaining  
Large chunks → LLM mixes multiple paragraphs → increases hallucination.

### ❌ 4. Multiple similar paragraphs  
Ambedkar’s texts often repeat concepts; similarity search sometimes picks the wrong section.

### ❌ 5. Citations mismatch  
The strict RAG prompt sometimes fails when relevant text is split across too many small chunks.

---

# 🔵 7. Key Insights

### Insight 1 — Retrieval needs precision  
Smaller chunks give the most accurate retrieval ranking.

### Insight 2 — LLM needs enough context  
Medium chunks provide the perfect balance so the LLM doesn’t guess.

### Insight 3 — Large chunks cause hallucination  
More text → more noise → less trustworthy answers.

### Insight 4 — Faithfulness depends on chunk size  
The LLM stays grounded only when retrieved context is clean and focused.

---

# 🔵 8. Recommendations

### ✔ Use **500–600 character chunks** as default  
They balance retrieval accuracy and answer quality.

### ✔ Improve chunking using **semantic splitters**  
Better than character-based splitting.

###  ✔ Increase chunk overlap (70–120 chars)  
Higher overlap helps maintain context continuity in Ambedkar’s long argumentative paragraphs and improves both retrieval recall and answer relevance.

### ✔ Increase retriever `top_k` (recommended: 3 → 5)  
A larger `top_k` helps surface more semantically related chunks, especially when the small 1B model struggles with subtle contextual matches.

### ✔ Expand evaluation to larger dataset  
More questions will reveal long-context failure patterns.

---

# 🔵 9. Final Conclusion

The evaluation clearly shows that:

- **Small chunks** are best for retrieval.  
- **Medium chunks** are best for final answer quality.  
- **Large chunks** reduce overall performance.  

Therefore, for AmbedkarGPT, the **optimal configuration** is:

### ⭐ **Chunk size: 500–600 chars**  
### ⭐ **Retriever: Similarity search (k=3)**  
### ⭐ **Embedding model: all-MiniLM-L6-v2**  
### ⭐ **LLM Used in Evaluation: llama3.2:1b (due to hardware limits)**  

🟨 *A larger model like Mistral-7B would significantly improve all result metrics.*

---

**End of Evaluation Report**
