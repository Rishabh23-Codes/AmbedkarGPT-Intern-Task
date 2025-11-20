import os
from pathlib import Path
from tqdm import tqdm

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from ragas import evaluate
from ragas.metrics import faithfulness,answer_relevancy
from datasets import Dataset

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import random
from config import TEST_DATASET,HF_EMBEDDING_MODEL,OLLAMA_MODEL,CHROMA_PERSIST_DIR,CORPUS_DIR
import json
import numpy as np
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_ollama import OllamaEmbeddings

#--------------------------------------------------------------------------------------------------------------------------------------------------

llm = OllamaLLM(
    model=OLLAMA_MODEL,  
    temperature=0
)
ragas_llm = LangchainLLMWrapper(llm)

# Assign to BOTH metrics
faithfulness.llm = ragas_llm
answer_relevancy.llm = ragas_llm

# Embeddings for Answer Relevancy 
emb = OllamaEmbeddings(model=OLLAMA_MODEL)
ragas_embed = LangchainEmbeddingsWrapper(emb)

answer_relevancy.embeddings = ragas_embed

#--------------------------------------------------------------------------------------------------------------------------------------------------

CHUNK_SIZES={
    'small':random.randint(200,300),
    'medium':random.randint(500,600),
    'large':random.randint(800,1000)
}


####### DOCUMENT LOADERS

def load_corpus(corpus_dir: str):
    """Load all .txt files from corpus directory."""
    corpus=[]
    for fp in sorted(Path(corpus_dir).glob("*.txt")):
        loader=TextLoader(str(fp))
        docs=loader.load()
        for d in docs:
            d.metadata['source']=fp.name
        corpus.extend(docs)
    print("✅ Corpus loaded.")
    return corpus

def load_test_data():
    """Load test questions from JSON file."""
    with open(TEST_DATASET) as f:
        data=json.load(f)['test_questions']
    print("✅ Test data loaded.")
    return data
    

####### TEXT PROCESSING

def make_text_splitter(chunk_size,overlap=50):
    """Create text splitter for chunking documents."""
    splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    print("✅ Text splitter created.")
    return splitter 

def get_embeddings():
    """Get HuggingFaceEmbeddings model"""
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
    print("✅ Embeddings loaded.")
    return embeddings




######## VECTOR STORE

def create_chroma(documents,embeddings,collection_name,persist=True):
    """Create Chroma vector store from documents."""
    persist_dir = str(CHROMA_PERSIST_DIR)
    vect = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name
    )
    print("✅ Chroma vector store created.")
    return vect


######## RAG GENERATION


def build_retrieval_qa(question,retriever):
    """Generate answer using retriever and LLM."""
    docs=retriever.invoke(question)
    context="\n\n".join(doc.page_content for doc in docs)
    prompt=f"""You are a 'STRICT' RAG assistant for answering the context-related answer.
    You must follow these rules:
    1. Use ONLY the information explicitly provided in the context below.
    2. Do NOT use prior knowledge, assumptions, or guesses.
    3. If the context contains the answer to the question, provide it.
    4. If the context does **not** contain the answer, respond ONLY with:
    "The correct/exact answer is not in the provided context.
    cite: Relevant 1 line from context that is more likely to have the solution."

    5. Cite the specific sentences or sections from the context that support your answer.
    6. Do NOT invent, hallucinate, or add extra facts. 
    Your behavior depends on the user's intent:


    --------------------
        Context:{context}
    --------------------
        Question: {question}
        Answer: """

    llm=OllamaLLM(model=OLLAMA_MODEL,temperature=0.1)
    answer=llm.invoke(prompt).strip()
    return answer,docs


######### RETRIEVAL METRICS

def hit_rate(retrieved,gold):
    """Calculate hit rate: percentage of queries with relevant retrievals."""
    hits=sum(1 for r,g in zip(retrieved,gold) if set(r) & set(g))
    return hits/len(gold)

def precision_at_k(retrieved,gold,k):
    """Calculate precision@k: average fraction of top-k docs that are relevant."""
    vals=[]
    for r,g in zip(retrieved,gold):
        vals.append(len(set(r[:k]) & set(g))/k)
    return sum(vals)/len(vals)


def mean_reciprocal_rank(retrieved,gold):
    """Calculate MRR: average reciprocal rank of first relevant document."""
    scores=[]
    for r,g in zip(retrieved,gold):
        rank=0
        for i,d in enumerate(r,1):
            if d in g:
                rank=1/i
                break
        scores.append(rank)
    return sum(scores)/len(scores)

def rouge_l(pred,ref):
    scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
    return scorer.score(ref,pred)['rougeL'].fmeasure

def bleu(pred,ref):
    smoothie=SmoothingFunction().method4
    return sentence_bleu([ref.split()],pred.split(),smoothing_function=smoothie)


####### EVALUATION LOOP

def evaluate_chunks(chunk_size):
    print(f'\nRunning Evaluation for chunk size = {chunk_size}')

    test_data=json.load(open(TEST_DATASET))['test_questions']                                    

    corpus=load_corpus(str(CORPUS_DIR))
    splitter=make_text_splitter(chunk_size)
    split_docs=splitter.split_documents(corpus)
    print("✅ Documents split into chunks.")

    embeddings=get_embeddings()
    vect=create_chroma(documents=split_docs,embeddings=embeddings,collection_name=f'chunk_{chunk_size}')
    retriever=vect.as_retriever(search_kwargs={'k':3})

    preds,golds,contexts,retrieved_docnames,questions=[],[],[],[],[]

    sbert=SentenceTransformer('all-MiniLM-L6-v2')

    for item in tqdm(test_data,desc=f"Chunk={chunk_size} | Generating answers", ncols=100):
        q=item['question']
        gold=item['ground_truth']
        gold_src=item['source_documents']

        answer,docs=build_retrieval_qa(q,retriever)

        preds.append(answer)
        golds.append(gold)
        contexts.append([doc.page_content for doc in docs])
        retrieved_docnames.append([doc.metadata.get('source') for doc in docs])
        questions.append(q)

    print("✅ Answers generated for all test questions.")

    #retrieval metrics
    hrate=hit_rate(retrieved_docnames,[t['source_documents'] for t in test_data])
    # we take top-k = (3) relevant docs
    p_k=precision_at_k(retrieved_docnames,[t['source_documents'] for t in test_data],3)
    mrr=mean_reciprocal_rank(retrieved_docnames,[t['source_documents'] for t in test_data])
    print("✅ Retrieval metrics computed.")

    #ragas metrics
    ragas_dataset=Dataset.from_dict({
        'user_input':questions,
        'response':preds,
        'retrieved_contexts':contexts,
        'reference':golds

    })
    
    ragas_result=evaluate(
        dataset=ragas_dataset,
        metrics=[answer_relevancy,faithfulness],
        run_config=RunConfig(
            timeout=120,        # 2 minutes per single LLM call (Ollama needs this)
            max_retries=2,
            max_workers=1       # ← prevents multiple parallel LLM calls that produce different answers
        ),
        llm=ragas_llm,
        embeddings=ragas_embed
    )
    print("✅ RAGAS metrics computed.")

    df = ragas_result.to_pandas()

    ans_rel = df["faithfulness"].mean(skipna=True)
    faithful = df["answer_relevancy"].mean(skipna=True)


    rouge_scores=[rouge_l(p,g) for p,g in zip(preds,golds)]
    bleu_scores=[bleu(p,g) for p,g in zip(preds,golds)]

    cos_scores=[]
    for p,g in zip(preds,golds):
        v1=sbert.encode(p)
        v2=sbert.encode(g)
        cos_scores.append(cosine_similarity([v1], [v2])[0][0])

    print("✅ Text generation metrics computed.")
    return {
    'chunk_size': chunk_size,
    
    'retrieval_metrics': {
        'hit_rate': round(hrate, 3),
        'precision@k': round(p_k, 3),
        'mrr': round(mrr, 3)
    },

    'ragas_metrics': {
        'answer_relevance_avg': round(float(ans_rel), 3),
        'faithfulness_avg': round(float(faithful), 3)
    },

    'rouge_l_avg': round(float(sum(rouge_scores) / len(rouge_scores)), 3),
    'bleu_avg': round(float(sum(bleu_scores) / len(bleu_scores)), 3),
    'cosine_similarity_avg': round(float(sum(cos_scores) / len(cos_scores)), 3)
}

def main():
    results = {}

    for size, val in CHUNK_SIZES.items():
        # Evaluate chunks
        chunk_results = evaluate_chunks(val)

        # Save results
        results[size + ' : ' + str(val)] = chunk_results
        print('#---------------------------------------------------------------------------------------------------------------#')
        # Print detailed metrics
        print(f"\nEvaluation Results for chunk size {val} (Total chunks: {len(chunk_results.get('split_docs', []))})")
        
        print(f"\n  => Retrieval Metrics:")
        print(f"     • Hit Rate:       {chunk_results['retrieval_metrics']['hit_rate']}")
        print(f"     • Precision@K:    {chunk_results['retrieval_metrics']['precision@k']}")
        print(f"     • MRR:            {chunk_results['retrieval_metrics']['mrr']}")

        print(f"\n  => Answer Quality Metrics:")
        print(f"     • Answer Relevance: {chunk_results['ragas_metrics']['answer_relevance_avg']}")
        print(f"     • Faithfulness:     {chunk_results['ragas_metrics']['faithfulness_avg']}")

        print(f"\n  => Text Generation Metrics:")
        print(f"     • Rouge-L:          {chunk_results['rouge_l_avg']}")
        print(f"     • BLEU:             {chunk_results['bleu_avg']}")
        print(f"     • Cosine Similarity: {chunk_results['cosine_similarity_avg']}")
        print('#---------------------------------------------------------------------------------------------------------------#')


    # Save all results to JSON
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Evaluation Completed. Results saved to test_results.json\n")


if __name__ == '__main__':
    main()


