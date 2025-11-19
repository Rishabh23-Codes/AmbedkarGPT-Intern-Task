
# --------------------------------------------------------
# main.py — Assignment 1 (Single Combined Working File)
# --------------------------------------------------------

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from config import HF_EMBEDDING_MODEL,OLLAMA_MODEL,CHROMA_PERSIST_DIR,SPEECH_FILE

from pathlib import Path
from typing import List
import os


# --------------------------------------------------
# LOADER (Assignment-1 only uses speech.txt)
# --------------------------------------------------
def load_single_file(path: str) -> List[Document]:
    loader = TextLoader(path)
    docs = loader.load()
    print("✅ Documents loaded")
    return docs


# --------------------------------------------------
# SPLITTER
# --------------------------------------------------
def make_text_splitter(chunk_size: int=500,overlap:int=50):
    splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    print("✅ Text splitter created")
    return splitter


# --------------------------------------------------
# EMBEDDER
# --------------------------------------------------
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
    print("✅ Embeddings loaded")
    return embeddings


# --------------------------------------------------
# VECTORSTORE
# --------------------------------------------------
def create_chroma(documents,embeddings,persist=True,collection_name="speech_assignment1"):
    persist_dir = str(CHROMA_PERSIST_DIR)
    vect = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name
    )
    print("✅ Vectorstore created")
    return vect


# --------------------------------------------------
# RAG:-  build_retrieval_qa function
# --------------------------------------------------
def build_retrieval_qa(question: str,retriever):
    
    # 1. Retrieve documnents
    docs=retriever.invoke(question)
    print("✅ Documents retrieved")

    # 2. Format context
    context="\n\n".join(doc.page_content for doc in docs)

    # 3. Create prompt
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
    answer=llm.invoke(prompt)
    print("✅ Answer generated")

    return {'answer':answer ,'source_documents':docs}


# --------------------------------------------------
# BUILD PIPELINE FOR ASSIGNMENT 1
# --------------------------------------------------
def assignment1():
    docs=load_single_file(str(SPEECH_FILE))
    splitter=make_text_splitter()
    split_docs=splitter.split_documents(docs)
    print("✅ Documents split into chunks")


    embeddings=get_embeddings()
    vect=create_chroma(documents=split_docs,embeddings=embeddings)
    retriever=vect.as_retriever(
        search_type='similarity',
        search_kwargs={'k':3}
    )
    print("✅ Retriever ready")
    return retriever

# --------------------------------------------------
# CLI LOOP
# --------------------------------------------------
def cli_loop(retriever):
    print("AmbedkarGPT (Assignemnt-1) - type 'exit' to quit.")
    while True:
        print('#------------------------------------------------------------------------------------------------------------------------------------------------------------#')
        q=input("🤖Ask your question Question (write exit to quit): ").strip()
        print('#------------------------------------------------------------------------------------------------------------------------------------------------------------#')
        print("")
        if q.lower() in ('exit','quit'):
            print("✅ Exiting CLI")
            break
        res=build_retrieval_qa(q,retriever)
        print("\nAnswer: ")
        print(res['answer'])


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    retriever=assignment1()
    cli_loop(retriever)

if __name__=="__main__":
    main()
