# config.py
# Central configuration for paths and default parameters

from pathlib import Path

ROOT = Path(__file__).parent
CORPUS_DIR = ROOT / "corpus"
SPEECH_FILE = ROOT / "speech.txt"
TEST_DATASET = ROOT / "test_dataset.json"
RESULTS_DIR = ROOT / "results"


# Embedding model from HF (local)
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Ollama model name (local), dont use Mistral as my laptop config is not so good to compute this model due to low RAM(low computing power)
OLLAMA_MODEL = "llama3.2:1b"  #size 1.3 GB


# Chroma persist directory
CHROMA_PERSIST_DIR = ROOT / "chroma_db"


