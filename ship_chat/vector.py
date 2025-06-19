import chromadb, chromadb.config
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

def _make_embedding_function(cfg: dict):
    model  = cfg["embedding"]["model"]
    device = cfg["embedding"].get("device", "cpu")   # default to CPU
    return SentenceTransformerEmbeddingFunction(model_name=model, device=device)

def get_collection(cfg: dict):
    client = chromadb.PersistentClient(
        path=cfg["db_path"],
        settings=chromadb.config.Settings(anonymized_telemetry=False)
    )
    ef = _make_embedding_function(cfg)
    # get_or_create will re-attach the same EF if the collection already exists
    return client.get_or_create_collection(cfg["collection"], embedding_function=ef)

def retrieve(col, query: str, k: int):
    result = col.query(query_texts=[query],
                       n_results=k,
                       include=["documents"])
    return "\n\n---\n\n".join(result["documents"][0])
