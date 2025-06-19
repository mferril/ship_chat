# Builder only
from sentence_transformers import SentenceTransformer
import numpy as np, itertools, chromadb
from tqdm import tqdm

def build_index(chunks, cfg):
    # Connect to Chroma (telemetry disabled)
    client = chromadb.PersistentClient(
        path=cfg["db_path"],
        settings=chromadb.config.Settings(anonymized_telemetry=False)
    )
    name = cfg["collection"]
    if name in client.list_collections():
        client.delete_collection(name)
    col = client.create_collection(name)

    # Embedding model
    model = SentenceTransformer(cfg["embedding"]["model"],
                                device=cfg["embedding"]["device"])

    bsz = cfg["embedding"]["batch_size"]
    for i in tqdm(range(0, len(chunks), bsz)):
        batch = chunks[i:i+bsz]
        vecs  = model.encode(batch, convert_to_numpy=True).astype(float).tolist()
        ids   = [f"c_{i+j}" for j in range(len(batch))]
        metas = [{"chunk": i+j} for j in range(len(batch))]
        col.add(ids=ids, documents=batch, embeddings=vecs, metadatas=metas)

    print("All chunks embedded and stored locally in ChromaDB.")
