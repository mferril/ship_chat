# Builder only
import tiktoken

def split(text: str, max_tok: int, overlap: int, tokenizer_name: str):
    enc = tiktoken.encoding_for_model(tokenizer_name)
    ids, out = enc.encode(text), []
    for start in range(0, len(ids), max_tok - overlap):
        out.append(enc.decode(ids[start : start + max_tok]))
    return out
