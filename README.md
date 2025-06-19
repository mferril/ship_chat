# SHiP chat — Private Retrieval-Augmented Chatbot

`ship_chat` is a two-part system:

| Role | Where it runs | What it does | Frequency |
|------|---------------|--------------|-----------|
| **Builder** | secure internal server | • extracts text from PDFs<br>• chunks & embeds once<br>• writes a **read-only** Chroma vector DB to `data/chroma/` | **one-time** (or whenever PDFs change) |
| **User** | any laptop | • loads the pre-built DB<br>• retrieves top-k chunks for each query<br>• feeds them to a local Ollama LLM | interactive, ad-hoc |

Everything stays on-prem; no API calls, no telemetry.

---

## Quick start (Builder)

```bash
git clone GitHub_PLACEHOLDER
cd ship_chat
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download models once
./scripts/pull_models.sh          # huggingface + ollama

# Put your PDFs into data/pdfs/
ship_chat build --cfg cfg/builder.yaml
# 👉  data/chroma/ is created – copy/mount it read-only for users
```

## Quick start (User)
```bash
# assume data/chroma/ + cfg/user.yaml are already present
ollama serve &                    # local LLM API
ship_chat chat --cfg cfg/user.yaml
> What is our neutrino DIS background?
```

## Directory Layout
```bash
ship_chat/
├── cfg/               # YAML configs (builder vs user)
├── data/
│   ├── pdfs/          # raw source documents (builder only)
│   └── chroma/        # persisted vector DB (read-only for users)
├── ship_chat/         # importable Python package
│   ├── ingest.py      # PDF to text
│   ├── chunk.py       # text to chunks
│   ├── embed.py       # chunks to vectors
│   ├── vector.py      # Chroma helpers
│   ├── rag.py         # retrieval-augmented generation
│   └── cli.py         # “ship_chat build | chat”
├── scripts/
│   └── pull_models.sh # pulls HuggingFace & Ollama models locally
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Security Notes
* No telemetry – Chroma is initialised with anonymized_telemetry=False.
* No cloud calls – models are downloaded once with scripts/pull_models.sh and then loaded from disk.
* Ollama listens on localhost only.