#!/usr/bin/env bash
# Pull all models once, so the pipeline can run fully offline.

set -e

echo "-- Downloading embedding model (from HuggingFace)…"
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
    --local-dir ~/.cache/huggingface/hub/sentence-transformers__all-MiniLM-L6-v2 --quiet

echo "-- Downloading generation model (from Ollama)…"
# ollama pull mistral:7b-instruct

# Default: lightweight reasoning that fits 8-10 GB RAM
ollama pull phi3:mini                 # 3.8 B params, MIT licence

# If you have ≥32 GB RAM and want explicit chain-of-thought, uncomment:
# ollama pull magistral:latest        # 24 B params, open weight

echo "-- All models pulled. You can now run 'ollama serve' and 'ship_chat chat'."
