"""
ship_chat/cli.py
----------------
Two entry-points exposed by Typer:

  $ ship_chat build --cfg cfg/builder.yaml   # run once to build the vector store
  $ ship_chat chat  --cfg cfg/user.yaml      # everyday use by end-users
"""

from __future__ import annotations

import itertools
import pathlib
import typer
import yaml

# local imports
from ship_chat.ingest import extract_text
from ship_chat.chunk import split as chunk_split
from ship_chat.embed import build_index               # writes vectors
from ship_chat import vector, rag                     # retrieval + RAG answer

app = typer.Typer(add_completion=False)               # no shell-completion noise


def _load_cfg(path: str | pathlib.Path) -> dict:
    """Load a YAML config file into a dict."""
    return yaml.safe_load(pathlib.Path(path).read_text())


@app.command(help="Ingest PDFs, embed chunks, and persist the Chroma DB.")
def build(
    cfg: str = typer.Option("cfg/builder.yaml", help="Path to builder YAML."),
):
    C = _load_cfg(cfg)

    # 1) gather raw text from every PDF file
    pdf_dir = pathlib.Path(C["pdf_dir"])
    texts = [extract_text(p) for p in pdf_dir.glob("*.pdf")]

    # 2) token-aware chunking
    chunks = list(
        itertools.chain.from_iterable(
            chunk_split(
                t,
                C["chunk"]["max_tokens"],
                C["chunk"]["overlap_tokens"],
                C["chunk"]["tokenizer"],
            )
            for t in texts
        )
    )

    # 3) embed + write to ChromaDB
    build_index(chunks, C)

    typer.secho("-- Vector store built and persisted.", fg=typer.colors.GREEN)


@app.command(help="Chat with the already-built corpus.")
def chat(
    cfg: str = typer.Option("cfg/user.yaml", help="Path to user YAML."),
):
    C = _load_cfg(cfg)

    # Get a collection with the proper embedding function attached
    col = vector.get_collection(C)

    typer.echo("💬  Ask me anything (Ctrl-D to quit)")
    while True:
        try:
            q = input("> ")
        except EOFError:
            break
        if not q.strip():
            continue

        for token in rag.stream_answer(q, col, C):
            typer.echo( typer.style(token, fg=typer.colors.BRIGHT_CYAN),
                        nl=False)   # print without newline
        typer.echo("")                    # final newline


if __name__ == "__main__":
    app()
