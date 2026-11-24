from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union


@dataclass
class Sentence:
    """Sentença com tokens e tags no esquema BIO."""
    tokens: List[str]
    tags: List[str]

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.tags):
            raise ValueError(
                f"Tamanho incompatível: {len(self.tokens)} tokens vs {len(self.tags)} tags"
            )

    def __len__(self) -> int:
        return len(self.tokens)


def read_conll(file_path: Union[str, Path], encoding: str = "utf-8") -> List[Sentence]:
    """Lê um dataset CoNLL/BIO (.conll ou .txt).
    
    - Linhas vazias separam sentenças.
    - Ignora linhas de cabeçalho iniciadas com '-DOCSTART-' ou '#'.
    - Primeira coluna é o token e última coluna é a tag BIO.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo CoNLL não encontrado: {path}")

    sentences: List[Sentence] = []
    tokens: List[str] = []
    tags: List[str] = []

    with open(path, "r", encoding=encoding) as f:
        for raw_line in f:
            line = raw_line.strip()

            if line.startswith(("-DOCSTART-", "#")):
                continue

            if not line:
                if tokens:
                    sentences.append(Sentence(tokens=tokens, tags=tags))
                    tokens, tags = [], []
                continue

            parts = line.split()
            token = parts[0]
            tag = parts[-1].upper() if len(parts) > 1 else "O"

            tokens.append(token)
            tags.append(tag)

        if tokens:
            sentences.append(Sentence(tokens=tokens, tags=tags))

    return sentences
