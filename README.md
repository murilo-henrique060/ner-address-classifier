# Avaliador de NER para Endereços Brasileiros

Avaliador modular para modelos de Reconhecimento de Entidades Nomeadas (NER) de endereços brasileiros utilizando o modelo **`ottema/bert-addresses-brazil`** e a biblioteca **`seqeval`**.

## Classes Suportadas

O modelo reconhece exclusivamente as 8 classes de endereços:
- `RUA`
- `NUMERO`
- `BAIRRO`
- `CIDADE`
- `ESTADO`
- `CEP`
- `COMPLEMENTO`
- `REFERENCIA`

## Formato do Dataset

Aceita **exclusivamente o formato CoNLL / BIO** (`.conll` ou `.txt`):
- Cada linha contém `token tag_BIO` (ou 4 colunas padrão CoNLL, pegando a primeira como token e a última como tag).
- Linhas vazias separam sentenças.
- Cabeçalhos como `-DOCSTART-` ou linhas comentadas com `#` são automaticamente ignorados.

## Estrutura do Código

```
src/ner_eval/
├── __init__.py
├── reader.py       # Leitor CoNLL simples e eficiente (tokens, tags BIO)
├── predictor.py    # Predição com ottema/bert-addresses-brazil e alinhamento de subtokens
├── filtering.py    # Resolução de classes medidas (interseção) e conversão para 'O'
├── evaluator.py    # Avaliação com seqeval, formatação de tabela e exportação
└── cli.py          # Interface CLI concisa
```

## Como Usar

### 1. Avaliar todos os datasets em `data/input/`
```bash
uv run python -m ner_eval.cli
```

### 2. Avaliar um arquivo CoNLL específico
```bash
uv run python -m ner_eval.cli --input-file data/input/sample_addresses.conll
```

> **Regra da Interseção**: As classes medidas na avaliação são automaticamente calculadas pela interseção entre as classes do modelo e as classes presentes no dataset ($\mathcal{C}_{\text{modelo}} \cap \mathcal{C}_{\text{dataset}}$). Qualquer tag fora desse escopo é convertida para `O`, sem penalizar o modelo por entidades desconhecidas.
