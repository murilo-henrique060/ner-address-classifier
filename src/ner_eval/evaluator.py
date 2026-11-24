from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from seqeval.metrics import classification_report
from tabulate import tabulate


def _to_serializable(val: Any) -> Any:
    """Converte tipos numpy para tipos primitivos do Python."""
    if isinstance(val, dict):
        return {k: _to_serializable(v) for k, v in val.items()}
    if hasattr(val, "item"):
        return val.item()
    return val


def evaluate_predictions(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> Dict[str, Any]:
    """Executa a avaliação seqeval com zero_division=0 e retorna o dicionário de métricas."""
    report_dict: Dict[str, Any] = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        output_dict=True,
        zero_division=0,
    )
    return _to_serializable(report_dict)


def format_table(report: Dict[str, Any]) -> str:
    """Formata as métricas do seqeval em uma tabela no terminal."""
    headers = ["Entidade / Média", "Precisão (%)", "Revocação (%)", "F1-Score (%)", "Suporte"]
    rows = []

    for ent, metrics in report.items():
        if isinstance(metrics, dict):
            p = metrics.get("precision", 0.0) * 100
            r = metrics.get("recall", 0.0) * 100
            f = metrics.get("f1-score", 0.0) * 100
            s = metrics.get("support", 0)
            rows.append([ent, f"{p:.2f}%", f"{r:.2f}%", f"{f:.2f}%", str(s)])

    return tabulate(rows, headers=headers, tablefmt="fancy_grid")


def save_report_json(
    report: Dict[str, Any],
    metadata: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Salva o relatório de métricas apenas em formato JSON."""
    path = Path(output_path).with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "metrics": report}, f, indent=2, ensure_ascii=False)

    return path


def _tokens_to_label_studio_result(
    tokens: List[str],
    tags: List[str],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Reconstrói o texto e converte tags BIO para spans no formato result do Label Studio."""
    text = " ".join(tokens)
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = cursor
        end = start + len(token)
        offsets.append((start, end))
        cursor = end + 1

    results: List[Dict[str, Any]] = []
    current_label: Optional[str] = None
    span_start: Optional[int] = None
    span_end: Optional[int] = None

    def flush_span() -> None:
        nonlocal current_label, span_start, span_end
        if current_label is not None and span_start is not None and span_end is not None:
            results.append(
                {
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "value": {
                        "start": span_start,
                        "end": span_end,
                        "text": text[span_start:span_end],
                        "labels": [current_label],
                    },
                }
            )
        current_label = None
        span_start = None
        span_end = None

    for (t_start, t_end), tag in zip(offsets, tags):
        if tag == "O" or not tag:
            flush_span()
            continue

        if "-" in tag:
            prefix, ent = tag.split("-", 1)
            prefix = prefix.upper()
            ent = ent.upper()
        else:
            prefix = "B"
            ent = tag.upper()

        if prefix == "B":
            flush_span()
            current_label = ent
            span_start = t_start
            span_end = t_end
        elif prefix == "I":
            if current_label == ent:
                span_end = t_end
            else:
                flush_span()
                current_label = ent
                span_start = t_start
                span_end = t_end

    flush_span()
    return text, results


def build_label_studio_predictions(
    sentences_tokens: List[List[str]],
    y_pred: List[List[str]],
    model_version: str,
    scores: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Constrói a lista de tasks com predições no formato JSON aceito pelo Label Studio."""
    tasks: List[Dict[str, Any]] = []
    for idx, (tokens, tags) in enumerate(zip(sentences_tokens, y_pred)):
        text, result = _tokens_to_label_studio_result(tokens, tags)
        score = scores[idx] if scores and idx < len(scores) else 1.0
        tasks.append(
            {
                "data": {"text": text},
                "predictions": [
                    {
                        "model_version": model_version,
                        "score": score,
                        "result": result,
                    }
                ],
            }
        )
    return tasks


def save_label_studio_predictions(
    sentences_tokens: List[List[str]],
    y_pred: List[List[str]],
    model_version: str,
    output_path: Union[str, Path],
    scores: Optional[List[float]] = None,
) -> Path:
    """Exporta as predições do modelo em JSON para visualização no Label Studio."""
    path = Path(output_path).with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)

    tasks = build_label_studio_predictions(
        sentences_tokens=sentences_tokens,
        y_pred=y_pred,
        model_version=model_version,
        scores=scores,
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    return path
