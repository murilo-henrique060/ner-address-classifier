from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

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
    """Executa a avaliação seqeval com division=0 e retorna o dicionário de métricas."""
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


def save_reports(
    report: Dict[str, Any],
    metadata: Dict[str, Any],
    output_prefix: Union[str, Path],
) -> Tuple[Path, Path]:
    """Salva os relatórios nos formatos JSON e CSV."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "metrics": report}, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entity", "precision", "recall", "f1_score", "support"])
        for ent, data in report.items():
            if isinstance(data, dict):
                writer.writerow([
                    ent,
                    round(data.get("precision", 0.0), 4),
                    round(data.get("recall", 0.0), 4),
                    round(data.get("f1-score", 0.0), 4),
                    data.get("support", 0),
                ])

    return json_path, csv_path
