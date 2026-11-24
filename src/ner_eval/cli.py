from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from ner_eval.evaluator import (
    evaluate_predictions,
    format_table,
    save_label_studio_predictions,
    save_report_json,
)
from ner_eval.filtering import MODEL_CLASSES, extract_entities, filter_tags, resolve_measured_classes
from ner_eval.predictor import AddressNERPredictor, MODEL_NAME
from ner_eval.reader import Sentence, read_conll


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ner-eval",
        description=f"Avaliador de NER de Endereços Brasileiros ({MODEL_NAME}) com seqeval.",
    )
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--input-file",
        "-i",
        type=str,
        help="Caminho para um arquivo CoNLL (.conll ou .txt).",
    )
    input_group.add_argument(
        "--input-dir",
        "-d",
        type=str,
        default="data/input",
        help="Pasta contendo arquivos CoNLL (padrão: data/input).",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Tamanho do lote de inferência (padrão: 32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo de execução ('cuda', 'cpu', 'auto').",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/output",
        help="Diretório para salvar relatórios (padrão: data/output).",
    )
    return parser


def run_evaluation(
    dataset_path: Path,
    predictor: AddressNERPredictor,
    output_dir: Path,
) -> None:
    sentences = read_conll(dataset_path)
    if not sentences:
        print(f"[AVISO] Nenhuma sentença encontrada em {dataset_path.name}")
        return

    dataset_entities = extract_entities(s.tags for s in sentences)
    measured, discarded = resolve_measured_classes(dataset_entities)

    print("\n" + "=" * 65)
    print(f"Dataset: {dataset_path.name} | Total sentenças: {len(sentences)}")
    print(f"• Classes suportadas pelo modelo: {list(MODEL_CLASSES)}")
    print(f"• Classes presentes no dataset:   {dataset_entities}")
    print(f"✓ Classes medidas na avaliação:   {measured}")
    if discarded:
        print(f"✗ Classes ignoradas (-> O):       {discarded}")
    print("=" * 65)

    allowed_set = set(measured)

    # Filtrar ground truth (y_true)
    y_true = [filter_tags(s.tags, allowed_set) for s in sentences]
    words_list = [s.tokens for s in sentences]

    # Predição com alinhamento de subtokens
    raw_y_pred, scores = predictor.predict_with_scores(words_list)
    y_pred = [filter_tags(p, allowed_set) for p in raw_y_pred]

    # Avaliação com seqeval
    report = evaluate_predictions(y_true, y_pred)
    print("\n" + format_table(report) + "\n")

    # Exportar relatório de métricas (apenas JSON) e predições para Label Studio
    report_path = save_report_json(
        report=report,
        metadata={
            "dataset": dataset_path.name,
            "model": MODEL_NAME,
            "measured_classes": measured,
            "discarded_classes": discarded,
            "total_sentences": len(sentences),
        },
        output_path=output_dir / f"report_{dataset_path.stem}.json",
    )
    ls_path = save_label_studio_predictions(
        sentences_tokens=words_list,
        y_pred=raw_y_pred,
        model_version=MODEL_NAME,
        output_path=output_dir / f"predictions_{dataset_path.stem}_label_studio.json",
        scores=scores,
    )
    print(f"Arquivos exportados:\n  • Métricas (JSON):      {report_path}\n  • Label Studio (JSON):  {ls_path}")


def main(args: Optional[List[str]] = None) -> None:
    parser = build_parser()
    opts = parser.parse_args(args)

    if opts.input_file:
        files = [Path(opts.input_file)]
    else:
        in_dir = Path(opts.input_dir)
        if not in_dir.is_dir():
            print(f"[ERRO] Pasta não encontrada: {in_dir}", file=sys.stderr)
            sys.exit(1)
        files = sorted(f for f in in_dir.iterdir() if f.suffix.lower() in (".conll", ".txt", ".bio"))

    if not files:
        print("[ERRO] Nenhum arquivo CoNLL encontrado.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Carregando modelo '{MODEL_NAME}'...")
    predictor = AddressNERPredictor(device=opts.device, batch_size=opts.batch_size)
    output_dir = Path(opts.output_dir)

    for f in files:
        run_evaluation(f, predictor, output_dir)


if __name__ == "__main__":
    main()
