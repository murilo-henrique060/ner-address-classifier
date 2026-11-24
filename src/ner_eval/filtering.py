from __future__ import annotations

from typing import Iterable, List, Optional, Set, Tuple

# Classes exclusivas do modelo ottema/bert-addresses-brazil
MODEL_CLASSES: Tuple[str, ...] = (
    "RUA",
    "NUMERO",
    "BAIRRO",
    "CIDADE",
    "ESTADO",
    "CEP",
    "COMPLEMENTO",
    "REFERENCIA",
)


def extract_entities(tags_list: Iterable[List[str]]) -> List[str]:
    """Extrai nomes únicos de entidades a partir de listas de tags BIO."""
    entities: Set[str] = set()
    for sentence_tags in tags_list:
        for tag in sentence_tags:
            if tag != "O" and "-" in tag:
                _, ent = tag.split("-", 1)
                entities.add(ent.upper())
    return sorted(entities)


def resolve_measured_classes(
    dataset_entities: List[str],
) -> Tuple[List[str], List[str]]:
    """Calcula a interseção entre as classes do modelo e as classes do dataset.
    
    Classes Medidas = Classes do Modelo ∩ Classes do Dataset.
    Retorna (measured_classes, discarded_classes).
    """
    model_classes_set = set(MODEL_CLASSES)
    dataset_classes_set = set(e.upper() for e in dataset_entities)

    measured = sorted(model_classes_set.intersection(dataset_classes_set))
    discarded = sorted(dataset_classes_set - set(measured))

    return measured, discarded


def filter_tags(tags: List[str], allowed_entities: Set[str]) -> List[str]:
    """Converte para 'O' qualquer tag cuja entidade não esteja em allowed_entities."""
    filtered: List[str] = []
    for tag in tags:
        if tag == "O" or not tag:
            filtered.append("O")
            continue

        if "-" in tag:
            prefix, ent = tag.split("-", 1)
            if ent.upper() in allowed_entities:
                filtered.append(f"{prefix.upper()}-{ent.upper()}")
            else:
                filtered.append("O")
        else:
            if tag.upper() in allowed_entities:
                filtered.append(f"B-{tag.upper()}")
            else:
                filtered.append("O")

    return filtered
