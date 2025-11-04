"""
prepare_dataset.py
-------------------

Função: carregar RACE e gerar Parquet flatten com colunas:
    article (str), question (str), answer (str), distractor (str), label (float=1.0)

Implementação principal:
- Carrega via datasets: load_dataset("race", "all") conforme config/datasets.yaml
- Para cada item, resolve resposta correta (se vier como letra) e distratores (opções erradas)
- Gera 1 linha por distractor com label=1.0
- Pula casos com dados inconsistentes
- Salva em data/processed/unified_{split}.parquet

CLI:
    python preprocess/prepare_dataset.py --config config/datasets.yaml --out data/processed [--sample_train N --sample_validation N --sample_test N]

Notas de extensão:
- Para novos datasets (CosmosQA, DREAM, etc.), crie mapeadores específicos que
  emitam (article, question, answer, distractors[]) por exemplo e registre no YAML.

Escolhas de implementação:
- O script valida colunas mínimas e trata exceções, registrando contagens básicas.
- Usa pandas + pyarrow para Parquet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_dataset
import yaml
from tqdm import tqdm


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _map_race_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Mapeia um item do RACE para a forma canônica.

    Esperado no item (varia conforme subsets):
    - article: str
    - question: str
    - options: List[str]
    - answer: str (pode ser letra "A".."D" ou o texto)
    """
    article = item.get("article")
    question = item.get("question")
    options = item.get("options")
    answer = item.get("answer")

    if not isinstance(article, str) or not isinstance(question, str):
        return None
    if not isinstance(options, list) or len(options) == 0:
        return None
    # Resolve resposta: pode vir como letra ou texto
    correct_text: Optional[str] = None
    if isinstance(answer, str):
        if answer in ["A", "B", "C", "D"]:
            idx = ["A", "B", "C", "D"].index(answer)
            if 0 <= idx < len(options):
                correct_text = options[idx]
        else:
            # Pode já ser o texto de uma das opções
            if answer in options:
                correct_text = answer
    if not correct_text:
        # Tente fallback: procurar por match aproximado
        for opt in options:
            if isinstance(opt, str) and opt.strip() == str(answer).strip():
                correct_text = opt
                break
    if not correct_text:
        return None

    distractors = [o for o in options if isinstance(o, str) and o != correct_text]
    if not distractors:
        return None

    return {
        "article": article,
        "question": question,
        "answer": correct_text,
        "distractors": distractors,
    }


def _flatten_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        art = rec["article"]
        q = rec["question"]
        a = rec["answer"]
        for d in rec["distractors"]:
            # Emite uma linha por distractor positivo (label=1.0)
            rows.append({
                "article": art,
                "question": q,
                "answer": a,
                "distractor": d,
                "label": 1.0,
            })
    return pd.DataFrame(rows)


def prepare_race(config: Dict[str, Any], out_dir: Path,
                 sample_train: Optional[int] = None,
                 sample_validation: Optional[int] = None,
                 sample_test: Optional[int] = None) -> None:
    """Carrega RACE e gera Parquets flatten por split.

    Args:
        config: subconfig de "race" com {config: str, splits: {train, validation, test}}
        out_dir: diretório de saída para Parquets
        sample_*: limites opcionais de amostragem por split
    """
    race_conf = config.get("race", {})
    race_name_conf = race_conf.get("config", "all")
    splits = race_conf.get("splits", {})
    if not splits:
        raise ValueError("Config RACE inválida: 'splits' ausente")

    # Carrega dataset RACE nos splits necessários
    for split_key, split_name in splits.items():
        ds = load_dataset("race", race_name_conf, split=split_name)

        # Amostragem opcional para acelerar
        if split_key == "train" and sample_train is not None:
            ds = ds.select(range(min(sample_train, len(ds))))
        elif split_key == "validation" and sample_validation is not None:
            ds = ds.select(range(min(sample_validation, len(ds))))
        elif split_key == "test" and sample_test is not None:
            ds = ds.select(range(min(sample_test, len(ds))))

        mapped: List[Dict[str, Any]] = []
        skipped = 0
        for item in tqdm(ds, desc=f"Map RACE {split_key}"):
            rec = _map_race_item(item)
            if rec is None:
                skipped += 1
                continue
            mapped.append(rec)
        df = _flatten_records(mapped)
        out_path = out_dir / f"unified_{split_key}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[prepare_dataset] Saved {len(df)} rows to {out_path} (skipped={skipped})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets for DISTO reproduction")
    parser.add_argument("--config", type=str, required=True, help="YAML de datasets (ex.: config/datasets.yaml)")
    parser.add_argument("--out", type=str, required=True, help="Diretório de saída dos Parquets flatten")
    parser.add_argument("--sample_train", type=int, default=None, help="Amostra de exemplos de treino")
    parser.add_argument("--sample_validation", type=int, default=None, help="Amostra de exemplos de validação")
    parser.add_argument("--sample_test", type=int, default=None, help="Amostra de exemplos de teste")

    args = parser.parse_args()
    cfg_path = Path(args.config)
    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    config = _load_yaml(cfg_path)
    if not isinstance(config, dict):
        raise ValueError("Config YAML inválida")

    # Por ora suportamos apenas RACE explicitamente; demais datasets devem ser adicionados aqui.
    if "race" not in config:
        raise ValueError("Config deve conter a chave 'race'")

    prepare_race(
        config=config,
        out_dir=out_dir,
        sample_train=args.sample_train,
        sample_validation=args.sample_validation,
        sample_test=args.sample_test,
    )


if __name__ == "__main__":
    main()

