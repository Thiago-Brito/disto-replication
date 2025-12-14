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
import random
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


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _map_pt_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Mapeia um item CSV PT (ex.: ENEM) para a forma canonica."""
    article = _clean_text(row.get("TX_ENUNCIADO"))
    question = _clean_text(row.get("TX_INTRODUCAO_ALTERNATIVAS"))
    answer_key = _clean_text(row.get("TX_GABARITO")).upper()
    option_letters = ["A", "B", "C", "D", "E"]
    options = []
    for letter in option_letters:
        txt = _clean_text(row.get(f"TX_ALTERNATIVA_{letter}"))
        if txt:
            options.append((letter, txt))
    if len(options) < 2 or answer_key not in option_letters:
        return None
    correct = None
    distractors: List[str] = []
    for letter, txt in options:
        if letter == answer_key:
            correct = txt
        else:
            distractors.append(txt)
    if not correct or not distractors or (not article and not question):
        return None
    return {
        "article": article or question,
        "question": question or article,
        "answer": correct,
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


def prepare_pt_csv(config: Dict[str, Any], out_dir: Path,
                   sample_train: Optional[int] = None,
                   sample_validation: Optional[int] = None,
                   sample_test: Optional[int] = None) -> None:
    """Carrega CSVs PT (por exemplo ENEM) e gera Parquets flatten."""
    pt_conf = config.get("pt_csv", {})
    csv_dir = pt_conf.get("path")
    if not csv_dir:
        raise ValueError("Config PT (pt_csv) requer 'path'")
    csv_dir = Path(csv_dir)
    if not csv_dir.exists():
        raise FileNotFoundError(f"Diret��rio inexistente: {csv_dir}")

    file_glob = pt_conf.get("file_glob", "*.csv")
    csv_files = sorted(csv_dir.glob(file_glob))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {csv_dir} com glob '{file_glob}'")

    records: List[Dict[str, Any]] = []
    skipped = 0
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Map PT {csv_path.name}"):
            rec = _map_pt_row(row)
            if rec is None:
                skipped += 1
                continue
            records.append(rec)

    if not records:
        raise ValueError("Nenhum registro v��lido encontrado nos CSVs PT")

    split_ratios = pt_conf.get("split_ratios", {"train": 0.8, "validation": 0.1, "test": 0.1})
    if not isinstance(split_ratios, dict) or not split_ratios:
        raise ValueError("pt_csv.split_ratios deve ser um dicion��rio n��o vazio")
    items = list(split_ratios.items())
    split_names = [name for name, _ in items]
    ratio_values: List[float] = []
    for _, value in items:
        if not isinstance(value, (int, float)):
            raise ValueError("Valores de split devem ser num��ricos")
        ratio_values.append(float(value))
    ratio_sum = sum(ratio_values)
    if ratio_sum <= 0:
        raise ValueError("Soma de split_ratios deve ser > 0")
    normalized = [val / ratio_sum for val in ratio_values]

    seed = int(pt_conf.get("seed", 42))
    rng = random.Random(seed)
    rng.shuffle(records)

    total_records = len(records)
    allocated = 0
    columns = ["article", "question", "answer", "distractor", "label"]
    split_dfs: Dict[str, pd.DataFrame] = {}
    for idx, name in enumerate(split_names):
        if idx < len(split_names) - 1:
            count = int(round(normalized[idx] * total_records))
            count = max(0, min(count, total_records - allocated))
        else:
            count = total_records - allocated
        subset = records[allocated: allocated + count]
        allocated += count
        if subset:
            split_dfs[name] = _flatten_records(subset)
        else:
            split_dfs[name] = pd.DataFrame(columns=columns)

    for split_name, df in split_dfs.items():
        if split_name == "train" and sample_train is not None and len(df) > sample_train:
            df = df.sample(n=sample_train, random_state=seed)
        elif split_name == "validation" and sample_validation is not None and len(df) > sample_validation:
            df = df.sample(n=sample_validation, random_state=seed)
        elif split_name == "test" and sample_test is not None and len(df) > sample_test:
            df = df.sample(n=sample_test, random_state=seed)
        out_path = out_dir / f"unified_{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[prepare_dataset] PT Saved {len(df)} rows to {out_path} (skipped={skipped})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets for DISTO reproduction")
    parser.add_argument("--config", type=str, required=True, help="YAML de datasets (ex.: config/datasets.yaml)")
    parser.add_argument("--out", type=str, required=True, help="Diretório de saída dos Parquets flatten")
    parser.add_argument("--datasets", type=str, default=None, help="Lista separada por virgulas de datasets a preparar (ex.: race,pt_csv)")
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

    dataset_list = []
    if args.datasets:
        dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        if "race" in config:
            dataset_list = ["race"]
        else:
            dataset_list = list(config.keys())

    handled = False
    for dataset_name in dataset_list:
        if dataset_name == "race":
            if "race" not in config:
                raise ValueError("Config n��o possui a chave 'race'")
            prepare_race(
                config=config,
                out_dir=out_dir,
                sample_train=args.sample_train,
                sample_validation=args.sample_validation,
                sample_test=args.sample_test,
            )
            handled = True
        elif dataset_name == "pt_csv":
            if "pt_csv" not in config:
                raise ValueError("Config n��o possui a chave 'pt_csv'")
            prepare_pt_csv(
                config=config,
                out_dir=out_dir,
                sample_train=args.sample_train,
                sample_validation=args.sample_validation,
                sample_test=args.sample_test,
            )
            handled = True
        else:
            raise ValueError(f"Dataset '{dataset_name}' n��o suportado")

    if not handled:
        raise ValueError("Nenhum dataset selecionado para preparo (ver --datasets e config)")


if __name__ == "__main__":
    main()
