"""
negative_sampling.py
--------------------

Gera dataset com negativos para cada split (train/validation/test) a partir dos Parquets flatten.

Métodos implementados:
- Bom (label=1.0): distractor original
- Answer Replication (0.0): distractor = answer
- Random (0.0): distrator aleatório do pool global do treino (≠ d original)
- Farthest in Cluster (0.0): usa KMeans(k=200) nos vetores (CLS DistilRoBERTa || POS TF || NER TF || length)
- BERT [MASK] filling (0.0): substitui 1 token POS {NOUN, VERB, ADJ} por predição de MLM (bert-base-uncased)

Notas de implementação:
- KMeans é treinado sobre o pool único de distractors do split de treino. Para val/test, utiliza-se o mesmo KMeans e o conjunto de membros do cluster para escolher o "mais distante". Caso falhe (cluster vazio, etc.), usa-se Random como fallback.
- Embeddings CLS via distilroberta-base. Para o MLM usamos bert-base-uncased para [MASK] filling.
- Uso de spaCy para POS/NER. Requer `python -m spacy download en_core_web_sm`.
- Inclui limites de amostragem e comprimento máximo para custo reduzido em CPU.

CLI:
    python sampling/negative_sampling.py --in data/processed --out data/ns --k 200 \
        [--max_length 128] [--batch_size 32] [--sample_train 20000 --sample_validation 5000 --sample_test 5000]
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    pipeline,
)

# Support running both as a module (python -m sampling.negative_sampling)
# and as a script (python sampling/negative_sampling.py)
try:
    from sampling.clustering_features import FeatureExtractor, fuse_features
except ModuleNotFoundError:
    import os
    import sys
    # Add project root to sys.path so absolute import works when run as a script
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from sampling.clustering_features import FeatureExtractor, fuse_features


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_cls_embeddings(
    texts: List[str],
    model_name: str = "distilroberta-base",
    device: str | torch.device | None = None,
    max_length: int = 128,
    batch_size: int = 32,
) -> np.ndarray:
    """Computa vetores CLS para uma lista de textos.

    Retorna np.ndarray [N, hidden_size]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_vecs: List[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embeddings (CLS)"):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            # DistilRoBERTa: usa hidden_state do primeiro token (<s>) como CLS
            hidden = out.last_hidden_state  # [B, T, H]
            cls = hidden[:, 0, :].detach().cpu().numpy()  # [B, H]
            all_vecs.append(cls)
    return np.concatenate(all_vecs, axis=0) if all_vecs else np.zeros((0, 768), dtype=np.float32)


def compute_feature_vectors(
    texts: List[str],
    fe: FeatureExtractor,
    emb_model: str = "distilroberta-base",
    device: str | torch.device | None = None,
    max_length: int = 128,
    batch_size: int = 32,
) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
    """Para cada texto, computa embedding CLS e features (POS, NER, length) e concatena.

    Returns:
        all_vecs: np.ndarray [N, D]
        feats: lista de dicionários brutos {pos, ent, length} (para reuso se necessário)
    """
    emb = compute_cls_embeddings(texts, model_name=emb_model, device=device, max_length=max_length, batch_size=batch_size)
    feat_list: List[Dict[str, np.ndarray]] = []
    fused_list: List[np.ndarray] = []
    for i, t in enumerate(tqdm(texts, desc="spaCy features")):
        fs = fe.featurize(t)
        feat_list.append(fs)
        fused = fuse_features(emb[i], fs["pos"], fs["ent"], fs["length"])
        fused_list.append(fused)
    return np.vstack(fused_list), feat_list


def build_random_pool(train_df: pd.DataFrame) -> List[str]:
    # Pool global de distractors únicos do split de treino
    pool = sorted({str(x) for x in train_df["distractor"].astype(str).tolist()})
    return pool


def pick_random_different(pool: List[str], original: str) -> str:
    if not pool:
        return original
    if len(pool) == 1 and pool[0] == original:
        return original
    for _ in range(10):
        cand = random.choice(pool)
        if cand != original:
            return cand
    # fallback
    return pool[0] if pool[0] != original else (pool[1] if len(pool) > 1 else original)


def build_kmeans(pool_vecs: np.ndarray, k: int, seed: int = 42) -> KMeans:
    if len(pool_vecs) < k:
        k = max(2, len(pool_vecs))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    km.fit(pool_vecs)
    return km


def farthest_in_cluster(
    km: KMeans,
    pool_vecs: np.ndarray,
    pool_items: List[str],
    target_vec: np.ndarray,
) -> str | None:
    """Retorna o item do pool no mesmo cluster mais distante do vetor alvo.
    Se cluster não tiver membros, retorna None.
    """
    if pool_vecs.size == 0 or len(pool_items) == 0:
        return None
    c = int(km.predict(target_vec.reshape(1, -1))[0])
    # membros do cluster c
    members = np.where(km.labels_ == c)[0]
    if members.size == 0:
        return None
    # distâncias euclidianas
    diffs = pool_vecs[members] - target_vec.reshape(1, -1)
    dists = np.sqrt((diffs * diffs).sum(axis=1))
    idx = int(np.argmax(dists))
    return pool_items[members[idx]]


def build_fill_mask_pipeline(device: str | torch.device | None = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    mlm = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    mlm.to(device)
    p = pipeline("fill-mask", model=mlm, tokenizer=tok, device=0 if torch.cuda.is_available() else -1, top_k=10)
    return p, tok


def masked_substitution(text: str, fe: FeatureExtractor, fill_mask_pipe, tokenizer) -> str | None:
    """Substitui 1 token com POS em {NOUN, VERB, ADJ} por um candidato de MLM.
    Retorna o texto modificado ou None se não possível.
    """
    if not text or not text.strip():
        return None

    doc = fe.nlp(text)
    candidates = [t for t in doc if t.pos_ in {"NOUN", "VERB", "ADJ"} and not t.is_space]
    if not candidates:
        return None
    tok = random.choice(candidates)
    start, end = tok.idx, tok.idx + len(tok.text)
    masked = text[:start] + " [MASK] " + text[end:]

    try:
        preds = fill_mask_pipe(masked)
    except Exception:
        return None

    original_word = tok.text
    for pred in preds:
        token_str = pred.get("token_str", "").strip()
        if token_str and token_str.lower() != original_word.lower():
            # substitui a primeira ocorrência do [MASK]
            return masked.replace("[MASK]", token_str, 1)
    return None


def process_split(
    df: pd.DataFrame,
    out_path: Path,
    train_pool_texts: List[str],
    train_pool_vecs: np.ndarray,
    km: KMeans,
    fe: FeatureExtractor,
    emb_model: str,
    max_length: int,
    batch_size: int,
) -> None:
    # Pré-carrega pipeline de fill-mask para o split
    fill_mask_pipe, bert_tok = build_fill_mask_pipeline()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preparar vetores do split para farthest (por linha)
    texts_for_vec = df["distractor"].astype(str).tolist()
    split_vecs, split_feat_raw = compute_feature_vectors(
        texts_for_vec, fe, emb_model=emb_model, device=device, max_length=max_length, batch_size=batch_size
    )

    rows = []
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc=f"Negatives -> {out_path.name}")):
        article = str(row.article)
        question = str(row.question)
        answer = str(row.answer)
        d_orig = str(row.distractor)

        # Bom (1.0)
        rows.append({"article": article, "question": question, "answer": answer, "distractor": d_orig, "label": 1.0})

        # Answer Replication (0.0)
        rows.append({"article": article, "question": question, "answer": answer, "distractor": answer, "label": 0.0})

        # Random (0.0)
        d_rand = pick_random_different(train_pool_texts, d_orig)
        rows.append({"article": article, "question": question, "answer": answer, "distractor": d_rand, "label": 0.0})

        # Farthest in Cluster (0.0)
        vec_d = split_vecs[i]
        d_far = farthest_in_cluster(km, train_pool_vecs, train_pool_texts, vec_d)
        if d_far is None or d_far == d_orig:
            d_far = pick_random_different(train_pool_texts, d_orig)
        rows.append({"article": article, "question": question, "answer": answer, "distractor": d_far, "label": 0.0})

        # BERT [MASK] filling (0.0)
        d_mlm = masked_substitution(d_orig, fe, fill_mask_pipe, bert_tok)
        if not d_mlm or d_mlm == d_orig:
            d_mlm = pick_random_different(train_pool_texts, d_orig)
        rows.append({"article": article, "question": question, "answer": answer, "distractor": d_mlm, "label": 0.0})

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    print(f"[negative_sampling] Saved {len(out_df)} rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Negative sampling for DISTO reproduction")
    parser.add_argument("--in", dest="input_dir", type=str, required=True, help="Diretório com unified_{split}.parquet")
    parser.add_argument("--out", dest="output_dir", type=str, required=True, help="Diretório de saída para ns_{split}.parquet")
    parser.add_argument("--k", type=int, default=200, help="Número de clusters KMeans")
    parser.add_argument("--max_length", type=int, default=128, help="Comprimento máximo para embeddings")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size para embeddings")
    parser.add_argument("--sample_train", type=int, default=None, help="Amostra de treino (linhas flatten)")
    parser.add_argument("--sample_validation", type=int, default=None, help="Amostra de validação (linhas flatten)")
    parser.add_argument("--sample_test", type=int, default=None, help="Amostra de teste (linhas flatten)")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória")

    args = parser.parse_args()
    set_seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # Carrega parquets flatten
    train_path = input_dir / "unified_train.parquet"
    val_path = input_dir / "unified_validation.parquet"
    test_path = input_dir / "unified_test.parquet"

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    if args.sample_train is not None:
        train_df = train_df.sample(n=min(args.sample_train, len(train_df)), random_state=args.seed)
    if args.sample_validation is not None:
        val_df = val_df.sample(n=min(args.sample_validation, len(val_df)), random_state=args.seed)
    if args.sample_test is not None:
        test_df = test_df.sample(n=min(args.sample_test, len(test_df)), random_state=args.seed)

    # Pool global de distractors do treino
    train_pool = build_random_pool(train_df)
    print(f"[negative_sampling] Train pool size: {len(train_pool)}")

    # Vetores do pool (para KMeans)
    fe = FeatureExtractor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool_vecs, _ = compute_feature_vectors(
        train_pool, fe, emb_model="distilroberta-base", device=device, max_length=args.max_length, batch_size=args.batch_size
    )

    # KMeans
    km = build_kmeans(pool_vecs, k=args.k, seed=args.seed)
    print(f"[negative_sampling] KMeans trained: k={km.n_clusters}")

    # Processa cada split e salva
    process_split(
        train_df,
        output_dir / "ns_train.parquet",
        train_pool_texts=train_pool,
        train_pool_vecs=pool_vecs,
        km=km,
        fe=fe,
        emb_model="distilroberta-base",
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    process_split(
        val_df,
        output_dir / "ns_validation.parquet",
        train_pool_texts=train_pool,
        train_pool_vecs=pool_vecs,
        km=km,
        fe=fe,
        emb_model="distilroberta-base",
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    process_split(
        test_df,
        output_dir / "ns_test.parquet",
        train_pool_texts=train_pool,
        train_pool_vecs=pool_vecs,
        km=km,
        fe=fe,
        emb_model="distilroberta-base",
        max_length=args.max_length,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
