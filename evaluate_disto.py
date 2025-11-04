"""
evaluate_disto.py
------------------

Avalia o modelo SepT treinado: MAE (%) e correlação de Pearson no split de teste.

Entrada:
- Checkpoint em runs/disto-sept/best.ckpt
- Parquet em data/ns/ns_test.parquet

CLI:
    python evaluate_disto.py --ckpt runs/disto-sept/best.ckpt --data data/ns --model distilroberta-base --bsz 16 --max_length 256
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy.stats import pearsonr

from models.model_sept import SepTDISTO


class NSDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, str | float]:
        row = self.df.iloc[idx]
        return {
            "article": str(row["article"]),
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "distractor": str(row["distractor"]),
            "label": float(row["label"]),
        }


def collate_fn_builder(model: SepTDISTO, max_length: int, device: torch.device):
    def _fn(batch: List[Dict[str, str | float]]):
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32, device=device)
        enc = model.encode_concat(batch, max_length=max_length, device=device)
        return enc, labels
    return _fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SepT (DISTO)")
    parser.add_argument("--ckpt", type=str, required=True, help="Caminho do checkpoint best.ckpt")
    parser.add_argument("--data", type=str, required=True, help="Diretório com ns_test.parquet")
    parser.add_argument("--model", type=str, default="distilroberta-base")
    parser.add_argument("--bsz", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)

    args = parser.parse_args()
    data_dir = Path(args.data)

    test_df = pd.read_parquet(data_dir / "ns_test.parquet")
    test_ds = NSDataset(test_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SepTDISTO(model_name=args.model).to(device)

    # Carrega checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    collate_fn = collate_fn_builder(model, max_length=args.max_length, device=device)
    loader = DataLoader(test_ds, batch_size=args.bsz, shuffle=False, num_workers=0, collate_fn=collate_fn)

    preds_list: List[float] = []
    labels_list: List[float] = []
    with torch.no_grad():
        for enc, labels in tqdm(loader, desc="Evaluate"):
            preds = model(enc)
            preds_list.extend(preds.detach().cpu().numpy().tolist())
            labels_list.extend(labels.detach().cpu().numpy().tolist())

    y_pred = np.array(preds_list, dtype=np.float32)
    y_true = np.array(labels_list, dtype=np.float32)

    mae = float(np.mean(np.abs(y_pred - y_true)))
    r, pval = pearsonr(y_pred, y_true)

    print(f"MAE: {mae*100:.2f}%")
    print(f"Pearson r: {r:.4f} (p={pval:.4g})")


if __name__ == "__main__":
    main()

