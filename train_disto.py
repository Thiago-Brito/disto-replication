"""
train_disto.py
--------------

Treina o modelo SepT (DistilRoBERTa) com MSE para a métrica DISTO.

Entrada:
- Parquets em data/ns/ns_train.parquet e data/ns/ns_validation.parquet

Configuração principal:
- Otimizador: AdamW (lr=1e-5 por padrão)
- Loss: MSE
- Scheduler: linear com 10% warmup
- Early stopping: salva best.ckpt quando MSE de validação cai

CLI:
    python train_disto.py --data data/ns --model distilroberta-base --output runs/disto-sept --epochs 2 --bsz 8 --lr 1e-5

Observações:
- Para Windows, definimos num_workers=0 por padrão.
- Inclui amostragem opcional para rodar rápido em CPU: --sample_train N, --sample_validation N.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from models.model_sept import SepTDISTO


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def evaluate(model: SepTDISTO, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for enc, labels in loader:
            preds = model(enc)
            loss = loss_fn(preds, labels)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SepT (DISTO) model")
    parser.add_argument("--data", type=str, required=True, help="Diretório com ns_{split}.parquet")
    parser.add_argument("--model", type=str, default="distilroberta-base", help="Nome do modelo encoder")
    parser.add_argument("--output", type=str, required=True, help="Diretório de saída para checkpoints")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sample_train", type=int, default=None)
    parser.add_argument("--sample_validation", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(data_dir / "ns_train.parquet")
    val_df = pd.read_parquet(data_dir / "ns_validation.parquet")
    if args.sample_train is not None:
        train_df = train_df.sample(n=min(args.sample_train, len(train_df)), random_state=args.seed)
    if args.sample_validation is not None:
        val_df = val_df.sample(n=min(args.sample_validation, len(val_df)), random_state=args.seed)

    train_ds = NSDataset(train_df)
    val_ds = NSDataset(val_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SepTDISTO(model_name=args.model).to(device)

    collate_fn = collate_fn_builder(model, max_length=args.max_length, device=device)
    train_loader = DataLoader(train_ds, batch_size=args.bsz, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.bsz, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_val = float("inf")
    best_path = out_dir / "best.ckpt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for enc, labels in pbar:
            optimizer.zero_grad(set_to_none=True)
            preds = model(enc)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())
            pbar.set_postfix({"train_mse": f"{np.mean(train_losses):.4f}"})

        train_mse = float(np.mean(train_losses)) if train_losses else float("nan")
        val_mse = evaluate(model, val_loader, loss_fn, device)
        print(f"[train_disto] epoch={epoch} train_mse={train_mse:.4f} val_mse={val_mse:.4f}")

        if val_mse < best_val:
            best_val = val_mse
            torch.save({
                "model_state_dict": model.state_dict(),
                "cfg": {
                    "model": args.model,
                    "max_length": args.max_length,
                }
            }, best_path)
            print(f"[train_disto] Saved best checkpoint to {best_path}")

    print(f"[train_disto] Best val_mse: {best_val:.4f}")


if __name__ == "__main__":
    main()

