# dump_predictions.py
import argparse, math, os
import torch
import pandas as pd
from transformers import AutoTokenizer
from models.model_sept import SepTDISTO

def build_input(q, a, d, art):
    return f"[QUES] {q} [ANS] {a} [DIS] {d} [ART] {art or ''}"

def load_model(ckpt_path, base_model="distilroberta-base", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(base_model)
    model = SepTDISTO(model_name=base_model)
    state = torch.load(ckpt_path, map_location=device)
    # aceita tanto state puro quanto dict com "model_state_dict"
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return tok, model, device

@torch.no_grad()
def predict_df(df, tok, model, device, max_length=256, batch_size=64):
    texts = [build_input(q, a, d, art)
             for q, a, d, art in zip(df["question"], df["answer"], df["distractor"], df["article"])]
    preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tok(batch_texts, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        y = model(enc).squeeze(-1)
        if y.ndim == 0:  # batch de 1
            y = y.unsqueeze(0)
        preds.extend(y.detach().cpu().tolist())
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Caminho do checkpoint (ex.: runs/disto-sept/best.ckpt)")
    ap.add_argument("--data", required=True, help="Parquet de teste (ex.: data/ns/ns_test.parquet)")
    ap.add_argument("--out", required=True, help="CSV de saída (ex.: runs/disto-sept/test_predictions.csv)")
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    # ler parquet
    try:
        df = pd.read_parquet(args.data)
    except Exception as e:
        raise SystemExit(
            f"Falha ao ler {args.data}. Se for erro de engine, tente:\n"
            f"  pip install pyarrow\nErro original: {e}"
        )

    required_cols = {"article","question","answer","distractor","label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Colunas faltando no parquet {args.data}: {missing}")

    tok, model, device = load_model(args.ckpt, args.model)
    preds = predict_df(df, tok, model, device, args.max_length, args.batch_size)

    out_df = df.copy()
    out_df["y_pred"] = preds
    # label pode estar em 0/1 ou 0-100; normalizamos MAE absoluto no intervalo original
    # aqui mantemos abs_err na mesma escala de 'label'
    out_df["abs_err"] = (out_df["y_pred"] - out_df["label"]).abs()
    # também em porcentagem (0-100)
    out_df["y_pred_pct"] = out_df["y_pred"] * 100
    out_df["label_pct"]   = out_df["label"]   * 100
    out_df["abs_err_pct"] = (out_df["abs_err"] * 100)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[dump_predictions] Salvo: {args.out}  ({len(out_df)} linhas)")

if __name__ == "__main__":
    main()
