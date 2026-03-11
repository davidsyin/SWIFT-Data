
# train_hawkeye.py
# MIT License
import argparse, os
import numpy as np
import pandas as pd
from hawkeye_crossformer import HawkeyeTrainer, TrainCfg, residual_cluster_scores
from sklearn.metrics import precision_recall_fscore_support

def load_csv(path, time_col, features, resample=None):
    df = pd.read_csv(path)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).set_index(time_col)
    if resample:
        df = df[features].resample(resample).mean().interpolate()
    else:
        df = df[features]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--time_col", default="timestamp")
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--resample", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--len_eps", type=int, default=24)
    ap.add_argument("--stride_eps", type=int, default=12)
    args = ap.parse_args()

    df = load_csv(args.csv_path, args.time_col, args.features, args.resample)
    X = df.values.astype(np.float32)

    cfg = TrainCfg(epochs=args.epochs, device=args.device, len_eps=args.len_eps, stride_eps=args.stride_eps)
    trainer = HawkeyeTrainer(input_dim=X.shape[1], cfg=cfg)
    stats = trainer.fit(X)
    print("Train stats:", stats)

    preds, residuals, starts, tgt_idx = trainer.forecast_series(X)
    scores, labels, mapping = residual_cluster_scores(residuals, k=3)

    timeline_scores = np.zeros(len(df), dtype=np.float32)
    ti = np.clip(tgt_idx, 0, len(df)-1)
    timeline_scores[ti] = scores

    out = df.copy()
    out["hawkeye_score"] = timeline_scores

    labels_path = os.path.splitext(args.csv_path)[0] + "_labels.csv"
    if os.path.exists(labels_path):
        lab = pd.read_csv(labels_path, parse_dates=[args.time_col]).set_index(args.time_col).reindex(out.index).fillna(0.0)
        y_true = lab["label"].values.astype(int)
        thr = np.quantile(out["hawkeye_score"], 0.99)
        y_pred = (out["hawkeye_score"].values >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        print(f"Pointwise metrics @99th pct threshold: Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")

    out_path = os.path.splitext(args.csv_path)[0] + "_hawkeye_scores.csv"
    out.to_csv(out_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
