from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# -----------------------
# Config (defaults if keys missing in cfg)
# -----------------------


@dataclass
class HeadTrainCfg:
    epochs: int = 200
    lr: float = 5e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    temperature: float = 1.0
    batch_size: int = 1024
    bg_class_weight: Optional[
        float] = None  # e.g. 0.2 to downweight background (class 0)


# -----------------------
# I/O helpers
# -----------------------


def _load_embeddings_matrix(pt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = torch.load(pt_path)
    embeddings = data["embeddings"]  # Tensor [N, D]
    indices = data["tile_idx"]
    return embeddings.numpy(), indices.numpy()


def _load_index(dataset_root: Path) -> pd.DataFrame:
    idx_path_parquet = dataset_root / "index.parquet"
    if idx_path_parquet.exists():
        return pd.read_parquet(idx_path_parquet).set_index('tile_idx')
    # Fallback to CSV if user chose CSV
    idx_path_csv = dataset_root / "index.csv"
    if idx_path_csv.exists():
        return pd.read_csv(idx_path_csv).set_index('tile_idx')
    raise FileNotFoundError(
        f"index.parquet (or index.csv) not found in {dataset_root}")


def _load_roi_gt(dataset_root: Path,
                 num_classes: int) -> Optional[pd.DataFrame]:
    p = dataset_root / "roi_ratios.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # Ensure expected columns exist (ratio_0..ratio_{C-1})
    needed = [f"ratio_{i}" for i in range(num_classes)]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return None
    return df[["roi_id"] + needed].copy()


# -----------------------
# Model + loss
# -----------------------


class LinearSoftmaxHead(nn.Module):
    """Single linear layer + temperature → softmax; trained on soft ratio targets."""

    def __init__(self,
                 in_dim: int,
                 n_classes: int,
                 p_drop: float = 0.1,
                 temperature: float = 1.0):
        super().__init__()
        self.drop = nn.Dropout(
            p_drop) if p_drop and p_drop > 0 else nn.Identity()
        self.fc = nn.Linear(in_dim, n_classes)
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(x)
        return self.fc(x) / self.temperature  # logits


def _dist_ce_loss(logits: torch.Tensor,
                  target: torch.Tensor,
                  class_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Distributional cross-entropy: -sum_c y_c log p_c, averaged over batch."""
    logp = logits.log_softmax(dim=1)
    loss = -(target * logp)  # (N, C)
    if class_weight is not None:
        loss = loss * class_weight.view(1, -1)
    return loss.sum(dim=1).mean()


def _rmse_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Root Mean Squared Error (RMSE) between predicted and target ratios.
    - Takes logits (not softmaxed)
    - Applies softmax to get probabilities
    - Supports optional per-class weighting
    """
    # convert logits to probabilities
    p = logits.softmax(dim=1)

    # ensure targets are normalized (sum=1)
    target = target / (target.sum(dim=1, keepdim=True) + eps)

    # per-class squared error
    loss = (p - target)**2  # (N, C)

    # apply per-class weights if given
    if class_weight is not None:
        loss = loss * class_weight.view(1, -1)

    # mean over classes and samples, then sqrt
    return torch.sqrt(loss.mean() + eps)


def _kl_div_loss(logits: torch.Tensor,
                 target: torch.Tensor,
                 class_weight: Optional[torch.Tensor] = None,
                 eps: float = 1e-12) -> torch.Tensor:
    """
    KL(target || pred) using logits directly:
    pred = softmax(logits)
    loss = sum_c target_c * (log target_c - log pred_c)
    """
    logp = F.log_softmax(logits, dim=1)  # log(p)
    t = torch.clamp(target, min=eps)
    loss_mat = t * (t.log() - logp)
    if class_weight is not None:
        loss_mat = loss_mat * class_weight.view(1, -1)
    return loss_mat.sum(dim=1).mean()


@torch.no_grad()
def _js_divergence(p: torch.Tensor,
                   q: torch.Tensor,
                   eps: float = 1e-12) -> torch.Tensor:
    """Mean Jensen–Shannon divergence between rows of p and q (p,q in simplex)."""
    m = 0.5 * (p + q)
    kl_pm = (p * (p.add(eps).log() - m.add(eps).log())).sum(dim=1)
    kl_qm = (q * (q.add(eps).log() - m.add(eps).log())).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm).mean()


# -----------------------
# Metrics
# -----------------------


def _per_class_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       eps: float = 1e-12) -> Dict[str, np.ndarray]:
    diff = y_pred - y_true
    mae = np.mean(np.abs(diff), axis=0)
    rmse = np.sqrt(np.mean(diff**2, axis=0))
    brier = np.mean(diff**2, axis=0)
    ce = -np.mean(y_true * np.log(y_pred + eps), axis=0)
    bias = np.mean(diff, axis=0)

    C = y_true.shape[1]
    pear = np.zeros(C)
    spear = np.zeros(C)
    r2 = np.zeros(C)
    for c in range(C):
        try:
            pear[c] = pearsonr(y_true[:, c], y_pred[:, c])[0]
        except Exception:
            pear[c] = np.nan
        try:
            spear[c] = spearmanr(y_true[:, c],
                                 y_pred[:, c]).correlation  # type: ignore
        except Exception:
            spear[c] = np.nan
        var_y = np.var(y_true[:, c])
        r2[c] = 1.0 - np.mean(
            (y_true[:, c] - y_pred[:, c])**2) / (var_y + 1e-12)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Brier": brier,
        "CE": ce,
        "Bias": bias,
        "Pearson_r": pear,
        "Spearman_rho": spear,
        "R2": r2
    }


def _summarize_metrics(per_class: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {
        f"{k}_macro":
        (np.nanmean(v) if "Pearson" in k or "Spearman" in k else float(
            np.mean(v)))
        for k, v in per_class.items()
    }  # type: ignore


# -----------------------
# Core training
# -----------------------


def soft_to_hard_labels_torch(Y: torch.Tensor,
                              bg_thresh: float = 0.9,
                              fg_thresh: float = 0.1) -> torch.Tensor:
    """
    Convert soft ratio targets (N, C) to integer class labels (0..C-1).
    """
    bg = Y[:, 0]
    non_bg_max, _ = Y[:, 1:].max(dim=1)
    cond = (bg < bg_thresh) & (non_bg_max > fg_thresh)
    labels = torch.zeros(len(Y), dtype=torch.long, device=Y.device)
    labels[cond] = Y[cond].argmax(dim=1)
    num_classes = Y.shape[1]

    return F.one_hot(labels, num_classes=num_classes).to(Y.device)


def _fit_one_fold(
    X: np.ndarray, Y: np.ndarray, tr_idx: np.ndarray, va_idx: np.ndarray,
    cfg: HeadTrainCfg, device: torch.device
) -> Tuple[LinearSoftmaxHead, float, np.ndarray, np.ndarray]:
    """Train one fold; return best model, best JS, y_true_val, y_pred_val."""
    in_dim = X.shape[1]
    n_classes = Y.shape[1]
    model = LinearSoftmaxHead(in_dim,
                              n_classes,
                              p_drop=cfg.dropout,
                              temperature=cfg.temperature).to(device)

    cw = None
    if cfg.bg_class_weight is not None:
        w = np.ones(n_classes, dtype=np.float32)
        w[0] = float(cfg.bg_class_weight)
        cw = torch.tensor(w, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(  # type: ignore
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    # minibatch SGD from numpy arrays
    def _iterate(idx: np.ndarray, shuffle: bool):
        N = len(idx)
        bs = cfg.batch_size
        order = np.arange(N)
        if shuffle:
            np.random.shuffle(order)
        for start in range(0, N, bs):
            sl = order[start:start + bs]
            yield idx[sl]

    best_js = float("inf")
    best_state = None

    X_tr = torch.from_numpy(X[tr_idx])
    Y_tr = torch.from_numpy(Y[tr_idx])
    X_va = torch.from_numpy(X[va_idx]).to(device)
    Y_va = torch.from_numpy(Y[va_idx]).to(device)
    Y_tr = soft_to_hard_labels_torch(Y_tr)
    # Y_va = soft_to_hard_labels_torch(Y_va)

    y_mean = Y_tr.to(torch.float32).mean(dim=0)  # (C,)
    class_weight = 1.0 / (y_mean + 1e-8)
    class_weight = class_weight / class_weight.mean()  # optional normalization
    class_weight = class_weight.to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weight)  # cw is optional class weight
    Y_tr = torch.argmax(Y_tr, dim=1)  # convert to integer labels for CE loss
    # Y_va = torch.argmax(Y_va, dim=1)  # convert to integer labels for CE loss

    for _ in range(cfg.epochs):
        model.train()
        for b_idx in _iterate(np.arange(len(tr_idx)), shuffle=True):
            xb = X_tr[b_idx].to(device, non_blocking=True)
            yb = Y_tr[b_idx].to(device, non_blocking=True)
            logits = model(xb)
            # loss = _dist_ce_loss(logits, yb, class_weight=class_weight)
            # loss = _kl_div_loss(logits, yb, class_weight=class_weight)
            # loss = _rmse_loss(logits, yb, class_weight=class_weight)
            loss = criterion(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sch.step()

        # validate (JS)
        model.eval()
        with torch.no_grad():
            logits = model(X_va)
            p = logits.softmax(dim=1)
            js = float(_js_divergence(p, Y_va))
        if js < best_js:
            best_js = js
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    assert best_state is not None
    model.load_state_dict(best_state)
    # val predictions
    with torch.no_grad():
        p = model(X_va).softmax(dim=1).cpu().numpy()
    return model, best_js, Y[va_idx], p


# -----------------------
# Public entrypoint used by your runner
# -----------------------


def run_linear_cv_and_eval(cfg,
                           pt_filepath: str,
                           encoder_name: str,
                           save_full_fit: bool = True) -> Dict[str, str]:
    """
    - loads embeddings (tile order == index.parquet row order)
    - cross-validated training (stratified by KMeans on ratios, grouped by roi_id)
    - saves heads/fold_k.pt (+ optional full_fit.pt)
    - writes OOF preds + metrics (tile + ROI level)
    """

    # rng = np.random.default_rng(getattr(cfg.run, "seed", 1337))

    torch.manual_seed(int(getattr(cfg.run, "seed", 1337)))

    # paths
    dataset_root = Path(cfg.dataset.root_dir)
    pt_path = Path(pt_filepath)
    out_heads = Path("heads")
    out_heads.mkdir(parents=True, exist_ok=True)

    # ---- load data
    X, _ = _load_embeddings_matrix(pt_path)  # [N, D]
    index_df = _load_index(dataset_root)  # rows: tiles, same order as X
    # num classes from config or infer from columns
    if hasattr(cfg.prep, "num_classes"):
        C = int(cfg.prep.num_classes)
    else:
        cols = [c for c in index_df.columns if c.startswith("ratio_")]
        C = max(int(c.split("_")[1]) for c in cols) + 1

    # targets + groups
    ratio_cols = [f"ratio_{i}" for i in range(C)]
    missing = [c for c in ratio_cols if c not in index_df.columns]
    if missing:
        raise ValueError(f"Missing ratio columns in index.parquet: {missing}")

    # correcting bg for possible padding
    ratio_cols_wo_bg = [f"ratio_{i}" for i in range(1, C)]
    index_df["ratio_0"] = 1.0 - index_df[ratio_cols_wo_bg].sum(axis=1)

    Y = index_df[ratio_cols].to_numpy(dtype=np.float32)  # [N, C]
    groups = index_df["roi_id"].to_numpy()

    # sanity
    if X.shape[0] != Y.shape[0]:
        raise RuntimeError(
            f"Embeddings rows ({X.shape[0]}) != index rows ({Y.shape[0]}). "
            f"Make sure dataloader used shuffle=False and you didn't reorder the index."
        )
    # normalize ratios (safety)
    Y_sum = Y.sum(axis=1, keepdims=True) + 1e-12
    Y = (Y / Y_sum).astype(np.float32)

    # ---- CV splits: stratify by KMeans on normalized ratios
    K = min(30, max(5, Y.shape[0] // 150))
    km = KMeans(n_clusters=K,
                n_init="auto",
                random_state=int(getattr(cfg.run, "seed", 1337)))
    cluster_y = km.fit_predict(Y / (Y.sum(axis=1, keepdims=True) + 1e-12))

    n_splits = int(getattr(cfg.evaluation.tile_head, "cv", 5))
    cv = StratifiedGroupKFold(n_splits=n_splits,
                              shuffle=True,
                              random_state=int(getattr(cfg.run, "seed", 1337)))

    # training hyperparams (defaults + optional overrides in cfg.evaluation.tile_head.train)
    train_cfg = HeadTrainCfg()
    if hasattr(cfg.evaluation.tile_head, "train"):
        tc = cfg.evaluation.tile_head.train
        for k in [
                "epochs", "lr", "weight_decay", "dropout", "temperature",
                "batch_size", "bg_class_weight"
        ]:
            if hasattr(tc, k):
                setattr(train_cfg, k, getattr(tc, k))

    device = torch.device(
        cfg.encoder_runtime.device if torch.cuda.is_available()
        and "cuda" in cfg.encoder_runtime.device else "cpu")

    # ---- CV loop
    oof_rows: List[pd.DataFrame] = []
    fold_js: List[float] = []
    for fold, (tr, va) in tqdm(enumerate(cv.split(X, cluster_y, groups)),
                               desc="CV folds",
                               total=n_splits):
        model, js, y_true_va, y_pred_va = _fit_one_fold(X,
                                                        Y,
                                                        tr,
                                                        va,
                                                        cfg=train_cfg,
                                                        device=device)
        fold_js.append(js)

        # save head
        ckpt = {
            "state_dict": model.state_dict(),
            "in_dim": int(X.shape[1]),
            "n_classes": int(Y.shape[1]),
            "dropout": float(train_cfg.dropout),
            "temperature": float(train_cfg.temperature),
            "encoder": str(encoder_name),
            "seed": int(getattr(cfg.run, "seed", 1337)),
        }
        torch.save(ckpt, out_heads / f"fold_{fold}.pt")

        # collect OOF
        df_va = pd.DataFrame({
            "tile_idx":
            index_df.iloc[va]["tile_idx"].to_numpy()
            if "tile_idx" in index_df.columns else va,
            "roi_id":
            index_df.iloc[va]["roi_id"].to_numpy(),
        })
        true_df = pd.DataFrame(
            y_true_va, columns=[f"ratio_true_{i}" for i in range(Y.shape[1])])
        pred_df = pd.DataFrame(y_pred_va,
                               columns=[f"p_{i}" for i in range(Y.shape[1])])
        oof_rows.append(
            pd.concat([df_va.reset_index(drop=True), true_df, pred_df],
                      axis=1))

    oof = pd.concat(oof_rows, axis=0)
    oof.to_parquet("oof_preds.parquet", index=False)

    # ---- Tile-level metrics (macro)
    y_true_all = oof[[f"ratio_true_{i}" for i in range(C)]].to_numpy()
    y_pred_all = oof[[f"p_{i}" for i in range(C)]].to_numpy()
    per_class = _per_class_metrics(y_true_all, y_pred_all)
    summary = _summarize_metrics(per_class)
    summary["JS_macro"] = float(np.mean(fold_js))

    # ---- ROI-level metrics (reduce by mean prob per ROI; compare to roi_ratios.csv)
    df_roi_gt = _load_roi_gt(dataset_root, C)
    if df_roi_gt is not None:
        pred_roi = oof.groupby("roi_id")[[f"p_{i}" for i in range(C)
                                          ]].mean().reset_index()
        # align
        m = pd.merge(df_roi_gt, pred_roi, on="roi_id", how="inner")
        if len(m):
            y_true_roi = m[[f"ratio_{i}" for i in range(C)]].to_numpy()
            y_pred_roi = m[[f"p_{i}" for i in range(C)]].to_numpy()
            roi_per_class = _per_class_metrics(y_true_roi, y_pred_roi)
            roi_summary = _summarize_metrics(roi_per_class)
        else:
            roi_summary = {}
    else:
        roi_summary = {}

    # ---- Write metrics JSON
    out_metrics = {
        "tile_level": {
            "per_class": {
                k: v.tolist()
                for k, v in per_class.items()
            },
            "summary": summary,
            "fold_js": [float(x) for x in fold_js],
        },
        "roi_level": {
            "summary": roi_summary
        },
        "config": {
            "encoder": encoder_name,
            "cv": n_splits,
            "train": asdict(train_cfg),
        }
    }
    Path("cv_metrics.json").write_text(json.dumps(out_metrics, indent=2))

    # ---- Optional: full-fit head on all tiles
    if save_full_fit:
        full_model, _, _, _ = _fit_one_fold(
            X,
            Y,
            tr_idx=np.arange(X.shape[0]),
            va_idx=np.arange(
                X.shape[0]),  # just to run the loop; best checkpoint kept
            cfg=train_cfg,
            device=device)
        ckpt_full = {
            "state_dict": full_model.state_dict(),
            "in_dim": int(X.shape[1]),
            "n_classes": int(Y.shape[1]),
            "dropout": float(train_cfg.dropout),
            "temperature": float(train_cfg.temperature),
            "encoder": str(encoder_name),
            "seed": int(getattr(cfg.run, "seed", 1337)),
        }
        torch.save(ckpt_full, out_heads / "full_fit.pt")

    return {
        "heads_dir": str(out_heads.resolve()),
        "oof_preds": str((Path("oof_preds.parquet")).resolve()),
        "metrics_json": str((Path("cv_metrics.json")).resolve()),
    }
