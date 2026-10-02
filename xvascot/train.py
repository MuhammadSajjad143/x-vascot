"""Datasets, training loop and evaluation.

Differences from the notebook, all forced by the local environment or by the
fixed-mode corrections:

* ``torch.cuda.amp.autocast`` / ``GradScaler`` are deprecated in torch 2.5 and
  are replaced by the ``torch.amp`` equivalents.
* ``nn.LSTM`` no longer receives a ``dropout=`` argument it ignored (a single
  layer LSTM cannot apply inter-layer dropout; it only emitted a warning).
* Variable-length batching is added for fixed mode, with length bucketing so a
  batch of four 10k-token contracts does not sit next to a 63-token one.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset, Sampler


class OpcodeDataset(Dataset):
    """Fixed-length padded sequences (repro mode)."""

    def __init__(self, X, y):
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.long)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class VarLenDataset(Dataset):
    """Variable-length sequences (fixed mode); padded per batch by the collate."""

    def __init__(self, seqs, y):
        self.seqs = [np.asarray(s, dtype=np.int64) for s in seqs]
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], int(self.y[i])


def pad_collate(batch):
    """Pre-pad a batch to its own maximum length, matching Keras' padding='pre'."""
    seqs, labels = zip(*batch)
    maxlen = max(len(s) for s in seqs)
    out = np.zeros((len(seqs), maxlen), dtype=np.int64)
    for i, s in enumerate(seqs):
        if len(s):
            out[i, -len(s):] = s          # leading zeros, like padding='pre'
    return torch.from_numpy(out), torch.tensor(labels, dtype=torch.long)


class LengthBucketSampler(Sampler):
    """Group similar-length sequences into batches to bound padding waste.

    Batches themselves are shuffled each epoch, so the model still sees a
    different order every epoch; only the *composition* of a batch is by length.
    """

    def __init__(self, lengths, batch_size: int, shuffle: bool = True, seed: int = 0):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = np.argsort(self.lengths, kind="stable")
        if self.shuffle:
            # jitter within the sorted order so buckets are not identical each epoch
            noise = rng.permutation(len(order))
            order = order[np.argsort(self.lengths[order] * 1000 + noise % 1000,
                                     kind="stable")]
        batches = [order[i:i + self.batch_size]
                   for i in range(0, len(order), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        for b in batches:
            yield list(b)

    def __len__(self):
        return int(np.ceil(len(self.lengths) / self.batch_size))


def make_loader(X, y, batch_size: int, shuffle: bool, variable: bool,
                seed: int = 0) -> DataLoader:
    """Build a loader. Evaluation loaders keep dataset order.

    Length bucketing is a training-time optimisation only. An evaluation loader
    must emit samples in dataset order, otherwise the returned predictions no
    longer line up with the contracts they came from, and anything that indexes
    back into ``X`` by prediction position (the case study) silently mismatches.
    """
    if not variable:
        return DataLoader(OpcodeDataset(X, y), batch_size=batch_size, shuffle=shuffle)
    ds = VarLenDataset(X, y)
    if not shuffle:
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          collate_fn=pad_collate)
    lengths = [len(s) for s in ds.seqs]
    sampler = LengthBucketSampler(lengths, batch_size, shuffle=True, seed=seed)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=pad_collate)


# ── Training ─────────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, config, device,
                label: str = "model", log=print, amp: bool = True):
    """Train with early stopping on validation loss (patience from config)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history = defaultdict(list)

    best_val_loss, patience_count = float("inf"), 0
    patience = config.get("patience", 7)
    best_state = None

    for epoch in range(config["epochs"]):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(X_b)
                loss = criterion(out, y_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.item()
            t_correct += (out.argmax(1) == y_b).sum().item()
            t_total += y_b.size(0)

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                with torch.amp.autocast("cuda", enabled=amp):
                    out = model(X_b)
                    loss = criterion(out, y_b)
                v_loss += loss.item()
                v_correct += (out.argmax(1) == y_b).sum().item()
                v_total += y_b.size(0)

        t_l, v_l = t_loss / len(train_loader), v_loss / len(val_loader)
        history["train_loss"].append(t_l)
        history["val_loss"].append(v_l)
        history["train_acc"].append(100 * t_correct / t_total)
        history["val_acc"].append(100 * v_correct / v_total)

        if (epoch + 1) % 5 == 0:
            log(f"[{label}] epoch {epoch+1:3d}/{config['epochs']} | "
                f"train {t_l:.4f}/{100*t_correct/t_total:.1f}% | "
                f"val {v_l:.4f}/{100*v_correct/v_total:.1f}%")

        if v_l < best_val_loss:
            best_val_loss, patience_count = v_l, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= patience:
                log(f"[{label}] early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return dict(history)


@torch.no_grad()
def evaluate_model(model, loader, device, label: str = "Model", log=print,
                   verbose: bool = True):
    """Accuracy / precision / recall / F1 / AUC plus raw predictions."""
    model.eval()
    preds, labels, probs = [], [], []
    multiclass = False
    for X_b, y_b in loader:
        out = model(X_b.to(device))
        p = F.softmax(out.float(), dim=1)
        multiclass = p.shape[1] > 2
        probs.extend((p if multiclass else p[:, 1]).cpu().numpy())
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y_b.numpy())

    preds, labels, probs = np.array(preds), np.array(labels), np.array(probs)
    if multiclass:
        # Dataset II (base paper Fig. 12): macro metrics, one-vs-rest AUC, per class
        from sklearn.metrics import confusion_matrix
        res = {
            "label": label,
            "acc": accuracy_score(labels, preds),
            "prec": precision_score(labels, preds, average="macro", zero_division=0),
            "rec": recall_score(labels, preds, average="macro", zero_division=0),
            "f1": f1_score(labels, preds, average="macro", zero_division=0),
            "auc": roc_auc_score(labels, probs, multi_class="ovr", labels=list(range(probs.shape[1]))),
            "per_class_recall": recall_score(labels, preds, average=None, zero_division=0).tolist(),
            "per_class_precision": precision_score(labels, preds, average=None, zero_division=0).tolist(),
            "confusion": confusion_matrix(labels, preds, labels=list(range(probs.shape[1]))).tolist(),
            "preds": preds, "labels": labels, "probs": probs,
        }
        if verbose:
            log(f"\n{'='*58}\n  {label} - test results (4-class)\n{'='*58}")
            log(f"  accuracy : {res['acc']*100:.2f}%   macro F1: {res['f1']:.4f}   AUC(ovr): {res['auc']:.4f}")
            log(f"  confusion (rows=true safe/suicidal/prodigal/greedy):\n{np.array(res['confusion'])}")
            log(classification_report(labels, preds, zero_division=0))
        return res
    res = {
        "label": label,
        "acc": accuracy_score(labels, preds),
        "prec": precision_score(labels, preds, zero_division=0),
        "rec": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs),
        "preds": preds, "labels": labels, "probs": probs,
    }
    if verbose:
        log(f"\n{'='*58}\n  {label} - test results\n{'='*58}")
        log(f"  accuracy : {res['acc']*100:.2f}%")
        log(f"  precision: {res['prec']:.4f}  recall: {res['rec']:.4f}  F1: {res['f1']:.4f}")
        log(f"  AUC-ROC  : {res['auc']:.4f}")
        log(classification_report(labels, preds, target_names=["Non-Vuln", "Greedy"]))
    return res
