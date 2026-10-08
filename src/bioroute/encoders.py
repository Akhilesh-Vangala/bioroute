"""Fine-tune open-weight encoder classifiers and calibrate them with temperature scaling."""

from __future__ import annotations

import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def fine_tune(
    model_id: str,
    x_train: list[str],
    y_train: list[int],
    n_labels: int,
    epochs: int = 4,
    lr: float = 5e-5,
    batch_size: int = 16,
    max_len: int = 64,
    seed: int = 13,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = _device()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=n_labels).to(dev)
    enc = tok(x_train, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
    ds = list(zip(enc["input_ids"], enc["attention_mask"], torch.tensor(y_train)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * total)) * max(0.0, (total - s) / total))
    model.train()
    start = time.perf_counter()
    for _ in range(epochs):
        for ids, mask, y in loader:
            out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), labels=y.to(dev))
            out.loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
    train_s = time.perf_counter() - start
    model.eval()
    return model, tok, train_s


@torch.no_grad()
def logits(model, tok, texts: list[str], max_len: int = 64, batch_size: int = 64) -> np.ndarray:
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i : i + batch_size], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)
        out.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def single_query_latency_ms(model, tok, texts: list[str], n: int = 50) -> float:
    dev = next(model.parameters()).device
    times = []
    for t in texts[:n]:
        enc = tok([t], truncation=True, max_length=64, return_tensors="pt").to(dev)
        start = time.perf_counter()
        model(**enc).logits.cpu()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times[5:] or times))


def fit_temperature(val_logits: np.ndarray, y_val: list[int]) -> float:
    """Single-parameter temperature scaling (Guo et al., 2017) fit by NLL on a validation split."""
    lg = torch.tensor(val_logits)
    y = torch.tensor(y_val)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def softmax(lg: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = lg / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
