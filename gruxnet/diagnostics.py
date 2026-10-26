"""Training-only memorization check to distinguish capacity from generalization."""
from pathlib import Path

import pandas as pd
import torch

from .data import sha256, write_json
from .model import CompactGRUXNet, time_frequency
from .train import WindowDataset, seed_everything


def overfit(cache: Path, split: Path, output: Path, steps=300):
    if steps < 1:
        raise ValueError("Diagnostic steps must be positive")
    seed_everything(7)
    frame = pd.read_csv(split, dtype={"channel_mask": str})
    # Never use validation or test windows for this diagnostic.
    selected = frame[frame.split == "train"].groupby(["dataset", "label"], group_keys=False).head(4)
    if len(selected) != 24:
        raise ValueError("Diagnostic expects at least four training windows for each dataset/class")
    dataset = WindowDataset(selected, cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
    mask = torch.stack([dataset[i][1] for i in range(len(dataset))]).to(device)
    y = torch.tensor(selected.label.tolist(), device=device)
    features, mask = time_frequency(x, mask, augment=False)
    model = CompactGRUXNet(channels=x.shape[1]).to(device)
    model.train()  # cuDNN recurrent backward requires training mode.
    # Deliberately disable dropout without switching the GRU to inference mode.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
        elif isinstance(module, (torch.nn.GRU, torch.nn.LSTM, torch.nn.MultiheadAttention)):
            module.dropout = 0.
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, mask)
        loss = torch.nn.functional.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite diagnostic loss")
        if step % 50 == 0 or step == steps:
            item = {"step": step, "loss": float(loss.detach()), "accuracy": float((logits.argmax(1) == y).float().mean())}
            history.append(item)
            print(f"Training-only memorization: {item}", flush=True)
        if step < steps:
            loss.backward()
            optimizer.step()
    report = {"purpose": "Training-only software/capacity diagnostic; no generalization claim", "dropout": False,
              "augmentation": False, "split_sha256": sha256(split), "sample_ids": selected.sample_id.tolist(),
              "training_trials": selected.trial_id.unique().tolist(), "history": history}
    write_json(output, report)
    return report
