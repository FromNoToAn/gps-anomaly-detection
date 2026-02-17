# -*- coding: utf-8 -*-
"""
Обучение и оценка LSTM для ETA.
Рекомендуется использовать Lightning: python run_train_lstm.py
Функции train_epoch / evaluate — для инференса и совместимости.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.config import ETA_SCALE_SEC, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS
from src.models.dataset import ETAWindowDataset
from src.models.lstm_model import LSTMEtaModel


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    criterion = nn.MSELoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += criterion(pred, y).item() * x.size(0)
            n += x.size(0)
    return total_loss / max(n, 1)


def run_training(
    preprocessed_dir: str,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    device: torch.device = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = ETAWindowDataset(preprocessed_dir=preprocessed_dir)
    if len(full) == 0:
        raise ValueError("No samples in preprocessed data. Run preprocessing first.")
    n_val = int(len(full) * val_frac)
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    model = LSTMEtaModel(hidden_size=LSTM_HIDDEN_SIZE, num_layers=LSTM_NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Epoch {ep+1}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                  f"(MSE scaled); RMSE_sec={val_loss**0.5 * ETA_SCALE_SEC:.1f}")
    return model
