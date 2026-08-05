"""Train the PointNet model on the augmented hand-landmark point-cloud dataset."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

from point_net import PointNet
from utils import char2int, load_config

device = "cuda" if torch.cuda.is_available() else "cpu"


class PointsDataSet(Dataset):
    def __init__(self, path, items):
        self.path = path
        self.items = items

    def __getitem__(self, item):
        points = np.load(os.path.join(self.path, self.items[item][0], self.items[item][1]))
        return np.array(points), char2int[self.items[item][0]]

    def __len__(self):
        return len(self.items)

    def collate_fn(self, batch):
        points, classes = list(zip(*batch))
        points = torch.cat([torch.Tensor(p) for p in points]).float().to(device)
        points = torch.reshape(points, (len(batch), 21, 3))
        classes = torch.cat([torch.Tensor([c]) for c in classes]).long().to(device)
        return points, classes


def split_files(path, divs):
    if divs[0] + divs[1] + divs[2] != 1.0:
        print(f"Wrong divisions: Train={divs[0]} Validation={divs[1]} Test={divs[2]} Total={sum(divs)}")
        sys.exit(1)

    items = []
    for root, _dirs, files in os.walk(path, topdown=True):
        for file in files:
            items.append((root.split(os.sep)[-1], file))

    np.random.shuffle(items)
    size = len(items)
    train = int(size * divs[0])
    val = int(size * divs[1])
    test = int(size * divs[2])
    train = train + (size - (train + val + test))

    return items[:train], items[train:train + val], items[train + val:]


def pointnet_loss(preds, targets):
    ce_loss = nn.CrossEntropyLoss()(preds, targets)
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    return ce_loss, acc


def train_batch(model, batch, optimizer, loss_fn):
    model.train()
    points, classes = batch
    preds = model(points)
    optimizer.zero_grad()
    loss, acc = loss_fn(preds, classes)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()


@torch.no_grad()
def validate_batch(model, batch, loss_fn):
    model.eval()
    points, classes = batch
    preds = model(points)
    loss, acc = loss_fn(preds, classes)
    return loss.item(), acc.item()


def main():
    config = load_config("config.yaml")
    model_path = config["model"]["model_path"]
    os.makedirs(model_path, exist_ok=True)
    results_path = config["paths"]["results_path"]
    os.makedirs(results_path, exist_ok=True)
    path = config["dataset"]["npy_dataset"]
    model_name = config["model"]["name"]

    status_path = os.path.join(results_path, "status.csv")
    with open(status_path, "w") as f:
        f.write("Loss train; Loss val; Acc train; Acc val\n")

    divs = (
        config["dataset"]["train_percent"],
        config["dataset"]["validation_percent"],
        config["dataset"]["test_percent"],
    )
    train_files, val_files, _test_files = split_files(path, divs)

    n_epochs = config["trainer"]["num_epochs"]
    batch_size = config["trainer"]["batch_size"]

    train_ds = PointsDataSet(path, train_files)
    val_ds = PointsDataSet(path, val_files)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_ds.collate_fn, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=val_ds.collate_fn, drop_last=True)

    model = PointNet(len(char2int)).to(device)
    optimizer_class = getattr(optim, config["optimizer"]["name"])
    optimizer = optimizer_class(model.parameters(), lr=config["optimizer"]["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=0, threshold=0.001, min_lr=1e-5, threshold_mode="abs",
    )

    loss_train_all, acc_train_all, loss_val_all, acc_val_all = [], [], [], []

    for epoch in range(n_epochs):
        print(f"Current epoch: {epoch}")

        loss_ep, acc_ep = [], []
        for batch in train_dl:
            loss, acc = train_batch(model, batch, optimizer, pointnet_loss)
            loss_ep.append(loss)
            acc_ep.append(acc)
        loss_train_all.append(np.mean(loss_ep))
        acc_train_all.append(np.mean(acc_ep))

        loss_ep, acc_ep = [], []
        for batch in val_dl:
            loss, acc = validate_batch(model, batch, pointnet_loss)
            loss_ep.append(loss)
            acc_ep.append(acc)
        loss_val_all.append(np.mean(loss_ep))
        acc_val_all.append(np.mean(acc_ep))
        scheduler.step(loss_val_all[-1])

        print(f"Loss train {loss_train_all[-1]} Loss val {loss_val_all[-1]} Acc train {acc_train_all[-1]} Acc val {acc_val_all[-1]}")
        with open(status_path, "a") as f:
            f.write(f"{loss_train_all[-1]}; {loss_val_all[-1]}; {acc_train_all[-1]}; {acc_val_all[-1]}\n")

        if (epoch + 1) % 10 == 0:
            torch.save(model, os.path.join(model_path, model_name))

    torch.save(model, os.path.join(model_path, model_name))

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    axs[0].plot(loss_train_all, color="green", label="Loss Train")
    axs[0].plot(loss_val_all, color="red", label="Loss val")
    axs[0].set_title("Loss")
    axs[0].legend()

    axs[1].plot(acc_train_all, color="green", label="Acc Train")
    axs[1].plot(acc_val_all, color="red", label="Acc Val")
    axs[1].set_title("Accuracy")
    axs[1].legend()

    plt.savefig(os.path.join(results_path, f"metrics_pointnet_{n_epochs}epochs.png"))
    plt.show()


if __name__ == "__main__":
    main()
