# MLP hypersearch: grid search over hidden layer sizes
# saves weights to weights/mlp/ and results to results/mlp_hypersearch/

import os
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision import transforms

from models.build import build_mlp
from utils.train.hypersearch import hypersearch

BATCH_SIZE = 256
MAX_BATCHES = 30000
LR = 3e-4

# 1-layer MLPs (hidden_dim2=0) + 2-layer MLPs (hidden_dim2 from 64 to hidden_dim1)
configs = [
    {"hidden_dim1": h1, "hidden_dim2": 0}
    for h1 in [64, 128, 192, 256, 320]
] + [
    {"hidden_dim1": h1, "hidden_dim2": h2}
    for h1 in [64, 128, 192, 256, 320]
    for h2 in range(64, h1 + 1, 64)
]

assert os.path.exists("data/EMNIST"), "EMNIST not found in data/ — run `python -m utils.emnist` first"

transform = transforms.ToTensor()
train_dataset = EMNIST(root="data", split="byclass", train=True, download=False, transform=transform)
val_dataset = EMNIST(root="data", split="byclass", train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
print(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")

hypersearch(
    build_model=build_mlp,
    configs=configs,
    train_loader=train_loader,
    val_loader=val_loader,
    max_batches=MAX_BATCHES,
    lr=LR,
    weights_folder="weights/mlp",
    metadata_folder="results/mlp_hypersearch",
)
