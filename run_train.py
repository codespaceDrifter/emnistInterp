# EMNIST hypersearch for MLP, CNN, and ViT
# runs grid search over model sizes for each architecture
# saves weights to weights/{model}/ and results to hypersearch_results/{model}/

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision import transforms

from models.MLP import MLP
from models.CNN import CNN
from models.ViT import ViT
from utils.train.hypersearch import hypersearch

NUM_CLASSES = 62  # EMNIST byclass: 10 digits + 26 upper + 26 lower
IMG_SIZE = 28
IN_CHANNELS = 1
BATCH_SIZE = 256
MAX_BATCHES = 5000
LR = 3e-4


# ViT outputs (batch, num_patches, visual_dim) — need pooling + classifier head
class ViTClassifier(nn.Module):
    def __init__(self, vit, visual_dim, num_classes):
        super().__init__()
        self.vit = vit
        self.classifier = nn.Linear(visual_dim, num_classes)

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = self.vit(x)
        # (batch, num_patches, visual_dim)
        x = x.mean(dim=1)
        # (batch, visual_dim) — mean pool over patches
        x = self.classifier(x)
        # (batch, num_classes)
        return x


# --- build functions ---

def build_mlp(config):
    # hidden_dims: same width repeated num_layers times
    hidden_dims = tuple([config["hidden_dim"]] * config["num_layers"])
    return MLP(input_dim=IMG_SIZE * IMG_SIZE, num_classes=NUM_CLASSES, hidden_dims=hidden_dims)

def build_cnn(config):
    # channels: base * 2^i for each block (e.g. base=32, 3 blocks -> (32, 64, 128))
    channels = tuple([config["base_channels"] * (2 ** i) for i in range(config["num_blocks"])])
    return CNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, channels=channels)

def build_vit(config):
    vit = ViT(
        img_size=IMG_SIZE,
        patch_size=4,  # 28/4 = 7x7 grid = 49 patches
        in_channels=IN_CHANNELS,
        visual_dim=config["visual_dim"],
        num_layers=config["num_layers"],
        num_q_heads=config["num_heads"],
        num_kv_heads=config["num_heads"],
        mlp_dim=config["mlp_dim"],
    )
    return ViTClassifier(vit, config["visual_dim"], NUM_CLASSES)


# --- configs (small -> large, evenly spaced) ---

mlp_configs = [
    {"hidden_dim": 64,  "num_layers": 1},
    {"hidden_dim": 128, "num_layers": 2},
    {"hidden_dim": 256, "num_layers": 2},
    {"hidden_dim": 512, "num_layers": 2},
    {"hidden_dim": 512, "num_layers": 3},
]

cnn_configs = [
    {"base_channels": 16, "num_blocks": 2},   # (16, 32)
    {"base_channels": 32, "num_blocks": 2},   # (32, 64)
    {"base_channels": 32, "num_blocks": 3},   # (32, 64, 128)
    {"base_channels": 64, "num_blocks": 2},   # (64, 128)
    {"base_channels": 64, "num_blocks": 3},   # (64, 128, 256)
]

# visual_dim must be divisible by num_heads, and head_dim must be even (for RoPE half-split)
vit_configs = [
    {"visual_dim": 64,  "num_layers": 2, "num_heads": 4, "mlp_dim": 256},
    {"visual_dim": 64,  "num_layers": 4, "num_heads": 4, "mlp_dim": 256},
    {"visual_dim": 128, "num_layers": 4, "num_heads": 4, "mlp_dim": 512},
    {"visual_dim": 128, "num_layers": 6, "num_heads": 8, "mlp_dim": 512},
    {"visual_dim": 256, "num_layers": 6, "num_heads": 8, "mlp_dim": 1024},
]


# --- data ---

assert os.path.exists("data/EMNIST"), "EMNIST not found in data/ — run `python -m utils.emnist` first"

transform = transforms.ToTensor()  # PIL -> (1, 28, 28) float [0, 1]
train_dataset = EMNIST(root="data", split="byclass", train=True, download=False, transform=transform)
val_dataset = EMNIST(root="data", split="byclass", train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
print(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")


# --- run hypersearch sequentially for each architecture ---

for name, build_fn, configs in [
    ("MLP", build_mlp, mlp_configs),
    ("CNN", build_cnn, cnn_configs),
    ("ViT", build_vit, vit_configs),
]:
    print(f"\n{'#'*60}")
    print(f"# HYPERSEARCH: {name}")
    print(f"{'#'*60}")

    hypersearch(
        build_model=build_fn,
        configs=configs,
        train_loader=train_loader,
        val_loader=val_loader,
        max_batches=MAX_BATCHES,
        lr=LR,
        weights_folder=f"weights/{name}",
        metadata_folder=f"hypersearch_results/{name}",
    )
