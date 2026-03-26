# collect post-ReLU activations from frozen MLP for SAE training
# saves activations + labels to weights/activations/

import json
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision import transforms

from models.build import build_mlp, config_name
from utils.train.saves import find_latest_checkpoint

with open("scripts/config.json") as f:
    cfg = json.load(f)

MLP_CONFIG = cfg["model"]
MLP_WEIGHTS = f"weights/mlp/{config_name(MLP_CONFIG)}"
SAVE_DIR = "weights/activations"
BATCH_SIZE = 512

transform = transforms.ToTensor()
train_dataset = EMNIST(root="data", split="byclass", train=True, download=False, transform=transform)
val_dataset = EMNIST(root="data", split="byclass", train=False, download=False, transform=transform)

# load frozen MLP
model = build_mlp(MLP_CONFIG)
checkpoint = torch.load(find_latest_checkpoint(MLP_WEIGHTS), map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval().cuda().bfloat16()


def collect_activations(dataset, split_name):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    all_layer1 = []
    all_labels = []
    layer1_act = None

    def hook_layer1(module, input, output):
        nonlocal layer1_act
        layer1_act = output.detach().float().cpu()

    # hidden is [Linear, ReLU] for 1-layer model
    h1 = model.hidden[1].register_forward_hook(hook_layer1)

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            # (batch, 1, 28, 28)
            images = images.cuda().bfloat16()
            model(images)
            all_layer1.append(layer1_act)
            all_labels.append(labels)

            if (i + 1) % 100 == 0:
                print(f"  {split_name}: {(i+1) * BATCH_SIZE}/{len(dataset)}")

    h1.remove()

    # (N, hidden_dim1)
    layer1 = torch.cat(all_layer1, dim=0)
    # (N,)
    labels = torch.cat(all_labels, dim=0)

    print(f"  {split_name}: layer1 {layer1.shape}, labels {labels.shape}")

    torch.save(layer1, os.path.join(SAVE_DIR, f"layer1_{split_name}.pt"))
    torch.save(labels, os.path.join(SAVE_DIR, f"labels_{split_name}.pt"))


os.makedirs(SAVE_DIR, exist_ok=True)
print("Collecting train activations...")
collect_activations(train_dataset, "train")
print("Collecting test activations...")
collect_activations(val_dataset, "test")
print("Done.")
