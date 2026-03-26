# generate all interpretability data:
#   1. MLP weight grid images -> results/mlp_interp/imgs/
#   2. MLP neuron profiles (class selectivity + max-activating images) -> results/mlp_interp/
#   3. SAE feature profiles (if SAE weights exist) -> results/sae_interp/

import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision import transforms

from models.build import build_mlp, config_name, LABELS
from models.SAE import SAE
from utils.train.saves import find_latest_checkpoint

with open("scripts/config.json") as f:
    cfg = json.load(f)

MLP_CONFIG = cfg["model"]
MLP_WEIGHTS = f"weights/mlp/{config_name(MLP_CONFIG)}"

# how many top classes / images to save per neuron or feature
TOP_CLASSES = 5
TOP_IMAGES = 8

# SAE config
SAE_DIR = "weights/sae"
ACT_DIR = "weights/activations"
SAE_CFG = cfg["sae"]
HIDDEN_DIMS = {SAE_CFG["layer"]: MLP_CONFIG["hidden_dim1"]}
EXPANSIONS = [SAE_CFG["expansion"]]
LAYERS = [SAE_CFG["layer"]]


def save_weight_grid(templates, path, ncols=16, title=None, labels=None):
    # templates: (N, 28, 28) numpy array
    # labels: optional list of strings for subplot titles
    n = templates.shape[0]
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.2))
    if nrows == 1:
        axes = [axes]

    # global normalization centered at 0: negative = black, zero = gray, positive = white
    vmax = max(abs(templates.min()), abs(templates.max()))

    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        ax.axis("off")
        if i < n:
            ax.imshow(templates[i], cmap="gray", vmin=-vmax, vmax=vmax)
            ax.set_title(labels[i] if labels else str(i), fontsize=6, color="#4a4", pad=1)

    if title:
        fig.suptitle(title, color="#4a4", fontsize=14)

    fig.patch.set_facecolor("#1a1a1a")
    plt.tight_layout(pad=0.3)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()
    print(f"Saved: {path}")


# ========================
# Load MLP
# ========================

model = build_mlp(MLP_CONFIG)
checkpoint = torch.load(find_latest_checkpoint(MLP_WEIGHTS), map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

h1 = MLP_CONFIG["hidden_dim1"]

# hidden is Sequential: [Linear, ReLU]
# (h1, 784) — layer 1 linear weights
W1 = model.hidden[0].weight.data
# (62, h1) — classifier head weights
W_cls = model.classifier.weight.data

# ========================
# 1. Weight grid images
# ========================

# (h1, 784) -> (h1, 28, 28), transpose for EMNIST orientation
layer1 = W1.reshape(h1, 28, 28).permute(0, 2, 1).numpy()

# (62, h1) @ (h1, 784) -> (62, 784) -> (62, 28, 28)
output_templates = (W_cls @ W1).reshape(62, 28, 28).permute(0, 2, 1).numpy()

os.makedirs("results/mlp_interp/imgs", exist_ok=True)
save_weight_grid(layer1, "results/mlp_interp/imgs/layer1_weights.png", title="Layer 1 — raw neuron templates")
save_weight_grid(output_templates, "results/mlp_interp/imgs/output_effective.png", ncols=16,
                 title="Output — effective class templates (W_cls @ W1)", labels=LABELS)


# ========================
# 2. MLP neuron profiles
# ========================

test_dataset = EMNIST(root="data", split="byclass", train=False, download=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

# collect post-ReLU activations via hook + classifier logits
all_acts = {1: [], 2: []}
all_labels = []
hook_act = None

def hook_fn(module, input, output):
    global hook_act
    hook_act = output.detach().float().cpu()

hook = model.hidden[1].register_forward_hook(hook_fn)

with torch.no_grad():
    for images, labels in test_loader:
        # (batch, 62) raw logits
        logits = model(images).detach().float().cpu()
        all_acts[1].append(hook_act)
        all_acts[2].append(logits)
        all_labels.append(labels)

hook.remove()

# (N, h1), (N, 62)
test_acts = {l: torch.cat(all_acts[l], dim=0) for l in [1, 2]}
test_labels = torch.cat(all_labels, dim=0)
n_test = test_labels.shape[0]

# pixel-space templates per layer
templates_by_layer = {1: layer1, 2: output_templates}

for layer_idx, dim in [(1, h1), (2, 62)]:
    acts = test_acts[layer_idx]
    neurons = []

    for neuron_idx in range(dim):
        neuron_acts = acts[:, neuron_idx]

        # per-class mean activation
        class_means = []
        for cls in range(len(LABELS)):
            mask = test_labels == cls
            if mask.sum() == 0:
                class_means.append(0.0)
                continue
            class_means.append(neuron_acts[mask].mean().item())

        class_means_t = torch.tensor(class_means)
        top_cls_vals, top_cls_idxs = class_means_t.topk(TOP_CLASSES)
        top_classes = [
            {"label": LABELS[idx.item()], "mean_act": round(val.item(), 4)}
            for val, idx in zip(top_cls_vals, top_cls_idxs) if val > 0
        ]

        # top activating real images
        top_img_vals, top_img_idxs = neuron_acts.topk(TOP_IMAGES)
        top_images = [test_dataset[idx.item()][0].squeeze(0).t().tolist() for idx in top_img_idxs]

        num_active = (neuron_acts > 0).sum().item()
        neuron_data = {
            "neuron_idx": neuron_idx,
            "num_active": num_active,
            "pct_active": round(num_active / n_test * 100, 1),
            "top_classes": top_classes,
            "top_images": top_images,
            "template": templates_by_layer[layer_idx][neuron_idx].tolist(),
        }
        # classifier weight column: which class logits this neuron pushes toward
        if layer_idx == 1:
            # (62,) — W_cls column for this neuron
            cls_col = W_cls[:, neuron_idx]
            top_w_vals, top_w_idxs = cls_col.topk(TOP_CLASSES)
            neuron_data["top_classifier_weights"] = [
                {"label": LABELS[idx.item()], "weight": round(val.item(), 4)}
                for val, idx in zip(top_w_vals, top_w_idxs)
            ]
        neurons.append(neuron_data)

    json_path = f"results/mlp_interp/layer{layer_idx}_profiles.json"
    with open(json_path, "w") as f:
        json.dump({"layer": layer_idx, "num_neurons": dim, "neurons": neurons}, f)
    print(f"Saved: {json_path} ({dim} neurons)")


# ========================
# 3. SAE feature profiles (if SAE weights exist)
# ========================

if not os.path.exists(SAE_DIR) or not os.path.exists(ACT_DIR):
    print(f"\nNo SAE weights/activations found — skipping SAE feature profiles.")
    print("Run `python -m scripts.activation_collect` then `python -m scripts.hypersearch_saes` first.")
else:
    os.makedirs("results/sae_interp", exist_ok=True)

    # (784, h1) — project decoder columns to pixel space
    pixel_proj = {1: W1.T}

    for layer in LAYERS:
        # (N, hidden_dim) test activations
        act_path = os.path.join(ACT_DIR, f"layer{layer}_test.pt")
        if not os.path.exists(act_path):
            print(f"No test activations for layer {layer} — skipping")
            continue

        sae_test_acts = torch.load(act_path).cuda()
        sae_test_labels = torch.load(os.path.join(ACT_DIR, "labels_test.pt"))

        for expansion in EXPANSIONS:
            tag = f"layer{layer}_exp{expansion}"
            hidden_dim = HIDDEN_DIMS[layer]
            dict_size = hidden_dim * expansion

            sae_path = os.path.join(SAE_DIR, f"{tag}_l1{SAE_CFG['l1_coeff']}.pt")
            if not os.path.exists(sae_path):
                print(f"No SAE weights at {sae_path} — skipping")
                continue

            print(f"\n{'='*50}")
            print(f"Analyzing: {tag}")
            print(f"{'='*50}")

            sae = SAE(hidden_dim, dict_size)
            sae_ckpt = torch.load(sae_path, map_location="cpu")
            sae.load_state_dict(sae_ckpt["model_state_dict"])
            sae.eval().cuda()
            for p in sae.parameters():
                p.requires_grad_(False)

            # encode test set
            with torch.no_grad():
                # (N, dict_size)
                encoded, decoded = sae(sae_test_acts)

            # sparsity stats
            recon_mse = (decoded - sae_test_acts).pow(2).mean().item()
            num_dead = (encoded.max(dim=0).values == 0).sum().item()
            mean_active = (encoded > 0).float().sum(dim=1).mean().item()
            # (dict_size,) — how many test samples activate each feature
            num_active_per_feature = (encoded > 0).float().sum(dim=0)

            print(f"  Recon MSE: {recon_mse:.4f}")
            print(f"  Dead features: {num_dead}/{dict_size}")
            print(f"  Mean active per sample: {mean_active:.1f}/{dict_size}")

            encoded_cpu = encoded.cpu()
            n_sae_test = sae_test_acts.shape[0]

            # all features sorted by num_active (including dead)
            sorted_indices = num_active_per_feature.argsort(descending=True).cpu().tolist()

            features_data = []
            for feat_idx in sorted_indices:
                # (N,) activations for this feature
                feat_acts = encoded_cpu[:, feat_idx]

                # class profile: mean activation per class
                class_means = []
                for cls in range(len(LABELS)):
                    mask = sae_test_labels == cls
                    if mask.sum() == 0:
                        class_means.append(0.0)
                        continue
                    class_means.append(feat_acts[mask].mean().item())

                class_means_t = torch.tensor(class_means)
                top_cls_vals, top_cls_idxs = class_means_t.topk(TOP_CLASSES)
                top_classes = [
                    {"label": LABELS[idx.item()], "mean_act": round(val.item(), 4)}
                    for val, idx in zip(top_cls_vals, top_cls_idxs) if val > 0
                ]

                # top activating real images
                top_img_vals, top_img_idxs = feat_acts.topk(TOP_IMAGES)
                top_images = [
                    test_dataset[img_idx.item()][0].squeeze(0).t().tolist()
                    for img_idx in top_img_idxs
                ]

                # pixel-space template: project decoder column through MLP weights
                # decoder.weight: (hidden_dim, dict_size)
                decoder_col = sae.decoder.weight[:, feat_idx].cpu()
                # (784,) -> (28, 28), transposed for EMNIST orientation
                template = (pixel_proj[layer] @ decoder_col).reshape(28, 28).T.tolist()

                # effective classifier weights: W_cls @ decoder_col -> (62,) per-class logit contribution
                effective_cls = W_cls @ decoder_col
                top_w_vals, top_w_idxs = effective_cls.topk(TOP_CLASSES)
                top_cls_weights = [
                    {"label": LABELS[idx.item()], "weight": round(val.item(), 4)}
                    for val, idx in zip(top_w_vals, top_w_idxs)
                ]

                features_data.append({
                    "feature_idx": feat_idx,
                    "num_active": int(num_active_per_feature[feat_idx].item()),
                    "pct_active": round(num_active_per_feature[feat_idx].item() / n_sae_test * 100, 1),
                    "top_classes": top_classes,
                    "top_images": top_images,
                    "template": template,
                    "top_classifier_weights": top_cls_weights,
                })

            json_path = os.path.join("results/sae_interp", f"{tag}_profiles.json")
            with open(json_path, "w") as f:
                json.dump({
                    "tag": tag,
                    "recon_mse": round(recon_mse, 6),
                    "mean_active": round(mean_active, 1),
                    "dict_size": dict_size,
                    "num_dead": num_dead,
                    "features": features_data,
                }, f)
            print(f"  Saved: {json_path} ({len(features_data)} features)")

            del sae, encoded, decoded, encoded_cpu
            torch.cuda.empty_cache()

        del sae_test_acts
        torch.cuda.empty_cache()

print("\nDone.")
