# serves the drawing + inference visualization on port 8000

import json
import re
import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from torchvision.datasets import EMNIST
from torchvision import transforms

from models.build import build_mlp, config_name as make_config_name, CONFIG_KEYS, LABELS
from models.SAE import SAE
from interp.mlp import attribute as mlp_attribute
from interp.sae import attribute as sae_attribute
from utils.train.saves import find_latest_checkpoint

with open("scripts/config.json") as f:
    cfg = json.load(f)

# default config name for attribution endpoints
DEFAULT_CONFIG = make_config_name(cfg["model"])
SAE_CFG = cfg["sae"]

app = FastAPI()

# load test dataset for sample browsing
dataset = None
# {label_idx: [dataset_indices]} for fast lookup by class
samples_by_class = {}

def init_dataset():
    global dataset, samples_by_class
    dataset = EMNIST(root="data", split="byclass", train=False, download=False, transform=transforms.ToTensor())
    for i, (_, label) in enumerate(dataset):
        if label not in samples_by_class:
            samples_by_class[label] = []
        samples_by_class[label].append(i)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# {config_name: model}
model_cache = {}


def parse_config(folder_name):
    # folder names are like "hidden_dim1192_hidden_dim2128"
    config = {}
    for key in CONFIG_KEYS:
        match = re.search(rf"{key}(\d+)", folder_name)
        assert match, f"Key '{key}' not found in '{folder_name}'"
        config[key] = int(match.group(1))
    return config


def load_model(config_name):
    if config_name in model_cache:
        return model_cache[config_name]

    config = parse_config(config_name)
    model = build_mlp(config)

    folder = f"weights/mlp/{config_name}"
    checkpoint = torch.load(find_latest_checkpoint(folder), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(DEVICE, DTYPE)

    model_cache[config_name] = model
    return model


@app.get("/api/config")
def get_config():
    return {"model": DEFAULT_CONFIG, "sae": SAE_CFG}


@app.get("/api/models")
def get_models():
    folder = "weights/mlp"
    if not os.path.exists(folder):
        return []
    return sorted([
        d for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d))
    ])


class PredictRequest(BaseModel):
    config_name: str
    # 28x28 float grid, 0.0 = background, 1.0 = stroke
    image: list[list[float]]


@app.post("/api/predict")
def predict(req: PredictRequest):
    model = load_model(req.config_name)

    # (28, 28) -> transpose to match training orientation -> (1, 1, 28, 28)
    img = torch.tensor(req.image, dtype=DTYPE).t().unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img)
        # (1, 62) -> (62,)
        probs = F.softmax(logits, dim=-1)[0]
        top5 = torch.topk(probs, 5)

    predictions = [
        {"label": LABELS[idx], "prob": round(prob.item(), 4)}
        for prob, idx in zip(top5.values, top5.indices)
    ]
    return {"predictions": predictions}


class AttributeRequest(BaseModel):
    # 28x28 float grid
    image: list[list[float]]
    k: int = 3


@app.post("/api/mlp-attribute")
def mlp_attribute_endpoint(req: AttributeRequest):
    model = load_model(DEFAULT_CONFIG)
    # (28, 28) -> transpose to match training orientation -> (1, 1, 28, 28)
    img = torch.tensor(req.image, dtype=DTYPE).t().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        result = mlp_attribute(model, img, k=req.k)
        # top-5 predictions
        logits = model(img)
        probs = F.softmax(logits, dim=-1)[0]
        top5 = torch.topk(probs, 5)
    result["label"] = LABELS[result["pred_class"]]
    result["predictions"] = [
        {"label": LABELS[idx], "prob": round(prob.item(), 4)}
        for prob, idx in zip(top5.values, top5.indices)
    ]
    return result


# {(layer, expansion): SAE} cache
sae_cache = {}
SAE_DIR = "weights/sae"
# SAE input dim per layer
SAE_HIDDEN_DIMS = {SAE_CFG["layer"]: cfg["model"]["hidden_dim1"]}


def load_sae(layer, expansion):
    cache_key = (layer, expansion)
    if cache_key in sae_cache:
        return sae_cache[cache_key]
    hidden_dim = SAE_HIDDEN_DIMS[layer]
    dict_size = hidden_dim * expansion
    path = os.path.join(SAE_DIR, f"layer{layer}_exp{expansion}_l1{SAE_CFG['l1_coeff']}.pt")
    sae = SAE(hidden_dim, dict_size)
    checkpoint = torch.load(path, map_location="cpu")
    sae.load_state_dict(checkpoint["model_state_dict"])
    sae.eval().to(DEVICE)
    for p in sae.parameters():
        p.requires_grad_(False)
    sae_cache[cache_key] = sae
    return sae


class SAEAttributeRequest(BaseModel):
    config_name: str
    # 28x28 float grid
    image: list[list[float]]
    layer: int = SAE_CFG["layer"]
    expansion: int = SAE_CFG["expansion"]
    k: int = 10


@app.post("/api/sae-attribute")
def sae_attribute_endpoint(req: SAEAttributeRequest):
    model = load_model(req.config_name)
    # (28, 28) -> transpose to match training orientation -> (1, 1, 28, 28)
    img = torch.tensor(req.image, dtype=torch.float32).t().unsqueeze(0).unsqueeze(0).to(DEVICE)

    # predictions (in bfloat16)
    with torch.no_grad():
        logits = model(img.to(DTYPE))
        probs = F.softmax(logits, dim=-1)[0]
        top5 = torch.topk(probs, 5)
    predictions = [
        {"label": LABELS[idx], "prob": round(prob.item(), 4)}
        for prob, idx in zip(top5.values, top5.indices)
    ]

    # SAE attribution (float32 for SAE compatibility)
    model.float()
    sae = load_sae(req.layer, req.expansion)
    result = sae_attribute(model, sae, img, layer=req.layer, k=req.k)
    model.to(DTYPE)

    result["predictions"] = predictions
    result["label"] = LABELS[result["pred_class"]]
    return result


@app.get("/api/sample")
def get_sample(label: str, index: int = 0):
    # label is a character like "A", "3", "z"
    assert label in LABELS, f"Unknown label: {label}"
    label_idx = LABELS.index(label)
    indices = samples_by_class.get(label_idx, [])
    assert indices, f"No samples for label: {label}"
    # wrap around if index out of range
    dataset_idx = indices[index % len(indices)]
    # (1, 28, 28) tensor -> transpose to fix EMNIST orientation -> 28x28 nested list
    img, _ = dataset[dataset_idx]
    pixels = img.squeeze(0).t().tolist()
    return {"image": pixels, "label": label, "index": index, "total": len(indices)}


# serve interp data as static files
if os.path.exists("results/mlp_interp"):
    app.mount("/interp_model", StaticFiles(directory="results/mlp_interp"), name="interp_model")
if os.path.exists("results/sae_interp"):
    app.mount("/interp_probes", StaticFiles(directory="results/sae_interp"), name="interp_probes")

# static files served last (API routes take priority)
app.mount("/", StaticFiles(directory="visual", html=True), name="static")

# init dataset at import time so it works with both `python -m run_visual` and uvicorn
if os.path.exists("data/EMNIST"):
    init_dataset()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
