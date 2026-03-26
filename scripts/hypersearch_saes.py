# SAE hyperparameter search over expansion factor and L1 coefficient
# trains SAE for each config, evaluates sparsity + reconstruction
# saves weights to weights/sae/ and results to results/sae_hypersearch/

import gc
import json
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import plotly.graph_objects as go

from models.SAE import SAE

with open("scripts/config.json") as f:
    cfg = json.load(f)

ACT_DIR = "weights/activations"
WEIGHTS_DIR = "weights/sae"
RESULTS_DIR = "results/sae_hypersearch"
# SAE input dim — must match the MLP config used to collect activations
HIDDEN_DIMS = {cfg["sae"]["layer"]: cfg["model"]["hidden_dim1"]}

BATCH_SIZE = 256
MAX_BATCHES = 10000
LR = 3e-4

# search grid
EXPANSIONS = list(range(2, 9))  # 2, 3, 4, 5, 6, 7, 8
L1_COEFFS = [0.5, 1.0, 1.5, 2.0, 3.0]
LAYERS = [1]

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_and_eval(layer, expansion, l1_coeff):
    hidden_dim = HIDDEN_DIMS[layer]
    dict_size = hidden_dim * expansion
    tag = f"layer{layer}_exp{expansion}_l1{l1_coeff}"

    # (N, hidden_dim) training activations
    train_acts = torch.load(os.path.join(ACT_DIR, f"layer{layer}_train.pt"))
    test_acts = torch.load(os.path.join(ACT_DIR, f"layer{layer}_test.pt"))
    loader = DataLoader(TensorDataset(train_acts), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    sae = SAE(hidden_dim, dict_size, l1_coeff=l1_coeff).cuda()
    optimizer = torch.optim.AdamW(sae.parameters(), lr=LR)

    sae.train()
    data_iter = iter(loader)
    for batch_idx in range(1, MAX_BATCHES + 1):
        try:
            (x,) = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            (x,) = next(data_iter)

        x = x.cuda()
        encoded, decoded = sae(x)
        total_loss, recon_loss, l1_loss = sae.loss(x, encoded, decoded)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        sae.normalize_decoder()

        if batch_idx % 2000 == 0:
            l0 = (encoded > 0).float().sum(dim=1).mean().item()
            print(f"    batch {batch_idx}/{MAX_BATCHES} | recon: {recon_loss.item():.4f} | L0: {l0:.1f}/{dict_size}")

    # save weights
    save_path = os.path.join(WEIGHTS_DIR, f"{tag}.pt")
    torch.save({"model_state_dict": sae.state_dict(), "config": {"input_dim": hidden_dim, "dict_size": dict_size, "l1_coeff": l1_coeff}}, save_path)

    # evaluate on test set
    sae.eval()
    with torch.no_grad():
        test_acts_gpu = test_acts.cuda()
        encoded, decoded = sae(test_acts_gpu)
        recon_mse = (decoded - test_acts_gpu).pow(2).mean().item()
        # mean number of active features per sample
        mean_active = (encoded > 0).float().sum(dim=1).mean().item()
        num_dead = (encoded.max(dim=0).values == 0).sum().item()

    del sae, optimizer, train_acts, test_acts, test_acts_gpu, encoded, decoded
    gc.collect()
    torch.cuda.empty_cache()

    return recon_mse, mean_active, num_dead, dict_size


# load existing results for resume
results_path = os.path.join(RESULTS_DIR, "results.json")
results = []
if os.path.exists(results_path):
    with open(results_path, "r") as f:
        results = json.load(f)
    print(f"Loaded {len(results)} existing results")

done_keys = {(r["config"]["layer"], r["config"]["expansion"], r["config"]["l1_coeff"]) for r in results}

configs = [(layer, exp, l1) for layer in LAYERS for exp in EXPANSIONS for l1 in L1_COEFFS]
print(f"SAE Hypersearch: {len(configs)} configs ({len(configs) - len(done_keys)} remaining)")

for i, (layer, expansion, l1_coeff) in enumerate(configs):
    if (layer, expansion, l1_coeff) in done_keys:
        print(f"\n[{i+1}/{len(configs)}] Skipping (done): layer{layer} exp{expansion} l1={l1_coeff}")
        continue

    tag = f"layer{layer}_exp{expansion}_l1{l1_coeff}"
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(configs)}] {tag}")
    print(f"{'='*60}")

    recon_mse, mean_active, num_dead, dict_size = train_and_eval(layer, expansion, l1_coeff)

    print(f"  recon_mse={recon_mse:.4f} | mean_active={mean_active:.1f}/{dict_size} | dead={num_dead}/{dict_size}")

    result = {
        "config": {"layer": layer, "expansion": expansion, "l1_coeff": l1_coeff},
        "recon_mse": round(recon_mse, 6),
        "mean_active": round(mean_active, 1),
        "num_dead": num_dead,
        "dict_size": dict_size,
    }
    results.append(result)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

# --- plot results per layer ---

for layer in LAYERS:
    layer_results = [r for r in results if r["config"]["layer"] == layer]
    if len(layer_results) < 2:
        continue

    dims = [
        dict(label="expansion", values=[r["config"]["expansion"] for r in layer_results]),
        dict(label="l1_coeff", values=[r["config"]["l1_coeff"] for r in layer_results]),
        dict(label="mean_active", values=[r["mean_active"] for r in layer_results]),
        dict(label="recon_mse", values=[r["recon_mse"] for r in layer_results]),
        dict(label="num_dead", values=[r["num_dead"] for r in layer_results]),
    ]

    active_vals = [r["mean_active"] for r in layer_results]
    fig = go.Figure()
    fig.add_trace(go.Parcoords(
        line=dict(
            color=active_vals,
            # green = few active features (sparse), red = many (dense)
            colorscale=[[0, "green"], [0.5, "yellow"], [1, "red"]],
            showscale=True,
            cmin=min(active_vals),
            cmax=max(active_vals),
        ),
        dimensions=dims,
    ))

    fig.update_layout(
        title=f"SAE Hypersearch — Layer {layer} (green = fewer active features)",
        font=dict(size=14),
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
        font_color="white",
    )

    html_path = os.path.join(RESULTS_DIR, f"layer{layer}.html")
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)
    print(f"Plot saved: {html_path}")

print("\nDone.")
