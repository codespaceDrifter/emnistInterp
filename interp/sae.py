# SAE feature attribution: for a given input, find which SAE features contribute most
# contribution = feature_activation * (W_cls[pred_class] @ decoder_column)
# template = decoder column projected to pixel space via W1.T

import torch


def attribute(model, sae, image, layer=1, k=5):
    # image: (1, 1, 28, 28) tensor on correct device/dtype
    device = image.device
    x = image.float().flatten().unsqueeze(0)

    with torch.no_grad():
        # (1, hidden_dim) post-ReLU activations
        if layer == 1:
            acts = model.hidden[:2](x)
        else:
            acts = model.hidden(x)

        # (dict_size,) sparse feature activations
        encoded = torch.relu(sae.encoder(acts))

        # (1, 62) logits from hidden activations
        logits = model.classifier(acts)
        pred_class = logits[0].argmax().item()

        # contribution to predicted class: feat_act * (W_cls[pred_class] @ decoder_col)
        # W_cls: (62, hidden_dim), decoder.weight: (hidden_dim, dict_size)
        W_cls = model.classifier.weight.data.float()
        # (dict_size,) effective classifier weight per feature direction
        effective_cls_weights = W_cls[pred_class] @ sae.decoder.weight
        # (dict_size,) contribution of each feature to predicted class logit
        contributions = encoded[0] * effective_cls_weights

        # top-k by contribution to prediction (not raw activation)
        _, indices = contributions.topk(k)
        vals = encoded[0][indices]
        top_contribs = contributions[indices]

        # R² — how much of the activation is explained by top-k features
        # (k, hidden_dim) each feature's reconstruction contribution
        topk_recon_parts = vals.unsqueeze(1) * sae.decoder.weight[:, indices].T
        # (hidden_dim,) sum of top-k contributions
        topk_recon = topk_recon_parts.sum(dim=0)
        residual = topk_recon - acts[0]
        r_squared = 1.0 - residual.pow(2).sum() / acts[0].pow(2).sum()

        # pixel-space templates: project decoder columns through W1
        # W1: (hidden_dim, 784), W1.T: (784, hidden_dim)
        W1 = model.hidden[0].weight.data.float()
        pixel_proj = W1.T

        features = []
        for i in range(k):
            # (hidden_dim,) decoder column for this feature
            decoder_col = sae.decoder.weight[:, indices[i]]
            # (784,) -> (28, 28), transposed for EMNIST orientation
            template = (pixel_proj @ decoder_col).reshape(28, 28).T.tolist()
            features.append({
                "feature_idx": indices[i].item(),
                "activation": round(vals[i].item(), 4),
                "contribution": round(top_contribs[i].item(), 4),
                "template": template,
            })

    return {
        "features": features,
        "pred_class": pred_class,
        "logit": round(logits[0, pred_class].item(), 4),
        "r_squared": round(r_squared.item() * 100, 1),
    }
