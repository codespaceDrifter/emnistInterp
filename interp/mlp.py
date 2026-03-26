# MLP attribution: for a given input, find which neurons contribute most to the predicted class
# contribution = neuron_activation * classifier_weight[pred_class, neuron]

import torch


def attribute(model, image, k=3):
    # image: (1, 1, 28, 28) tensor, already on correct device/dtype
    x = image.flatten(1)
    # (1, h1) post-ReLU hidden activations
    acts = model.hidden(x)
    # (1, 62) logits
    logits = model.classifier(acts)
    pred_class = logits[0].argmax().item()

    # (h1,) hidden activations
    act = acts[0].detach().float()
    # (h1,) classifier weights for predicted class
    cls_w = model.classifier.weight.data.float()[pred_class]
    # (h1,) contribution of each neuron to predicted class logit
    contributions = act * cls_w

    top_vals, top_idxs = contributions.topk(k)

    # (h1, 28, 28) weight templates, transposed for EMNIST orientation
    W1 = model.hidden[0].weight.data.float()
    templates = W1.reshape(-1, 28, 28).permute(0, 2, 1)

    neurons = []
    for i in range(k):
        idx = top_idxs[i].item()
        neurons.append({
            "neuron": idx,
            "contribution": round(top_vals[i].item(), 4),
            "activation": round(act[idx].item(), 4),
            "template": templates[idx].tolist(),
        })

    return {
        "pred_class": pred_class,
        "logit": round(logits[0, pred_class].item(), 4),
        "neurons": neurons,
    }
