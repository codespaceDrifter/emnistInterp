# shared visualization for template images
# converts grayscale templates to RGB with red highlighting above threshold

import numpy as np


def templates_to_rgb(templates, threshold=0.2):
    # templates: (N, H, W) numpy array, values centered at 0
    # returns: (N, H, W, 3) float RGB array [0, 1]
    # pixels normalized to [-1, 1], above threshold -> red, rest -> grayscale
    n, h, w = templates.shape
    vmax = max(abs(templates.min()), abs(templates.max()))
    if vmax == 0:
        vmax = 1
    # (N, H, W) normalized to [-1, 1]
    normed = templates / vmax

    # grayscale base: map [-1, 1] to [0, 1]
    gray = (normed + 1) / 2
    # (N, H, W, 3) RGB
    rgb = np.stack([gray, gray, gray], axis=-1)

    # pixels above threshold: mark red
    mask = normed > threshold
    rgb[mask] = [1.0, 0.0, 0.0]

    return rgb
