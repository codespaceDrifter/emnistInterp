# MLP model builder + EMNIST constants

from models.MLP import MLP

NUM_CLASSES = 62  # EMNIST byclass: 10 digits + 26 upper + 26 lower
IMG_SIZE = 28

# EMNIST byclass label mapping: index -> character
LABELS = [str(i) for i in range(10)] + [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)]

# config keys for parsing folder names back to config dicts
CONFIG_KEYS = ["hidden_dim1", "hidden_dim2"]


def config_name(config):
    # {"hidden_dim1": 128, "hidden_dim2": 0} -> "hidden_dim1128_hidden_dim20"
    return "_".join(f"{k}{v}" for k, v in config.items())


def build_mlp(config):
    # hidden_dim2=0 means 1-layer model
    if config["hidden_dim2"] == 0:
        hidden_dims = (config["hidden_dim1"],)
    else:
        hidden_dims = (config["hidden_dim1"], config["hidden_dim2"])
    return MLP(input_dim=IMG_SIZE * IMG_SIZE, num_classes=NUM_CLASSES, hidden_dims=hidden_dims)
