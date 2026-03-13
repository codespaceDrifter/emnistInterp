from torchvision.datasets import EMNIST

DATA_DIR = "data"

train = EMNIST(root=DATA_DIR, split="byclass", train=True, download=True)
test = EMNIST(root=DATA_DIR, split="byclass", train=False, download=True)

print(f"train: {len(train)} samples")
print(f"test: {len(test)} samples")
print(f"classes: {len(train.classes)}")
print(f"image shape: {train[0][0].size}")
