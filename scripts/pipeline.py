# full pipeline: train MLPs -> collect activations -> train SAEs -> generate interp data
# run in tmux: python -m scripts.pipeline

import subprocess
import sys


def run(module):
    print(f"\n{'='*60}")
    print(f"Running: python -m {module}")
    print(f"{'='*60}\n")
    subprocess.run([sys.executable, "-m", module], check=True)


# 1. MLP hypersearch (trains all configs, skips existing)
run("scripts.hypersearch_models")

# 2. collect activations from hardcoded model for SAE training
run("scripts.activation_collect")

# 3. SAE hypersearch (trains all expansion/L1 combos, skips existing)
run("scripts.hypersearch_saes")

# 4. generate all interp data (neuron profiles + SAE feature profiles)
run("scripts.interp_collect")

print(f"\n{'='*60}")
print("All done.")
print(f"{'='*60}")
