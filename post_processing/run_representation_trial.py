#!/usr/bin/env python3
"""Quick trial run of representation_studies pipeline for pl128_s16 model."""
import os
os.environ["WANDB_MODE"] = "disabled"

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
import json
import gzip
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from models.patch_tst import PatchTST

MODEL_DIR = "pl128_s16_lr0.001_do0.2_L5_optadamw_h8_wd0p03/"
DATA_DIR = "capture24/final_data_1024_mode_v/"
MAX_SAMPLES = 1500
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42

def load_model_from_dir(model_dir, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model_dir = model_dir.rstrip("/")
    config_path = os.path.join(model_dir, "config.yaml")
    checkpoint_path = os.path.join(model_dir, "checkpoint.pt")
    state_path = os.path.join(model_dir, "patchtst_model.pth")

    if os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        state_dict = ckpt["model_state_dict"]
    elif os.path.isfile(config_path) and os.path.isfile(state_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        state_dict = torch.load(state_path, map_location=device, weights_only=True)
    else:
        raise FileNotFoundError(f"Model not found in {model_dir}.")

    config["hook_attention_maps"] = False
    model = PatchTST(config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, config, device

def extract_representations(model, X, device, batch_size=128):
    collected = []
    def hook_fn(module, input, output):
        collected.append(input[0].detach())
    handle = model.classifier.register_forward_hook(hook_fn)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            _ = model(x_batch)
    handle.remove()
    return torch.cat(collected, dim=0).cpu().numpy()

def main():
    print("Loading model from", MODEL_DIR)
    model, config, device = load_model_from_dir(MODEL_DIR)
    print("Device:", device, "| lookback:", config.get("lookback_window"))

    print("Loading data from", DATA_DIR)
    with gzip.open(f"{DATA_DIR}/X_test.npy.gz", "rb") as f:
        X_test = np.load(f)
    with gzip.open(f"{DATA_DIR}/Y_test.npy.gz", "rb") as f:
        Y_test = np.load(f)
    with open(f"{DATA_DIR}/label_to_index.json") as f:
        label_data = json.load(f)
    idx_to_label = label_data["index_to_label"]

    if MAX_SAMPLES is not None and len(X_test) > MAX_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_test), size=MAX_SAMPLES, replace=False)
        X_test = X_test[idx]
        Y_test = Y_test[idx]
    print("Samples:", len(X_test), "shape:", X_test.shape)

    print("Extracting representations...")
    features = extract_representations(model, X_test, device, batch_size=config.get("num_batches", 128))
    labels = Y_test.astype(int)
    print("Features shape:", features.shape)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=TSNE_RANDOM_STATE, n_iter=1000)
    embeddings_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_labels))))
    for i, label_id in enumerate(unique_labels):
        mask = labels == label_id
        name = idx_to_label.get(str(label_id), str(label_id))
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], c=[colors[i]], label=name, alpha=0.6, s=15, edgecolors="none")
    ax.set_title("t-SNE of PatchTST representations (Capture-24) — pl128_s16 trial")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    out_path = "representation_trial_tsne.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print("Saved plot to", out_path)
    print("Done.")

if __name__ == "__main__":
    main()
