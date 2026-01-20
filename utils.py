import torch
import numpy as np
import matplotlib.pyplot as plt


def progressive_inference(model, embeddings, temporal_feat, step=5, device="cpu"):
    """
    Simulates early warning by gradually feeding user history.
    """
    model.eval()
    risks = []

    T = embeddings.shape[0]

    with torch.no_grad():
        for t in range(step, T + 1, step):
            seq = embeddings[:t].unsqueeze(0).to(device)
            temp = temporal_feat.unsqueeze(0).to(device)

            logit = model(seq, temp)
            prob = torch.sigmoid(logit).item()
            risks.append(prob)

    return risks


def detect_first_warning(risks, step, threshold=0.6, patience=3):
    """
    patience = how many consecutive windows must exceed threshold
    """
    count = 0

    for i, r in enumerate(risks):
        if r >= threshold:
            count += 1
            if count >= patience:
                post_index = (i + 1) * step
                return {
                    "window_index": i + 1,
                    "post_index": post_index,
                    "risk": r
                }
        else:
            count = 0

    return None



def plot_risk_curve(risks):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(risks, marker="o")
    ax.set_xlabel("Time window")
    ax.set_ylabel("Depression risk")
    ax.set_title("Early-Warning Risk Curve")
    ax.axhline(0.6, linestyle="--")
    return fig
