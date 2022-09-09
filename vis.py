import numpy as np
from sklearn.decomposition import PCA
import torch
from run_nerf import render
import matplotlib.pyplot as plt


def load_settings():
    return {
        "flower": {
            "rc": [200, 300],
            "sz": 50,
            "thr": 1.2,
            "margin": 0.15,
            "view_a": 0,
            "view_b": -1,
        },
    }


def calc_pca(emb, vis=False):
    X = emb.flatten(0, -2).cpu().numpy()
    np.random.seed(6)
    pca = PCA(n_components=3)
    pca.fit(X)
    X_rgb = pca.transform(X).reshape(*emb.shape[:2], 3)
    if vis:
        plt.imshow(X_rgb)
        plt.axis("off")
        plt.show()
    return X_rgb


def calc_feature_dist(embq, emb):
    dist = torch.norm(embq - emb, dim=-1)
    return dist.cpu()


def calc_query_emb(emb, r, c, extent, rgb=None):
    if rgb is not None:

        rgb_cut = rgb.clone().cpu()
        rgb_cut[r : r + extent, c : c + extent] = 0
        rgb_patch = rgb[r : r + extent, c : c + extent].cpu()

        f, ax = plt.subplots(1, 2, figsize=(5, 2))
        ax[0].imshow(rgb_cut)
        ax[0].axis("off")
        ax[1].imshow(rgb_patch)
        ax[1].axis("off")
        plt.suptitle("Rendered image without patch vs. patch")
        plt.show()

    emb_patch = emb[r : r + extent, c : c + extent]
    embq = torch.nn.functional.normalize(emb_patch.flatten(0, -2).mean(dim=0), dim=0)

    return embq


def render_composed(state, img_i):
    pose = state["poses"][img_i, :3, :4].clone()
    with torch.no_grad():
        rgb, disp, acc, emb, extras = render(
            state["H"],
            state["W"],
            state["K"],
            chunk=state["args"].chunk,
            c2w=pose[:3, :4],
            **state["render_kwargs_test"],
        )
    return rgb, emb


def render_decomposed(state, img_i, embq, dist_thr, foreground=True):
    pose = state["poses"][img_i, :3, :4].clone()
    with torch.no_grad():
        rgb, disp, acc, emb, extras = render(
            state["H"],
            state["W"],
            state["K"],
            chunk=state["args"].chunk,
            c2w=pose[:3, :4],
            **state["render_kwargs_test"],
            retraw=True,
            embq=embq,
            dist_thr=dist_thr,
            dist_less=foreground,
        )
    return rgb, emb
