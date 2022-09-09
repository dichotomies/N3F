import os
import torch
import numpy as np


def interpolate_imlabel(imlabel, imh, imw):
    # inp: [C (optional), H, W]
    if type(imlabel) is np.ndarray:
        imlabel = torch.from_numpy(imlabel)
    if len(imlabel.shape) > 2:
        # for labeled image
        imlabel = imlabel[None].float()
    else:
        imlabel = imlabel[None, None].float()
    return torch.nn.functional.interpolate(imlabel, size=(imh, imw), mode="nearest")[0]


def initialize_optimizer(optimizer, model, model_fine):
    param_group_emb = {
        k: optimizer.param_groups[0][k]
        for k in optimizer.param_groups[0]
        if k != "params"
    }
    param_group_emb["params"] = list(model.emb_linear.parameters()) + list(
        model_fine.emb_linear.parameters()
    )
    optimizer.add_param_group(param_group_emb)

    for pg in optimizer.param_groups:
        new_lr = 1e-5


def set_requires_grad(model, keys_incl=None, keys_excl=None, requires_grad=True):
    assert (keys_incl is not None and keys_excl is not None) == False
    for p in model.named_parameters():
        assert type(p[0]) is str
        if keys_incl is None and keys_excl is None:
            p[1].requires_grad = requires_grad
        elif keys_incl is not None:
            if any([k in p[0] for k in keys_incl]):
                p[1].requires_grad = requires_grad
        elif keys_excl is not None:
            if any([k in p[0] for k in keys_excl]):
                pass
            else:
                p[1].requires_grad = requires_grad


def x2samples(x, i_train):
    """
    x: images or features

    returns [(N)*H*W], 1, C] with N: train images, C: channels

    verify:
    z = x2samples(images)
    (rays_rgb[:, -1:] == z).all()
    """

    n_channels = x.shape[-1]
    assert n_channels in [3, 64]  # either rgb or pca features
    x = x[:, None]
    x = np.transpose(x, [0, 2, 3, 1, 4])
    x = np.stack([x[i] for i in i_train], 0)  # train images only
    x = np.reshape(x, [-1, 1, n_channels])  # [(N-1)*H*W, ro+rd+rgb, 3]
    x = x.astype(np.float32)
    return x


def load_features(vid, root="data/dino/pca64", normalised=True, imhw=None):
    features = torch.load(os.path.join(root, f"{vid}.pt"))

    if normalised:
        for k in features:
            features[k] = torch.nn.functional.normalize(features[k], dim=0)

    if imhw is not None:
        for k in features:
            features[k] = interpolate_imlabel(features[k], imhw[0], imhw[1])

    return features
