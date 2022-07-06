import re

# import comet_ml
import os
import time
from functools import reduce
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia import tensor_to_image


def norm_im(im):
    return (im - im.min(axis=(0, 1))) / (im.max(axis=(0, 1)) - im.min(axis=(0, 1)))


def show_image_gray(image, title="", save=None, show=True):
    show_image(image, title=title, save=save, cmap='gray', show=show)


def create_image_figure(image, norm_by=None, title="", cmap=None, colorbar=True):
    vmin , vmax = None, None

    if norm_by is not None:
        if type(norm_by) is tuple:
            vmin, vmax = norm_by
        else:
            vmin = norm_by.min()
            vmax = norm_by.max()

    fig = plt.figure()
    plt.imshow(tensor_to_image(image), vmin=vmin, vmax=vmax, cmap=cmap)
    if colorbar:
        plt.colorbar()
    plt.title(title)

    return fig


def show_image(image, norm_by=None, title="", cmap="gray", save=None, show=True, colorbar=True):
    fig = create_image_figure(image, norm_by=norm_by, title=title, cmap=cmap, colorbar=colorbar)
    if show:
        plt.show()

    if save is not None:
        fig.savefig(save)
    if not show:
        plt.clf()


def create_empty_fig(rows, cols, figsize=None):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    [ax.set_xticks([]) for ax in axes.ravel()]
    [ax.set_yticks([]) for ax in axes.ravel()]
    [axes.ravel()[i].axis('off') for i in range(rows * cols)]
    axes = axes.reshape(rows, cols)
    return fig, axes


def edit_image_ax(fig, ax, image, norm_by=None, title="", cmap=None):
    vmin , vmax = None, None
    if norm_by is not None:
        vmin = norm_by.min()
        vmax = norm_by.max()

    im = ax.imshow(tensor_to_image(image), vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)


def show_depths(global_step,
                pred_depth_dict: Dict[str, torch.Tensor],
                gt_depth_dict: Dict[str, torch.Tensor],
                figsize=None):
    cols_num = len(pred_depth_dict)
    fig, axes = create_empty_fig(1, cols_num, figsize)
    for i, (level, depth) in enumerate(pred_depth_dict.items()):
        edit_image_ax(fig, axes[0, i], depth[0], norm_by=gt_depth_dict[level], title=f"depth_{level}")
    plt.suptitle(f"depths step {global_step}")
    fig.tight_layout()
    return fig


def show_mean_images(global_step, mean_images_warped, figsize=None):
    fig = create_image_figure(mean_images_warped, cmap='gray', title=f"mean images step {global_step}")
    fig.tight_layout()
    return fig


def print_link(path: Path, first='', second=''):
    print(first + ' file:///' + str(path).replace('\\', '/') + ' ' + second)


def tuple_str(tuple_data):
    return reduce(lambda a, b: f"{a}_{b}", tuple_data)


def get_latest_file(root: Path, suffix="*"):
    return max(root.glob(suffix), key=os.path.getmtime)


if __name__ == '__main__':
    pass


def decide_resume_training(resume_training, resume_last, specific_config_name):
    if resume_training:
        print("************* RESUME TRAINING ************")
        if resume_last:
            config_path = max(Path(CKPT_ROOT).glob("*"), key=os.path.getmtime)
            config_name = config_path.stem
            print(
                f"RESUME LAST MODIFIED CKPT: [{config_name}], last modified time [{time.ctime(os.path.getmtime(config_path))}]")
        else:
            config_name = specific_config_name
            config_path = CKPT_ROOT / specific_config_name
            print(
                f"RESUME SPECIFIED CKPT: [{config_name}], last modified time [{time.ctime(os.path.getmtime(config_path))}]")
    else:
        print("************* NEW TRAINING ************")
        config_name = None
    print("\n")
    print("\n")
    print("\n")
    return config_name


def float_str(weight):
    if weight == 0:
        return weight
    return f"{weight:.0e}" if weight < 1 else str(weight)


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def get_latest_file(root: Path, suffix="*"):
    return max(root.glob(suffix), key=os.path.getmtime)