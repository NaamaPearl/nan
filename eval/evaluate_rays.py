# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import imageio

from eval.slideshow import slider_show, slider_show_rgb_ray
from nan.dataloaders.basic_dataset import process_fn, process_fn_np
from nan.utils.io_utils import tuple_str
from visualizing.plotting import *
plt.rcParams["axes.prop_cycle"]   = mpl.cycler(color=["darkorchid", "darkorange", "lightskyblue", "dodgerblue", "mediumblue"])

from utils import to_uint
from eval.init_eval import init_eval

from nan.render_ray import RayRender
from nan.raw2output import RaysOutput
from nan.sample_ray import RaySampler



# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def extract_per_pixel(output: RaysOutput, pixel):
    T_alpha = output.debug[pixel]['w'].numpy()
    z = output.debug[pixel]['z'].cpu().numpy()
    feat = output.debug[pixel]['feat'].cpu().numpy()
    w_rg = output.debug[pixel]['w_rgb'].unsqueeze(-2).cpu().numpy()
    rgb_in = output.debug[pixel]['feat'][..., :3].unsqueeze(-2).cpu().numpy()
    return T_alpha, z, feat, w_rg, rgb_in


def create_w_hist(w, z, depth_range, h):
    bins = np.linspace(depth_range[0], depth_range[1], num=h)
    tau = (bins[1] - bins[0]) / 2
    k = 1 / (1 + ((z[..., None] - bins) / tau) ** 2)
    H = (k * w[..., None]).sum(-2)
    H = H / (H.sum() + TINY_NUMBER)
    return H, bins


def plot_T_alpha(z_coarse, T_alpha_coarse, z_fine, T_alpha_fine, save_pixel, filename=None, show=True):
    fig = plt.figure(figsize=(4, 2))

    # hist, bins = create_w_hist(T_alpha_fine, z_fine,
    #                            (z_fine.min(), z_fine.max()),
    #                            int(z_fine.shape[0] * 2))
    # plt.plot(bins, hist, label='fine')
    plt.plot(z_fine, T_alpha_fine, label='fine', marker='D', markersize=3)

    if z_coarse is not None:
        plt.plot(z_coarse, T_alpha_coarse, marker='D', markersize=3, label='coarse')
        plt.xlim(left=min(z_fine.min(), z_coarse.min()), right=max(z_fine.max(), z_coarse.max()))
        plt.ylim(top=0.7, bottom=-1e-2)
        # plt.yscale('log')
    else:
        plt.xlim(left=z_fine.min(), right=z_fine.max())
        plt.ylim(top=0.07, bottom=-1e-2)
        # plt.yscale('log')

    fig.axes[0].xaxis.set_major_locator(plt.MaxNLocator(3))
    fig.axes[0].yaxis.set_major_locator(plt.MaxNLocator(2))

    plt.subplots_adjust(top=0.95,
                        bottom=0.155,
                        left=0.165,
                        right=0.98,
                        hspace=0.2,
                        wspace=0.2)

    if filename is not None:
        plt.savefig(str(filename), dpi=200)

    if show:
        plt.show()


def plot_T_alpha_samples(z, T_alpha, save_pixel, w_rgb, rgb_in, filename=None, show=True):
    fig, axes = plt.subplots(3, figsize=(4, 3.5))

    axes[0].plot(z)
    axes[0].set_ylabel(r'$z(i)$ values')
    axes[0].set_xlim((0, len(z)))

    axes[1].plot(T_alpha)
    axes[1].set_ylabel(r'$T(z(i))\cdot\alpha(z(i))$')
    axes[1].set_xlim((0, len(z)))

    rgb_on_ray = process_fn_np((w_rgb * rgb_in).sum((1,2,3)).repeat(20, 1).transpose((1, 0, 2)))
    axes[2].imshow(rgb_on_ray)
    axes[2].set_xlabel(r'#$i$ samples')
    axes[2].set_ylabel(f"RGB\nalong ray\n")
    axes[2].set_yticks([])

    fig.suptitle(f"Fine samples along ray of pixel {save_pixel}")
    plt.subplots_adjust(top=0.895,
                        bottom=0.105,
                        left=0.17,
                        right=0.98,
                        hspace=0.39,
                        wspace=0.015)

    if filename is not None:
        plt.savefig(str(filename), dpi=1200)
    if show:
        plt.show()


def expander_square(k):
    full = np.arange(-(k // 2), k // 2 + 1)
    exp_x, exp_y = np.meshgrid(full, full)
    return exp_x.ravel(), exp_y.ravel()


def expander_empty_square(k, w=2):
    small = -(k // 2)
    large = k // 2 + 1
    full = np.arange(small - w + 1, large + w)
    empty = tuple((small - i for i in range(w))) + tuple((large + i for i in range(w)))
    exp_x0, exp_y0 = np.meshgrid(empty, full)
    exp_x1, exp_y1 = np.meshgrid(full, empty)

    return np.concatenate((exp_x0.ravel(), exp_x1.ravel())), np.concatenate((exp_y0.ravel(), exp_y1.ravel()))


def get_pixels_around(exp_x, exp_y, save_pixel_list):
    exp_x = exp_x.ravel()
    exp_y = exp_y.ravel()
    pixel_around = [(pic[0] + dy, pic[1] + dx) for pic in save_pixel_list for dx, dy in zip(exp_x, exp_y)]
    return pixel_around


def analyze_per_pixel(ret, data, save_pixel_list, res_dir: Path, show=True):
    rays_exp_dir = res_dir / 'exps'
    rays_exp_dir.mkdir(exist_ok=True)
    # for f in rays_exp_dir.glob("*"):
    #     f.unlink()

    gt_rgb = process_fn(data['rgb_clean'][0])
    noisy_rgb = process_fn(data['rgb'][0])

    gt_rgb_np_uint8 = to_uint(noisy_rgb.numpy())
    gt_rgb_np_uint8[tuple(zip(*get_pixels_around(*expander_empty_square(9, 3), save_pixel_list)))] = (220, 10, 49)
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(gt_rgb.numpy())
    plt.show()
    """
    for pixel in save_pixel_list:
        y, x = pixel
        c = 9 # 30
        imageio.imwrite(str(rays_exp_dir / f"{tuple_str(pixel)}_gt_noisy.png"), gt_rgb_np_uint8[y - c:y + c, x - c:x + c])
    imageio.imwrite(str(rays_exp_dir / f"save_pixels_marked.png"), gt_rgb_np_uint8)

    # show ground truth image
    # plt.figure()
    # plt.imshow(gt_rgb_np_uint8)
    # plt.show()

    for save_pixel in save_pixel_list:
        # debug data
        T_alpha_coarse, z_coarse, feat_coarse, *_ = extract_per_pixel(ret['coarse'], save_pixel)
        T_alpha_fine, z_fine, feat_fine, w_rgb_fine, rgb_in_fine = extract_per_pixel(ret['fine'], save_pixel)

        # plot T(z)*a(z) vs z
        plot_T_alpha(z_coarse,
                     T_alpha_coarse,
                     z_fine,
                     T_alpha_fine,
                     save_pixel=save_pixel,
                     filename=rays_exp_dir / f"log_w_z_{tuple_str(save_pixel)}.png",
                     show=show)

        # plot T(z)*a(z) vs z
        plot_T_alpha(None,
                     T_alpha_coarse,
                     z_fine,
                     T_alpha_fine,
                     save_pixel=save_pixel,
                     filename=rays_exp_dir / f"log_w_z_fine_{tuple_str(save_pixel)}.png",
                     show=show)

        # show coarse features slider
        var_coarse = feat_coarse[..., 35:70]
        # feat_coarse_fig, feat_coarse_slider = slider_show(var_coarse)

        # plot z(i), T(z(i))*a(z(i)) vs samples num
        plot_T_alpha_samples(z_fine, T_alpha_fine, save_pixel=save_pixel, show=show, rgb_in=rgb_in_fine,
                             w_rgb=w_rgb_fine,
                             filename=rays_exp_dir / f"w_samples_{tuple_str(save_pixel)}.png")

        main_rgb = process_fn_np((w_rgb_fine * rgb_in_fine).sum((1, 2, 3))[T_alpha_fine > T_alpha_fine.mean()].squeeze())
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(main_rgb[..., 0].clip(0, 1),
                   main_rgb[..., 1].clip(0, 1),
                   main_rgb[..., 2].clip(0, 1), s=300,
                   marker='^',
                   color=main_rgb.clip(0, 1))

        ax.set_xlabel('R', fontsize=28, labelpad=0)
        ax.set_ylabel('G', fontsize=28, labelpad=0)
        ax.set_zlabel('B', fontsize=28, labelpad=-4)

        ax.xaxis.set_tick_params(pad=-2, labelsize=10)
        ax.yaxis.set_tick_params(pad=-2, labelsize=10)
        ax.zaxis.set_tick_params(pad=-2, labelsize=10)

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])

        ax.set_xticks([0, 0.4, 0.8])
        ax.set_yticks([0, 0.4, 0.8])
        ax.set_zticks([0, 0.4, 0.8])

        plt.subplots_adjust(top=1.0,
                            bottom=0.06,
                            left=0.0,
                            right=1,
                            hspace=0.0,
                            wspace=0.0)
        plt.savefig(rays_exp_dir/f"scatter_{tuple_str(save_pixel)}.png")

        # show sliders of features, rgb in, rgb weights for specific pixel
        # feat_fig, feat_slider     = slider_show(feat_fine - feat_fine.mean(2, keepdims=True))
        # feat_fig, feat_slider     = slider_show(feat_fine.std(2, keepdims=True))
        # feat_fig, feat_slider     = slider_show(feat_fine)
        # var = feat_fine[..., 35:70]
        # var_fig, var_slider     = slider_show(var)
        rgb_in_fig, rgb_in_slider = slider_show_rgb_ray(w_rgb_fine[:,0,0], process_fn_np(rgb_in_fine[:,0,0]), show=False)
        max_coarse_idx = T_alpha_coarse.argmax()
        z_coarse_max = z_coarse[max_coarse_idx]
        z_fine_idx = ((z_fine - z_coarse_max) ** 2).argmin()
        rgb_in_slider.set_val(z_fine_idx)
        plt.savefig(rays_exp_dir/f"rgb_{tuple_str(save_pixel)}_{z_fine_idx}.png")

        if w_rgb_fine.shape[1] > 1:
            kernels = w_rgb_fine[z_fine_idx, :, :, :, 0, :]
            kernel_by_channel = kernels.sum(2)
            # kernel_by_channel = kernel_by_channel / kernel_by_channel.max(axis=(0, 1), keepdims=True)
            # kernel_by_channel = kernel_by_channel / kernel_by_channel.sum(axis=(0, 1), keepdims=True)

            plt.figure(figsize=(3,3))
            plt.imshow(kernel_by_channel, vmin=0, vmax=1)
            plt.yticks([])
            plt.xticks([])
            plt.subplots_adjust(top=0.995,
                                bottom=0.01,
                                left=0.005,
                                right=0.99,
                                hspace=0.0,
                                wspace=0.0)
            plt.savefig(rays_exp_dir/f"kernel_nor_by_sum_{tuple_str(save_pixel)}_{z_fine_idx}.png")
        plt.close('all')

    print('')


def evaluate_rays(add_args, differ_args):
    test_loader, scene, res_dir, eval_args, model = init_eval(add_args, open_dir=False, differ_args=differ_args)
    if scene == 'fern':
        save_pixel = ((271, 926), (350, 69), (558, 339))
    elif scene == 'orchids':
        save_pixel = ((392, 353), (300, 700))
    elif scene == 'trex':
        save_pixel = ((387, 411), (300, 700))
    elif scene == 'horns':
        save_pixel = ((392, 353), (362, 187))
    elif scene == 'flower':
        save_pixel = ((392, 353), (362, 187))
    elif scene == 'fortress':
        save_pixel = ((392, 353), (362, 187))
    elif scene == 'leaves':
        # save_pixel = ((392, 353), (362, 187))
        save_pixel = ((35, 35),)
    elif scene == 'room':
        save_pixel = ((392, 353), (362, 187))
    else:
        return

    # if scene == 'fern':
    #     save_pixel = ((274, 440), (254, 945), (289, 158), (350, 69), (558, 339), (446, 527), (271, 926), (499, 279))
    # elif scene == 'flower':
    #     save_pixel = ((185, 478), (498, 194), (454, 567))
    # else:
    #     save_pixel = ((392, 353),(387, 411), (362, 187))
    device = torch.device(f'cuda:{eval_args.local_rank}')
    ray_render = RayRender(model=model, args=eval_args, device=device, save_pixel=save_pixel)
    loader_iter = iter(test_loader)
    # _ = next(loader_iter)
    data = next(loader_iter)
    model.switch_to_eval()
    with torch.no_grad():
        ray_sampler = RaySampler(data, device=device)
        src_rgbs, featmaps, src_rgbs_clean, featmaps_clean = ray_render.calc_featmaps(src_rgbs=ray_sampler.src_rgbs,
                                                                                      src_rgbs_clean=ray_sampler.src_rgbs_clean)
        ray_batch = ray_sampler.sample_ray_batch_from_pixel(save_pixel)
        ret = ray_render.render_batch(ray_batch, src_rgbs, featmaps, src_rgbs_clean, featmaps_clean,
                                      ray_sampler.src_rgbs.to(device),
                                      ray_sampler.sigma_estimate.to(device))

        analyze_per_pixel(ret, data, save_pixel, res_dir, show=False)



