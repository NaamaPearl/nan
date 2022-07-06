import os
import time

import imageio
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval.init_eval import init_eval
from nan.dataloaders.basic_dataset import process_fn
from nan.raw2output import RaysOutput
from nan.render_image import render_single_image
from nan.sample_ray import RaySampler
from nan.utils.eval_utils import SSIM, img2psnr
from nan.utils.geometry_utils import warp_KRt_wrapper
# from utils import to_uint, SSIM, img2psnr


class Data:
    def __init__(self):
        self.sum = 0
        self.running_mean = 0

    def update(self, new, i):
        self.sum += new
        self.running_mean = self.sum / (i + 1)


class Summary:
    def __init__(self):
        self.psnr = Data()
        self.ssim = Data()
        self.lpips = Data()
        self.depth_mse = Data()
        self.time = Data()

    def update(self, i, psnr=0, lpips=0, ssim=0, depth_mse=0, time=0):
        self.psnr.update(psnr, i)
        self.lpips.update(lpips, i)
        self.ssim.update(ssim, i)
        self.depth_mse.update(depth_mse, i)
        self.time.update(time, i)

    def total(self, total_num):
        return self.psnr.sum / total_num, self.ssim.sum / total_num, self.lpips.sum / total_num, \
               self.depth_mse.sum / total_num, self.time.sum / total_num

    def __str__(self):
        return f"psnr: {self.psnr.running_mean:03f}, ssim: {self.ssim.running_mean:03f} \n"


class RunSummary:
    def __init__(self):
        self.coarse_sum = Summary()
        self.fine_sum = Summary()

    def __str__(self):
        return f"running mean coarse: {self.coarse_sum}\nrunning mean fine:{self.fine_sum}"


lpips_fn = lpips.LPIPS(net='alex').double()
ssim_fn = SSIM(window_size=11)
psnr_fn = img2psnr


def calculate_metrics(pred_rgb, gt_rgb):
    pred_rgb_clamp = pred_rgb.clamp(min=0., max=1.).permute((2, 0, 1)).unsqueeze(0)
    gt_rgb_for_loss = gt_rgb.permute((2, 0, 1)).unsqueeze(0)
    ssim_val = ssim_fn(img1=pred_rgb_clamp, img2=gt_rgb_for_loss).item()
    psnr_val = psnr_fn(x=pred_rgb_clamp, y=gt_rgb_for_loss).item()
    lpips_val = lpips_fn(in0=pred_rgb_clamp, in1=gt_rgb_for_loss).item()
    return psnr_val, ssim_val, lpips_val


cmap_jet = plt.get_cmap('jet')
norm_0_1 = plt.Normalize(vmin=0, vmax=1)


def summary_output_level(ray_sampler, data, gt_rgb, rays_output, depth_range, out_scene_dir, file_id, i, summary, level,
                         rerun, process_time, eval_args):
    pred_rgb = rays_output.rgb.detach().cpu()
    if rerun and eval_args.process_output:
        pred_rgb = process_fn(pred_rgb, data['white_level'])
    if eval_args.eval_dataset == 'usfm':
        pred_rgb = pred_rgb * data['white_level']

    device = torch.device(f'cuda:{eval_args.local_rank}')

    if gt_rgb is not None:
        psnr, ssim, lpips = calculate_metrics(pred_rgb, gt_rgb)
    else:
        psnr, ssim, lpips = 0, 0, 0

    norm_depth = plt.Normalize(vmin=0, vmax=depth_range.squeeze()[1] / 20)

    # predicted depth
    pred_depth = rays_output.depth.detach().cpu().numpy().squeeze()
    if 'gt_depth' in data:
        gt_depth = data['gt_depth'].detach().cpu().numpy().squeeze()
        if gt_depth.shape != () and gt_depth.shape == pred_depth.shape and pred_depth.sum() > 0:
            depth_error_map = ((gt_depth - pred_depth) ** 2)
            depth_error = depth_error_map.mean()
            imageio.imwrite(str(out_scene_dir / f"{file_id}_depth_error_{level}.png"),
                            to_uint(norm_depth(depth_error_map)), 'PNG-FI')
        else:
            depth_error = 0
    else:
        depth_error = 0

    if gt_rgb is not None:
        summary.update(i, psnr=psnr, ssim=ssim, lpips=lpips, depth_mse=depth_error, time=process_time)

    if rerun:
        # saving outputs ...
        # error map
        if gt_rgb is not None:
            err_map = (((pred_rgb - gt_rgb) ** 2).sum(-1).clamp(0, 1) ** (1 / 3)).numpy()
            err_map_colored = to_uint(cmap_jet(err_map)[..., :3])
            imageio.imwrite(str(out_scene_dir / f"{file_id}_err_map_{level}.png"), err_map_colored, 'PNG-FI')

        # predicted rgb
        pred_rgb = to_uint(pred_rgb.numpy())
        imageio.imwrite(str(out_scene_dir / f"{file_id}_pred_{level}.png"), pred_rgb, 'PNG-FI')

        # accuracy map
        acc_map = torch.sum(rays_output.weights, dim=-1).detach().cpu()
        acc_map_colored = to_uint(cmap_jet(acc_map)[..., :3])
        imageio.imwrite(str(out_scene_dir / f"{file_id}_acc_map_{level}.png"), acc_map_colored, 'PNG-FI')

        # predicted depth

        imageio.imwrite(str(out_scene_dir / f"{file_id}_depth_{level}.png"), (pred_depth * 1000).astype(np.uint16), 'PNG-FI')

        norm_depth = plt.Normalize(vmin=depth_range.squeeze()[0], vmax=depth_range.squeeze()[1])
        pred_depth_colored = to_uint(cmap_jet(norm_depth(pred_depth))[..., :3])
        imageio.imwrite(str(out_scene_dir / f"{file_id}_depth_vis_{level}.png"), pred_depth_colored, 'PNG-FI')

        # warp images
        if ray_sampler.render_stride == 1:
            warped_img_rgb = warped_images_by_depth(ray_sampler, rays_output, data, out_scene_dir, file_id, level,
                                                    device)
            res_image = np.concatenate((pred_rgb, warped_img_rgb, pred_depth_colored), axis=1)
        else:
            res_image = np.concatenate((pred_rgb, pred_depth_colored), axis=1)

        imageio.imwrite(str(out_scene_dir / f"{file_id}_sum_{level}.png"), res_image, 'PNG-FI')

    return psnr, ssim, lpips, depth_error


def warped_images_by_depth(ray_sampler, output, data, out_scene_dir, file_id, level, device):
    depth = output.depth.detach().to(device)

    K0 = ray_sampler.intrinsics
    Rt0 = ray_sampler.c2w_mat
    Ki = data['src_cameras'][:, :, 2:18].reshape((-1, 4, 4))
    Rti = data['src_cameras'][:, :, 18:34].reshape((-1, 4, 4))

    Ks = torch.cat((K0, Ki), dim=0).unsqueeze(0)  # add batch dimension
    Rts = torch.cat((Rt0, Rti), dim=0).unsqueeze(0)  # add batch dimension
    images = torch.cat((data['rgb'].unsqueeze(0).permute((0, 1, 4, 2, 3)), data['src_rgbs'].permute((0, 1, 4, 2, 3))),
                       dim=1)

    warped_images = warp_KRt_wrapper(images.to(device), Ks.to(device), Rts.inverse().to(device), 1 / depth)
    warped_images = warped_images[0].mean(0).permute((1, 2, 0))
    warped_images_rgb = to_uint(warped_images.cpu().numpy().squeeze())
    imageio.imwrite(str(out_scene_dir / f"{file_id}_warped_images_{level}.png"), warped_images_rgb, 'PNG-FI')
    return warped_images_rgb


def summary_output(data, rays_output, out_scene_dir, file_id, i, run_summary, ray_sampler, scene_name, results_dict,
                   rerun, process_time, eval_args):
    if 'rgb_clean' in data:
        gt_rgb = data['rgb_clean'][0][::ray_sampler.render_stride, ::ray_sampler.render_stride]
        # always process rgb, since it came from dataloader which perform unprocess
        # same idea as "Unprocessing Images for Learned Raw Denoising"

        if eval_args.process_output:
            gt_rgb = process_fn(gt_rgb, data['white_level'])
        if eval_args.eval_dataset == 'usfm':
            gt_rgb = gt_rgb * data['white_level']
    else:
        gt_rgb = None

    # gt_rgb_np_uint8 = to_uint8(gt_rgb.numpy())

    # if rerun:
    # imageio.imwrite(str(out_scene_dir / f"{file_id}_gt_rgb.png"), gt_rgb_np_uint8)

    coarse_psnr, coarse_ssim, coarse_lpips, coarse_depth_error = summary_output_level(ray_sampler, data, gt_rgb,
                                                                                      rays_output['coarse'],
                                                                                      data['depth_range'],
                                                                                      out_scene_dir, file_id, i,
                                                                                      run_summary.coarse_sum, 'coarse',
                                                                                      rerun, 0, eval_args)
    if rays_output['fine'] is not None:
        fine_psnr, fine_ssim, fine_lpips, fine_depth_error = summary_output_level(ray_sampler, data, gt_rgb,
                                                                                  rays_output['fine'],
                                                                                  data['depth_range'], out_scene_dir,
                                                                                  file_id, i, run_summary.fine_sum,
                                                                                  'fine', rerun, process_time,
                                                                                  eval_args)
    else:
        fine_ssim = fine_psnr = fine_lpips = fine_depth_error = 0.

    results_dict[scene_name][file_id] = {'coarse_psnr': coarse_psnr,
                                         'fine_psnr': fine_psnr,
                                         'coarse_ssim': coarse_ssim,
                                         'fine_ssim': fine_ssim,
                                         'coarse_lpips': coarse_lpips,
                                         'fine_lpips': fine_lpips,
                                         'coarse_depth': coarse_depth_error,
                                         'fine_depth': fine_depth_error,
                                         'time': process_time
                                         }

    return


def evaluate_images(add_args, differ_args, rerun=True, post=''):
    test_loader, scene_name, res_dir, eval_args, model = init_eval(add_args, open_dir=False, differ_args=differ_args)
    device = torch.device(f'cuda:{eval_args.local_rank}')
    results_dict = {scene_name: {}}
    sum_results_dict = {scene_name: {}}

    deeprep = res_dir.parent.parent.parent.stem == 'deeprep'
    bpn = res_dir.parent.parent.parent.stem == 'bpn'
    ours = not bpn and not deeprep
    assert not (not ours and rerun)

    run_summary = RunSummary()

    for i, data in enumerate(test_loader):
        rgb_path = data['rgb_path'][0]
        file_id = os.path.basename(rgb_path).split('.')[0]
        with torch.no_grad():
            ray_sampler = RaySampler(data, device=device, render_stride=eval_args.render_stride)
        if rerun:
            src_rgbs = data['src_rgbs'][0].cpu().numpy()
            noisy_rgb = data['rgb'][0]
            averaged_img = np.mean(src_rgbs, axis=0)

            if eval_args.process_output:
                noisy_rgb = process_fn(noisy_rgb, data['white_level'])
                averaged_img = process_fn(averaged_img, data['white_level']).cpu().numpy()
            if eval_args.eval_dataset == 'usfm':
                noisy_rgb = noisy_rgb * data['white_level']
                averaged_img = averaged_img * data['white_level'].numpy()
            imageio.imwrite(str(res_dir / f"{file_id}_noisy.png"), to_uint(noisy_rgb.cpu().numpy()), 'PNG-FI')
            imageio.imwrite(str(res_dir / f"{file_id}_average.png"), to_uint(averaged_img), 'PNG-FI')

            model.switch_to_eval()
            with torch.no_grad():
                start = time.time()
                rays_output = render_single_image(ray_sampler=ray_sampler, model=model, args=eval_args)
                process_time = time.time() - start
        else:
            pred_fine = torch.tensor(imageio.imread(res_dir / f"{post}{file_id}_pred_fine.png")) / 255

            if ours:
                pred_depth_fine = torch.tensor(
                    imageio.imread(res_dir / f"{file_id}_depth_fine.png").__array__() / 1000)
                pred_coarse = torch.tensor(imageio.imread(res_dir / f"{file_id}_pred_coarse.png")) / 255
                pred_depth_coarse = torch.tensor(
                    imageio.imread(res_dir / f"{file_id}_depth_coarse.png").__array__() / 1000)
            else:
                pred_depth_fine = torch.zeros_like(pred_fine[..., 0])
                pred_coarse = torch.zeros_like(pred_fine)
                pred_depth_coarse = torch.zeros_like(pred_fine[..., 0])

            rays_output = {'coarse': RaysOutput(rgb_map=pred_coarse, depth_map=pred_depth_coarse, weights=None,
                                                mask=None, alpha=None, z_vals=None, sigma=None, rho=None, debug=None),
                           'fine': RaysOutput(rgb_map=pred_fine, depth_map=pred_depth_fine, weights=None,
                                              mask=None, alpha=None, z_vals=None, sigma=None, rho=None, debug=None)}
            process_time = 0
        summary_output(data, rays_output, res_dir, file_id, i, run_summary, ray_sampler, scene_name, results_dict,
                       rerun, process_time, eval_args)

    total_num = len(test_loader)
    mean_coarse_psnr, mean_coarse_ssim, mean_coarse_lpips, mean_coarse_depth, _ = run_summary.coarse_sum.total(total_num)
    mean_fine_psnr, mean_fine_ssim, mean_fine_lpips, mean_fine_depth, mean_time = run_summary.fine_sum.total(total_num)

    sum_results_dict[scene_name]['coarse_mean_psnr'] = mean_coarse_psnr
    sum_results_dict[scene_name]['fine_mean_psnr'] = mean_fine_psnr
    sum_results_dict[scene_name]['coarse_mean_ssim'] = mean_coarse_ssim
    sum_results_dict[scene_name]['fine_mean_ssim'] = mean_fine_ssim
    sum_results_dict[scene_name]['coarse_mean_lpips'] = mean_coarse_lpips
    sum_results_dict[scene_name]['fine_mean_lpips'] = mean_fine_lpips
    sum_results_dict[scene_name]['coarse_mean_depth'] = mean_coarse_depth
    sum_results_dict[scene_name]['fine_mean_depth'] = mean_fine_depth
    sum_results_dict[scene_name]['time'] = mean_time


table_width = 15
table_column = 9


def get_float_fmt(width, column):
    return ' | '.join([f'{{:{width}}}'] + [f'{{:{width}.4f}}'] * column)


def get_string_fmt(width, column):
    return ' | '.join([f'{{:{width}}}'] + [f'{{:{width}}}'] * column)


def get_separator(fmt, width, column):
    return fmt.format(*['-' * width for _ in [width] * (column + 1 + 4)])


float_fmt = get_float_fmt(table_width, table_column)
string_fmt = get_string_fmt(table_width, table_column)
separator = get_separator(string_fmt, table_width, table_column)


def print_only_mean(extra_out_dir, results_dict, sum_results_dict, step, f):
    for scene, scene_dict in results_dict.items():
        print(f"{extra_out_dir.parent.parent.name}", file=f)
        print(f"{scene}, {step}", file=f)

        print(separator, file=f)
        print(float_fmt.format('mean', *sum_results_dict[scene].values()), file=f)
        print(separator, file=f)


def print_result(extra_out_dir, results_dict, sum_results_dict, step, f):
    for scene, scene_dict in results_dict.items():
        print(f"{extra_out_dir.parent.parent.name}", file=f)
        print(f"{scene}, {step}", file=f)

        for i, (k, v) in enumerate(scene_dict.items()):
            if i == 0:
                # print title
                print(string_fmt.format('file_id', *v.keys()), file=f)
                print(separator, file=f)
            print(float_fmt.format(k, *v.values()), file=f)
            print(separator, file=f)

        print(separator, file=f)
        print(float_fmt.format('mean', *sum_results_dict[scene].values()), file=f)
        print(separator, file=f)


def save_results(results_dict, fpath):
    assert len(results_dict) == 1
    results_dict = list(results_dict.values())[0]
    res = np.array([[v for v in im_res.values()] for (image, im_res) in results_dict.items()])
    np.save(fpath, res)


def analyze_sigma(ret):
    sigma_array = ret['coarse'].sigma
    from eval.slideshow import slider_show
    slider_show(sigma_array.transpose(0, 2))
    import matplotlib.pyplot as plt
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure()
    idx_list = list(range(400, 410)) + list(range(298768, 298778)) + list(range(9000, 9010))
    sigma_vec = sigma_array.reshape((-1, sigma_array.shape[-1]))
    plt.plot(sigma_vec[list(range(400, 410))].T, c=colors[0])
    plt.plot(sigma_vec[list(range(298768, 298778))].T, c=colors[1])
    plt.plot(sigma_vec[list(range(9000, 9010))].T, c=colors[2])
    plt.xlabel("coarse samples")
    plt.ylabel(r"$\sigma$")
    plt.show()
    sigma_vals = sigma_array.reshape((-1, sigma_array.shape[-1]))[[405, 298772, 9005]].T
    fig, ax = plt.subplots()
    plt.plot(sigma_vals, c='k', label=r'$\sigma$')
    for i, beta in enumerate([1, 5, 10]):
        sigma_softmax = torch.softmax(beta * sigma_vals, dim=0) * sigma_vals.sum(0)
        plt.plot(sigma_softmax, c=colors[i], label=r'softmax$(\sigma)$, $\beta={}$'.format(beta))
    plt.xlabel("coarse samples")
    plt.ylabel(r"$\sigma$")
    handles, labels = ax.get_legend_handles_labels()
    labels_dict = dict(zip(labels, handles))
    labels_dict = {k: labels_dict[k] for k in labels_dict.keys()}
    plt.legend(labels_dict.values(), labels_dict.keys(), frameon=False)
    plt.show()
    fig, ax = plt.subplots()
    plt.plot(sigma_vals, c='k', label=r'$\sigma$')
    for i, power in enumerate([2, 5, 10]):
        sigma_square = sigma_vals ** power
        sigma_square = sigma_square * (sigma_vals.sum(0) / sigma_square.sum(0))
        plt.plot(sigma_square, c=colors[3 + i], label=r'$\sigma^{}$'.format(power))
    plt.xlabel("coarse samples")
    plt.ylabel(r"$\sigma$")
    handles, labels = ax.get_legend_handles_labels()
    labels_dict = dict(zip(labels, handles))
    labels_dict = {k: labels_dict[k] for k in labels_dict.keys()}
    plt.legend(labels_dict.values(), labels_dict.keys(), frameon=False)
    plt.show()
