import os
from typing import List, Dict

import imageio
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

from eval.init_eval import init_eval
from nan.dataloaders.basic_dataset import process_fn
from nan.dataloaders.data_utils import to_uint
from nan.raw2output import RaysOutput
from nan.render_image import render_single_image
from nan.sample_ray import RaySampler
from nan.utils.eval_utils import SSIM, img2psnr
from nan.utils.geometry_utils import warp_KRt_wrapper


class Data(List):
    def update(self, new, i):
        self.append(new)

    def mean(self):
        return sum(self) / len(self)


class Summary:
    def __init__(self):
        self.psnr = Data()
        self.ssim = Data()
        self.lpips = Data()
        self.depth_mse = Data()
        self.process_time = Data()
        self.file_id = Data()

    def update(self, psnr_val=0, lpips_val=0, ssim_val=0, depth_mse=0):
        self.psnr.append(psnr_val)
        self.lpips.append(lpips_val)
        self.ssim.append(ssim_val)
        self.depth_mse.append(depth_mse)

    def single_dict(self, i, level):
        return {f'{level}_psnr' : self.psnr[i],
                f'{level}_ssim' : self.ssim[i],
                f'{level}_lpips': self.lpips[i],
                f'{level}_depth': self.depth_mse[i],
                f'processed_time': self.process_time[i]}

    def results_dict(self, level):
        return {self.file_id[i]: self.single_dict(i, level) for i in range(len(self.psnr))}

    def mean_dict(self, total_num, level):
        result_dict = {f'{level}_mean_psnr' : self.psnr.mean(),
                       f'{level}_mean_ssim' : self.ssim.mean(),
                       f'{level}_mean_lpips': self.lpips.mean(),
                       f'{level}_mean_depth': self.depth_mse.mean()}

        if level == 'fine':
            result_dict['time'] = self.process_time.mean()

        return result_dict

    def __str__(self):
        return f"psnr: {self.psnr.running_mean:03f}, ssim: {self.ssim.running_mean:03f} \n"


class SceneEvaluator:
    def __init__(self, additional_eval_args, differ_args, rerun=True, post=''):
        self.rerun = rerun
        self.post = post
        self.summary_obj_dict: Dict[str, Summary] = {'coarse': Summary(), 'fine': Summary()}

        self.test_loader, self.scene_name, self.res_dir, self.eval_args, self.model = init_eval(additional_eval_args,
                                                                                                open_dir=False,
                                                                                                differ_from_train_args=differ_args)
        self.model.switch_to_eval()
        self.device = torch.device(f'cuda:{self.eval_args.local_rank}')
        self.ours = self.check_if_ours()

    @property
    def coarse_sum(self):
        return self.summary_obj_dict['coarse']

    @property
    def fine_sum(self):
        return self.summary_obj_dict['fine']

    def __str__(self):
        return f"running mean coarse: {self.coarse_sum}\nrunning mean fine:{self.fine_sum}"

    @classmethod
    def scene_evaluation(cls, add_args, differ_args, rerun=True, post=''):
        evaluator = cls(add_args, differ_args, rerun, post)
        return evaluator.start_scene_evaluation()

    def start_scene_evaluation(self):
        for i, data in enumerate(self.test_loader):
            self.evaluate_single_burst(data)

        total_num = len(self.test_loader)

        sum_results_dict = {self.scene_name: {}}
        sum_results_dict[self.scene_name].update(self.coarse_sum.mean_dict(total_num, 'coarse'))
        sum_results_dict[self.scene_name].update(self.fine_sum.mean_dict(total_num, 'fine'))

        if self.eval_args.eval_dataset != 'usfm':
            filename = f'{self.post}psnr_{self.scene_name}_{self.model.start_step}'
            results_dict = self.get_all_results_dict()
            save_results(results_dict, self.res_dir.parent / f'{filename}.npy')
            print_result(self.res_dir.parent, results_dict, sum_results_dict, step=self.model.start_step, f=None)
            with open(str(self.res_dir.parent / f"{filename}.txt"), "w") as f:
                print_result(self.res_dir.parent, results_dict, sum_results_dict, step=self.model.start_step, f=f)

    def sum_burst_output(self, data, rays_output, file_id, ray_sampler):
        # process ground truth
        gt_rgb = self.process_gt(data, ray_sampler)

        # coarse
        self.sum_burst_output_per_level(ray_sampler, data, gt_rgb, rays_output, file_id, 'coarse')

        # fine
        if rays_output['fine'] is not None:
            self.sum_burst_output_per_level(ray_sampler, data, gt_rgb, rays_output, file_id, 'fine')

    def sum_burst_output_per_level(self, ray_sampler, data, gt_rgb, rays_output, file_id, level):
        pred_rgb = rays_output[level].rgb.detach().cpu()

        # process output if needed (white balance and gamma correction)
        if self.rerun and self.eval_args.process_output:
            pred_rgb = process_fn(pred_rgb, data['white_level'])
        if self.eval_args.eval_dataset == 'usfm':
            pred_rgb = pred_rgb * data['white_level']

        if gt_rgb is not None:
            psnr, ssim, lpips = calculate_metrics(pred_rgb, gt_rgb)
        else:
            psnr, ssim, lpips = 0, 0, 0

        # predicted depth
        pred_depth, depth_error = self.depth_evaluation(rays_output, data, level, file_id)

        self.summary_obj_dict[level].update(psnr_val=psnr, lpips_val=lpips, ssim_val=ssim, depth_mse=depth_error)

        if self.rerun:
            # saving outputs: predicted rgb, accuracy map, predicted depth,  warp images
            self.save_outputs(gt_rgb, pred_rgb, file_id, level, rays_output, pred_depth, ray_sampler, data)

    def check_if_ours(self):
        is_deeprep = self.res_dir.parent.parent.parent.stem == 'deeprep'
        is_bpn = self.res_dir.parent.parent.parent.stem == 'bpn'
        ours = not is_bpn and not is_deeprep
        assert not (not ours and self.rerun)
        return ours

    def evaluate_single_burst(self, data):
        rgb_path = data['rgb_path'][0]
        file_id = os.path.basename(rgb_path).split('.')[0]
        self.summary_obj_dict['fine'].file_id.append(file_id)
        self.summary_obj_dict['coarse'].file_id.append(file_id)

        with torch.no_grad():
            ray_sampler = RaySampler(data, device=self.device, render_stride=self.eval_args.render_stride)

        if self.rerun:
            self.save_input(data, file_id)

            with torch.no_grad():
                start = time.time()
                rays_output = render_single_image(ray_sampler=ray_sampler, model=self.model, args=self.eval_args)
                process_time = time.time() - start
                self.summary_obj_dict['fine'].process_time.append(process_time)
                self.summary_obj_dict['coarse'].process_time.append(process_time) # TODO Naama don't really need this
        else:
            rays_output, process_time = self.load_results(file_id)

        self.sum_burst_output(data, rays_output, file_id, ray_sampler)

    def save_input(self, data, file_id):
        src_rgbs = data['src_rgbs'][0].cpu().numpy()
        noisy_rgb = data['rgb'][0]
        averaged_img = np.mean(src_rgbs, axis=0)

        if self.eval_args.process_output:
            noisy_rgb = process_fn(noisy_rgb, data['white_level'])
            averaged_img = process_fn(averaged_img, data['white_level']).cpu().numpy()
        if self.eval_args.eval_dataset == 'usfm':
            noisy_rgb = noisy_rgb * data['white_level']
            averaged_img = averaged_img * data['white_level'].numpy()
        imageio.imwrite(str(self.res_dir / f"{file_id}_noisy.png"), to_uint(noisy_rgb.cpu().numpy()), 'PNG-FI')
        imageio.imwrite(str(self.res_dir / f"{file_id}_average.png"), to_uint(averaged_img), 'PNG-FI')

    def load_results(self, file_id):
        pred_fine = torch.tensor(imageio.imread(self.res_dir / f"{self.post}{file_id}_pred_fine.png")) / 255

        if self.ours:
            pred_depth_fine = torch.tensor(
                imageio.imread(self.res_dir / f"{file_id}_depth_fine.png").__array__() / 1000)
            pred_coarse = torch.tensor(imageio.imread(self.res_dir / f"{file_id}_pred_coarse.png")) / 255
            pred_depth_coarse = torch.tensor(
                imageio.imread(self.res_dir / f"{file_id}_depth_coarse.png").__array__() / 1000)
        else:
            pred_depth_fine = torch.zeros_like(pred_fine[..., 0])
            pred_coarse = torch.zeros_like(pred_fine)
            pred_depth_coarse = torch.zeros_like(pred_fine[..., 0])

        rays_output = {'coarse': RaysOutput(rgb_map=pred_coarse, depth_map=pred_depth_coarse),
                       'fine'  : RaysOutput(rgb_map=pred_fine, depth_map=pred_depth_fine)}
        process_time = 0

        return rays_output, process_time

    def process_gt(self, data, ray_sampler):
        if 'rgb_clean' in data:
            gt_rgb = data['rgb_clean'][0][::ray_sampler.render_stride, ::ray_sampler.render_stride]
            # always process rgb, since it came from dataloader which perform unprocess
            # same idea as "Unprocessing Images for Learned Raw Denoising"

            if self.eval_args.process_output:
                gt_rgb = process_fn(gt_rgb, data['white_level'])
            if self.eval_args.eval_dataset == 'usfm':
                gt_rgb = gt_rgb * data['white_level']
        else:
            gt_rgb = None

        return gt_rgb

    def save_outputs(self, gt_rgb, pred_rgb, file_id, level, rays_output, pred_depth, ray_sampler, data):
        # saving outputs ...
        # error map
        if gt_rgb is not None:
            err_map = (((pred_rgb - gt_rgb) ** 2).sum(-1).clamp(0, 1) ** (1 / 3)).numpy()
            err_map_colored = to_uint(cmap_jet(err_map)[..., :3])
            imageio.imwrite(str(self.res_dir / f"{file_id}_err_map_{level}.png"), err_map_colored, 'PNG-FI')

        # predicted rgb
        pred_rgb = to_uint(pred_rgb.numpy())
        imageio.imwrite(str(self.res_dir / f"{file_id}_pred_{level}.png"), pred_rgb, 'PNG-FI')

        # accuracy map
        acc_map = torch.sum(rays_output[level].weights, dim=-1).detach().cpu()
        acc_map_colored = to_uint(cmap_jet(acc_map)[..., :3])
        imageio.imwrite(str(self.res_dir / f"{file_id}_acc_map_{level}.png"), acc_map_colored, 'PNG-FI')

        # predicted depth (depth error is saved under calc_depth)
        imageio.imwrite(str(self.res_dir / f"{file_id}_depth_{level}.png"),
                        (pred_depth * 1000).astype(np.uint16), 'PNG-FI')

        depth_range = data['depth_range']
        norm_depth = plt.Normalize(vmin=depth_range.squeeze()[0], vmax=depth_range.squeeze()[1])
        pred_depth_colored = to_uint(cmap_jet(norm_depth(pred_depth))[..., :3])
        imageio.imwrite(str(self.res_dir / f"{file_id}_depth_vis_{level}.png"), pred_depth_colored, 'PNG-FI')

        # warp images
        if ray_sampler.render_stride == 1:
            warped_img_rgb = warped_images_by_depth(rays_output[level], data, self.device)
            if self.rerun and self.eval_args.process_output:
                warped_img_rgb = process_fn(warped_img_rgb, data['white_level'])
            if self.eval_args.eval_dataset == 'usfm':
                warped_img_rgb = warped_img_rgb * data['white_level']
            imageio.imwrite(str(self.res_dir / f"{file_id}_warped_images_{level}.png"), warped_img_rgb, 'PNG-FI')

            res_image = np.concatenate((pred_rgb, warped_img_rgb, pred_depth_colored), axis=1)
        else:
            res_image = np.concatenate((pred_rgb, pred_depth_colored), axis=1)

        imageio.imwrite(str(self.res_dir / f"{file_id}_sum_{level}.png"), res_image, 'PNG-FI')

    def depth_evaluation(self, rays_output, data, level, file_id):
        depth_range = data['depth_range']
        norm_depth = plt.Normalize(vmin=0, vmax=depth_range.squeeze()[1] / 20)
        pred_depth = rays_output[level].depth.detach().cpu().numpy().squeeze()
        if 'gt_depth' in data:  # gt depth in the paper is the depth of the original IBRNet on the clean images
            gt_depth = data['gt_depth'].detach().cpu().numpy().squeeze()
            if gt_depth.shape != () and gt_depth.shape == pred_depth.shape and pred_depth.sum() > 0:
                depth_error_map = ((gt_depth - pred_depth) ** 2)
                depth_error = depth_error_map.mean()
                imageio.imwrite(str(self.res_dir / f"{file_id}_depth_error_{level}.png"),
                                to_uint(norm_depth(depth_error_map)), 'PNG-FI')  # TODO Naama move to all save output?
            else:
                depth_error = 0
        else:
            depth_error = 0

        return pred_depth, depth_error

    def get_all_results_dict(self):
        coarse_dict = self.summary_obj_dict['coarse'].results_dict('coarse')
        fine_dict = self.summary_obj_dict['fine'].results_dict('fine')

        assert coarse_dict.keys() == fine_dict.keys()

        merged_dict = coarse_dict.copy()
        for key in coarse_dict.keys():
            merged_dict[key].update(fine_dict[key])
            merged_dict[key]['processed_time'] = merged_dict[key].pop('processed_time')
        return {self.scene_name: merged_dict}


def save_results(results_dict, fpath):
    assert len(results_dict) == 1
    results_dict = list(results_dict.values())[0]
    res = np.array([[v for v in im_res.values()] for (image, im_res) in results_dict.items()])
    np.save(fpath, res)


TABLE_WIDTH = 15
TABLE_COLUMN = 9


def get_float_fmt(width, column):
    return ' | '.join([f'{{:{width}}}'] + [f'{{:{width}.4f}}'] * column)


def get_string_fmt(width, column):
    return ' | '.join([f'{{:{width}}}'] + [f'{{:{width}}}'] * column)


def get_separator(fmt, width, column):
    return fmt.format(*['-' * width for _ in [width] * (column + 1 + 4)])


float_fmt = get_float_fmt(TABLE_WIDTH, TABLE_COLUMN)
string_fmt = get_string_fmt(TABLE_WIDTH, TABLE_COLUMN)
separator = get_separator(string_fmt, TABLE_WIDTH, TABLE_COLUMN)


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


def warped_images_by_depth(output, data, device):
    depth = output.depth.detach().to(device)
    Ki = data['src_cameras'][:, :, 2:18].reshape((-1, 4, 4)).unsqueeze(0)
    Rti = data['src_cameras'][:, :, 18:34].reshape((-1, 4, 4)).unsqueeze(0)
    images = data['src_rgbs'].permute((0, 1, 4, 2, 3))
    warped_images = warp_KRt_wrapper(images.to(device), Ki.to(device), Rti.inverse().to(device), 1 / depth)
    warped_images = warped_images[0].mean(0).permute((1, 2, 0))
    warped_images_rgb = to_uint(warped_images.cpu().numpy().squeeze())
    return warped_images_rgb
