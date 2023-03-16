import os
from typing import List, Dict

import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

from eval.evaluate_rays import analyze_per_pixel
from eval.init_eval import init_eval
from nan.dataloaders.basic_dataset import de_linearize
from nan.dataloaders.data_utils import to_uint, imwrite, imread
from nan.raw2output import RaysOutput
from nan.render_image import render_single_image
from nan.render_ray import RayRender
from nan.sample_ray import RaySampler
from nan.utils.eval_utils import SSIM, img2psnr
from nan.utils.geometry_utils import warp_KRt_wrapper

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


class Data(List):
    def update(self, new):
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
        return {f'{level}_psnr': self.psnr[i],
                f'{level}_ssim': self.ssim[i],
                f'{level}_lpips': self.lpips[i],
                f'{level}_depth': self.depth_mse[i],
                f'processed_time': self.process_time[i]}

    def results_dict(self, level):
        return {self.file_id[i]: self.single_dict(i, level) for i in range(len(self.psnr))}

    def mean_dict(self, level):
        result_dict = {f'{level}_mean_psnr': self.psnr.mean(),
                       f'{level}_mean_ssim': self.ssim.mean(),
                       f'{level}_mean_lpips': self.lpips.mean(),
                       f'{level}_mean_depth': self.depth_mse.mean()}

        if level == 'fine':
            result_dict['time'] = self.process_time.mean()

        return result_dict

    def __str__(self):
        return f"psnr: {self.psnr.mean():03f}, ssim: {self.ssim.mean():03f} \n"


def get_table_fmt(width, column, float_fmt):
    return ' | '.join([f'{{:{width}}}'] + [f'{{:{width}{float_fmt}}}'] * column)


def get_separator(fmt, width, column):
    return fmt.format(*['-' * width for _ in [width] * (column + 1 + 4)])


TABLE_WIDTH = 15
TABLE_COLUMN = 9
FLOAT_FMT = get_table_fmt(TABLE_WIDTH, TABLE_COLUMN, '.4f')
STRING_FMT = get_table_fmt(TABLE_WIDTH, TABLE_COLUMN, '')
SEPARATOR = get_separator(STRING_FMT, TABLE_WIDTH, TABLE_COLUMN)


class SceneEvaluator:
    CMAP = plt.get_cmap('jet')

    def __init__(self, additional_eval_args, differ_args):
        """

        :param additional_eval_args: additional args to pass to the evaluation
               (different from the default config, like ckpt, scenes)
        :param differ_args: if args.same = True in the default eval config, the model setup is copied from the train
               config. These are args that should be different from the training setup when using args.same = True
        """

        self.summary_obj_dict: Dict[str, Summary] = {'coarse': Summary(), 'fine': Summary()}

        self.test_loader, self.scene_name, self.res_dir, self.eval_args, self.model = init_eval(additional_eval_args,
                                                                                                open_dir=False,
                                                                                                differ_from_train_args=differ_args)
        self.model.switch_to_eval()
        self.device = torch.device(f'cuda:{self.eval_args.local_rank}')
        self.ours = self.is_ours()

    @property
    def coarse_sum(self):
        return self.summary_obj_dict['coarse']

    @property
    def fine_sum(self):
        return self.summary_obj_dict['fine']

    def __str__(self):
        return f"running mean coarse: {self.coarse_sum}\nrunning mean fine:{self.fine_sum}"

    @classmethod
    def scene_evaluation(cls, add_args, differ_args):
        evaluator = cls(add_args, differ_args)
        return evaluator.start_scene_evaluation()

    def start_scene_evaluation(self):
        for i, data in enumerate(self.test_loader):
            self.evaluate_single_burst(data)

        sum_results_dict = {self.scene_name: {}}
        sum_results_dict[self.scene_name].update(self.coarse_sum.mean_dict('coarse'))
        sum_results_dict[self.scene_name].update(self.fine_sum.mean_dict('fine'))

        if self.eval_args.eval_dataset != 'usfm':
            filename = f'{self.eval_args.post}psnr_{self.scene_name}_{self.model.start_step}'
            results_dict = self.get_all_results_dict()
            self.save_npy(results_dict, self.res_dir.parent / f'{filename}.npy')
            self.print_result(results_dict, sum_results_dict, f=None)
            with open(str(self.res_dir.parent / f"{filename}.txt"), "w") as f:
                self.print_result(results_dict, sum_results_dict, f=f)

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
        if self.eval_args.rerun and self.eval_args.process_output:
            pred_rgb = de_linearize(pred_rgb, data['white_level'])
        if self.eval_args.eval_dataset == 'usfm':
            pred_rgb = pred_rgb * data['white_level']

        if gt_rgb is not None:
            psnr, ssim, lpips = calculate_metrics(pred_rgb, gt_rgb)
        else:
            psnr, ssim, lpips = 0, 0, 0

        # predicted depth, save depth_map
        pred_depth, depth_error = self.depth_evaluation(rays_output, data, level, file_id)

        self.summary_obj_dict[level].update(psnr_val=psnr, lpips_val=lpips, ssim_val=ssim, depth_mse=depth_error)

        if self.eval_args.rerun:
            # saving outputs: predicted rgb, accuracy map, predicted depth,  warp images
            self.save_outputs(gt_rgb, pred_rgb, file_id, level, rays_output, pred_depth, ray_sampler, data)

    def is_ours(self):
        is_deeprep = self.res_dir.parent.parent.parent.stem == 'deeprep'
        is_bpn = self.res_dir.parent.parent.parent.stem == 'bpn'
        ours = not is_bpn and not is_deeprep
        assert not (not ours and self.eval_args.rerun)
        return ours

    def evaluate_single_burst(self, data):
        rgb_path = data['rgb_path'][0]
        file_id = os.path.basename(rgb_path).split('.')[0]
        self.summary_obj_dict['fine'].file_id.append(file_id)
        self.summary_obj_dict['coarse'].file_id.append(file_id)

        with torch.no_grad():
            ray_sampler = RaySampler(data, device=self.device, render_stride=self.eval_args.render_stride)

        if self.eval_args.eval_images:
            print("********** evaluate images ***********")
            self.evaluate_images(data, ray_sampler, file_id)

        if self.eval_args.eval_rays and self.eval_args.factor == 4:
            print("********** evaluate rays   ***********")
            self.evaluate_rays(data, ray_sampler)

    def save_input(self, data, file_id):
        src_rgbs = data['src_rgbs'][0].cpu().numpy()
        averaged_img = np.mean(src_rgbs, axis=0)

        noisy_rgb = data['rgb'][0]

        if self.eval_args.process_output:
            noisy_rgb = de_linearize(noisy_rgb, data['white_level'])
            averaged_img = de_linearize(averaged_img, data['white_level']).cpu().numpy()
        if self.eval_args.eval_dataset == 'usfm':
            noisy_rgb = noisy_rgb * data['white_level']
            averaged_img = averaged_img * data['white_level'].numpy()
        imwrite(str(self.res_dir / f"{file_id}_noisy.png"), to_uint(noisy_rgb.cpu().numpy()))
        imwrite(str(self.res_dir / f"{file_id}_average.png"), to_uint(averaged_img))

    def load_results(self, file_id):
        pred_fine = torch.tensor(imread(self.res_dir / f"{self.eval_args.post}{file_id}_pred_fine.png"))

        if self.ours:
            pred_depth_fine = torch.tensor(
                imread(self.res_dir / f"{file_id}_depth_fine.png", to_float=False).__array__() / 1000)
            pred_coarse = torch.tensor(imread(self.res_dir / f"{file_id}_pred_coarse.png"))
            pred_depth_coarse = torch.tensor(
                imread(self.res_dir / f"{file_id}_depth_coarse.png", to_float=False).__array__() / 1000)
        else:
            pred_depth_fine = torch.zeros_like(pred_fine[..., 0])
            pred_coarse = torch.zeros_like(pred_fine)
            pred_depth_coarse = torch.zeros_like(pred_fine[..., 0])

        rays_output = {'coarse': RaysOutput(rgb_map=pred_coarse, depth_map=pred_depth_coarse),
                       'fine': RaysOutput(rgb_map=pred_fine, depth_map=pred_depth_fine)}
        process_time = 0

        return rays_output, process_time

    def process_gt(self, data, ray_sampler):
        if 'rgb_clean' in data:
            gt_rgb = data['rgb_clean'][0][::ray_sampler.render_stride, ::ray_sampler.render_stride]
            # always process rgb, since it came from dataloader which perform unprocessing
            # same idea as "Unprocessing Images for Learned Raw Denoising"

            if self.eval_args.process_output:
                gt_rgb = de_linearize(gt_rgb, data['white_level'])
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
            err_map_colored = to_uint(self.CMAP(err_map)[..., :3])
            imwrite(str(self.res_dir / f"{file_id}_err_map_{level}.png"), err_map_colored)

        # predicted rgb
        pred_rgb = to_uint(pred_rgb.numpy())
        imwrite(str(self.res_dir / f"{file_id}_pred_{level}.png"), pred_rgb)

        # accuracy map
        acc_map = torch.sum(rays_output[level].weights, dim=-1).detach().cpu()
        acc_map_colored = to_uint(self.CMAP(acc_map)[..., :3])
        imwrite(str(self.res_dir / f"{file_id}_acc_map_{level}.png"), acc_map_colored)

        # predicted depth (depth error is saved under calc_depth)
        imwrite(str(self.res_dir / f"{file_id}_depth_{level}.png"), (pred_depth * 1000).astype(np.uint16))

        depth_range = data['depth_range']
        norm_depth = plt.Normalize(vmin=depth_range.squeeze()[0], vmax=depth_range.squeeze()[1])
        pred_depth_colored = to_uint(self.CMAP(norm_depth(pred_depth))[..., :3])
        imwrite(str(self.res_dir / f"{file_id}_depth_vis_{level}.png"), pred_depth_colored)

        # warp images
        if ray_sampler.render_stride == 1:
            warped_img_rgb = self.warped_images_by_depth(rays_output[level], data)
            if self.eval_args.rerun and self.eval_args.process_output:
                warped_img_rgb = de_linearize(warped_img_rgb, data['white_level'])
            if self.eval_args.eval_dataset == 'usfm':
                warped_img_rgb = warped_img_rgb * data['white_level']
            warped_img_rgb = to_uint(warped_img_rgb.cpu().numpy())
            imwrite(str(self.res_dir / f"{file_id}_warped_images_{level}.png"), warped_img_rgb)

            summary_image = np.concatenate((pred_rgb, warped_img_rgb, pred_depth_colored), axis=1)
        else:
            summary_image = np.concatenate((pred_rgb, pred_depth_colored), axis=1)

        imwrite(str(self.res_dir / f"{file_id}_sum_{level}.png"), summary_image)

    def depth_evaluation(self, rays_output, data, level, file_id):
        depth_range = data['depth_range']
        norm_depth = plt.Normalize(vmin=0, vmax=depth_range.squeeze()[1] / 20)
        pred_depth = rays_output[level].depth.detach().cpu().numpy().squeeze()
        if 'gt_depth' in data:  # gt depth in the paper is the depth of the original IBRNet on the clean images
            gt_depth = data['gt_depth'].detach().cpu().numpy().squeeze()
            if gt_depth.shape != () and gt_depth.shape == pred_depth.shape and pred_depth.sum() > 0:
                depth_error_map = ((gt_depth - pred_depth) ** 2)
                depth_error = depth_error_map.mean()
                imwrite(str(self.res_dir / f"{file_id}_depth_error_{level}.png"), to_uint(norm_depth(depth_error_map)))
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

    def evaluate_rays(self, data, ray_sampler):
        """
        Method for debug processing of rays of specific pixels
        @param ray_sampler:
        @param data: burst data.
        @return:
        """
        # Only for llff test dataset. Should work for every dataset
        if self.scene_name == 'fern':
            save_pixel = ((271, 926), (350, 69), (558, 339))
        elif self.scene_name == 'orchids':
            save_pixel = ((392, 353), (300, 700))
        elif self.scene_name == 'trex':
            save_pixel = ((387, 411), (300, 700))
        elif self.scene_name == 'horns':
            save_pixel = ((392, 353), (362, 187))
        elif self.scene_name == 'flower':
            save_pixel = ((392, 353), (362, 187))
        elif self.scene_name == 'fortress':
            save_pixel = ((392, 353), (362, 187))
        elif self.scene_name == 'leaves':
            save_pixel = ((35, 35),)
        elif self.scene_name == 'room':
            save_pixel = ((392, 353), (362, 187))
        else:
            return

        ray_render = RayRender(model=self.model, args=self.eval_args, device=self.device, save_pixel=save_pixel)
        self.model.switch_to_eval()
        with torch.no_grad():
            org_src_rgbs = ray_sampler.src_rgbs.to(self.device)
            proc_src_rgbs, featmaps = ray_render.calc_featmaps(src_rgbs=org_src_rgbs)
            ray_batch_in = ray_sampler.sample_ray_batch_from_pixel(save_pixel)
            ray_batch_out = ray_render.render_batch(ray_batch=ray_batch_in, proc_src_rgbs=proc_src_rgbs,
                                                    featmaps=featmaps,
                                                    org_src_rgbs=org_src_rgbs,
                                                    sigma_estimate=ray_sampler.sigma_estimate.to(self.device))

            analyze_per_pixel(ray_batch_out, data, save_pixel, self.res_dir, show=False)

    def evaluate_images(self, data, ray_sampler, file_id):
        if self.eval_args.rerun:
            self.save_input(data, file_id)

            with torch.no_grad():
                start = time.time()
                rays_output = render_single_image(ray_sampler=ray_sampler, model=self.model, args=self.eval_args)
                process_time = time.time() - start
                self.summary_obj_dict['fine'].process_time.append(process_time)
                self.summary_obj_dict['coarse'].process_time.append(process_time)
        else:
            rays_output, process_time = self.load_results(file_id)

        self.sum_burst_output(data, rays_output, file_id, ray_sampler)

    def print_result(self, results_dict, sum_results_dict, f):
        for scene, scene_dict in results_dict.items():
            print(f"{self.res_dir.parent.parent.parent.name}", file=f)
            print(f"{scene}, {self.model.start_step}", file=f)

            for i, (k, v) in enumerate(scene_dict.items()):
                if i == 0:
                    # print title
                    print(STRING_FMT.format('file_id', *v.keys()), file=f)
                    print(SEPARATOR, file=f)
                print(FLOAT_FMT.format(k, *v.values()), file=f)
                print(SEPARATOR, file=f)

            print(SEPARATOR, file=f)
            print(FLOAT_FMT.format('mean', *sum_results_dict[scene].values()), file=f)
            print(SEPARATOR, file=f)

    @staticmethod
    def save_npy(results_dict, fpath):
        assert len(results_dict) == 1
        results_dict = list(results_dict.values())[0]
        res = np.array([[v for v in im_res.values()] for (image, im_res) in results_dict.items()])
        np.save(fpath, res)

    def warped_images_by_depth(self, output, data):
        depth = output.depth.detach().to(self.device)
        Ki = data['src_cameras'][:, :, 2:18].reshape((-1, 4, 4)).unsqueeze(0)
        Rti = data['src_cameras'][:, :, 18:34].reshape((-1, 4, 4)).unsqueeze(0)
        images = data['src_rgbs'].permute((0, 1, 4, 2, 3))
        warped_images = warp_KRt_wrapper(images.to(self.device), Ki.to(self.device),
                                         Rti.inverse().to(self.device), 1 / depth)
        warped_images = warped_images[0].mean(0).permute((1, 2, 0))
        warped_images_rgb = to_uint(warped_images.cpu().numpy().squeeze())
        return warped_images_rgb
