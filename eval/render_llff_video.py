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
import os
from pathlib import Path

import numpy as np
import torch
import sys

from configs.local_setting import LOG_DIR, VIDEO_CONFIG
from eval.init_eval import rearrange_args_for_eval
from nan.dataloaders import load_llff_data, batch_parse_llff_poses, get_nearest_pose_ids, LLFFTestDataset
from nan.dataloaders.basic_dataset import Mode
from nan.model import NANScheme
from nan.render_image import render_single_image
from nan.sample_ray import RaySampler
from nan.utils.io_utils import print_link, colorize_np

sys.path.append('../')
from torch.utils.data import DataLoader
import imageio
import time


class LLFFRenderDataset(LLFFTestDataset):
    def __init__(self, args, mode, **kwargs):
        self.h = []
        self.w = []
        super().__init__(args, mode, **kwargs)

    def add_single_scene(self, i, scene_path):
        _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(scene_path, load_imgs=False, factor=4)
        near_depth = np.min(bds)
        far_depth = np.max(bds)
        intrinsics, c2w_mats = batch_parse_llff_poses(poses)
        h, w = poses[0][:2, -1]
        render_intrinsics, render_c2w_mats = batch_parse_llff_poses(render_poses)

        i_test = [i_test]
        i_val = i_test
        i_train = np.array([i for i in np.arange(len(rgb_files)) if
                            (i not in i_test and i not in i_val)])

        self.src_intrinsics.append(intrinsics[i_train])
        self.src_poses.append(c2w_mats[i_train])
        self.src_rgb_files.append(np.array(rgb_files)[i_train].tolist())

        num_render = len(render_intrinsics)
        self.render_intrinsics.extend([intrinsics_ for intrinsics_ in render_intrinsics])
        self.render_poses.extend([c2w_mat for c2w_mat in render_c2w_mats])
        self.render_depth_range.extend([[near_depth, far_depth]] * num_render)
        self.render_train_set_ids.extend([i] * num_render)
        self.h.extend([int(h)] * num_render)
        self.w.extend([int(w)] * num_render)

    def __len__(self):
        return len(self.render_poses)

    def __getitem__(self, idx):
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.src_rgb_files[train_set_id]
        train_poses = self.src_poses[train_set_id]
        train_intrinsics = self.src_intrinsics[train_set_id]

        h, w = self.h[idx], self.w[idx]
        camera = np.concatenate(([h, w], intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        id_render = -1
        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                self.num_source_views,
                                                tar_id=id_render,
                                                angular_dist_method='dist')

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                         train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])

        return self.create_batch_from_numpy(None, camera, None, src_rgbs, src_cameras, depth_range,
                                            gt_depth=None)


if __name__ == '__main__':
    sys.argv += ['--config', str(VIDEO_CONFIG)]
    differ_from_train_args = [('factor', 4), ('eval_gain', 16)]
    additional_eval_args = ['--eval_scenes', 'fern']
    train_args, eval_args = rearrange_args_for_eval(additional_eval_args, differ_from_train_args)

    # Create NAN model
    if eval_args.same:
        model = NANScheme(train_args)
    else:
        model = NANScheme(eval_args)

    scene_name = eval_args.eval_scenes[0]
    res_dir : Path = LOG_DIR / train_args.expname / eval_args.eval_dataset / eval_args.expname / \
                    f"{scene_name}_{model.start_step:06d}" / 'videos'
    res_dir.mkdir(parents=True, exist_ok=True)
    print_link(res_dir, 'Saving results to ')

    assert len(eval_args.eval_scenes) == 1, "only accept single scene"
    test_dataset = LLFFRenderDataset(eval_args, Mode.test, scenes=eval_args.eval_scenes)
    save_prefix = scene_name
    test_loader = DataLoader(test_dataset, batch_size=1)
    total_num = len(test_loader)
    out_frames = []
    crop_ratio = 0.075
    device = torch.device(f'cuda:{eval_args.local_rank}')
    for i, data in enumerate(test_loader):
        start = time.time()
        src_rgbs = data['src_rgbs'][0].cpu().numpy()
        averaged_img = (np.mean(src_rgbs, axis=0) * 255.).astype(np.uint8)
        imageio.imwrite(os.path.join(res_dir, '{}_average.png'.format(i)), averaged_img)

        model.switch_to_eval()
        with torch.no_grad():
            ray_sampler = RaySampler(data, device=device, render_stride=eval_args.render_stride)
            rays_output = render_single_image(ray_sampler=ray_sampler, model=model, args=eval_args)
            torch.cuda.empty_cache()

        coarse_pred_rgb = rays_output['coarse'].rgb.detach().cpu()
        coarse_pred_rgb = (255 * np.clip(coarse_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
        imageio.imwrite(os.path.join(res_dir, '{}_pred_coarse.png'.format(i)), coarse_pred_rgb)

        coarse_pred_depth = rays_output['coarse'].depth.detach().cpu()
        imageio.imwrite(os.path.join(res_dir, '{}_depth_coarse.png'.format(i)),
                        (coarse_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
        coarse_pred_depth_colored = colorize_np(coarse_pred_depth,
                                                range=tuple(data['depth_range'].squeeze().numpy()))
        imageio.imwrite(os.path.join(res_dir, '{}_depth_vis_coarse.png'.format(i)),
                        (255 * coarse_pred_depth_colored).astype(np.uint8))

        coarse_acc_map = torch.sum(rays_output['coarse'].weights.detach().cpu(), dim=-1)
        coarse_acc_map_colored = (colorize_np(coarse_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(res_dir, '{}_acc_map_coarse.png'.format(i)),
                        coarse_acc_map_colored)

        if rays_output['fine'] is not None:
            fine_pred_rgb = rays_output['fine'].rgb.detach().cpu()
            fine_pred_rgb = (255 * np.clip(fine_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            imageio.imwrite(os.path.join(res_dir, '{}_pred_fine.png'.format(i)), fine_pred_rgb)
            fine_pred_depth = rays_output['fine'].depth.detach().cpu()
            imageio.imwrite(os.path.join(res_dir, '{}_depth_fine.png'.format(i)),
                            (fine_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
            fine_pred_depth_colored = colorize_np(fine_pred_depth,
                                                  range=tuple(data['depth_range'].squeeze().cpu().numpy()))
            imageio.imwrite(os.path.join(res_dir, '{}_depth_vis_fine.png'.format(i)),
                            (255 * fine_pred_depth_colored).astype(np.uint8))
            fine_acc_map = torch.sum(rays_output['fine'].weights.detach().cpu(), dim=-1)
            fine_acc_map_colored = (colorize_np(fine_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(res_dir, '{}_acc_map_fine.png'.format(i)),
                            fine_acc_map_colored)
        else:
            fine_pred_rgb = None

        out_frame = fine_pred_rgb if fine_pred_rgb is not None else coarse_pred_rgb
        h, w = averaged_img.shape[:2]
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        # crop out image boundaries
        out_frame = out_frame[crop_h:h - crop_h, crop_w:w - crop_w, :]
        out_frames.append(out_frame)

        print('frame {} completed, {}'.format(i, time.time() - start))

    imageio.mimwrite(os.path.join(res_dir, '{}.mp4'.format(scene_name)), out_frames, fps=3, quality=8)