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


# #### Modified version of LLFF dataset code
# #### see https://github.com/googleinterns/IBRNet for original
import sys
from abc import ABC
from copy import copy
from pathlib import Path

import exifread
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import rawpy
from tqdm import tqdm

from configs.config import CustomArgumentParser
from configs.local_setting import LOG_DIR, EVAL_CONFIG
from nan.dataloaders.basic_dataset import NoiseDataset
from nan.dataloaders.data_utils import random_crop, get_nearest_pose_ids, random_flip, to_uint
from nan.dataloaders.llff_data_utils import load_llff_data, batch_parse_llff_poses
from nan.utils.geometry_utils import warp_KRt_wrapper


class COLMAPDataset(NoiseDataset, ABC):
    name = 'colmap'

    def __init__(self, args, mode, scenes=(), random_crop=True, **kwargs):
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_ids = []
        self.render_depth_range = []

        self.src_intrinsics = []
        self.src_poses = []
        self.src_rgb_files = []
        super().__init__(args, mode, scenes=scenes, random_crop=random_crop, **kwargs)
        self.depth_range = self.render_depth_range[0]

    def get_i_test(self, N):
        return np.arange(N)[::self.args.llffhold]

    @staticmethod
    def get_i_train(N, i_test, mode):
        return np.array([j for j in np.arange(int(N)) if j not in i_test])

    @staticmethod
    def load_scene(scene_path, factor):
        return load_llff_data(scene_path, load_imgs=False, factor=factor)

    def __len__(self):
        return len(self.render_rgb_files) * 100000 if self.mode == 'train' else len(self.render_rgb_files)

    def __getitem__(self, idx):
        idx = idx % len(self.render_rgb_files)
        rgb_file: Path = self.render_rgb_files[idx]
        rgb = self.read_image(rgb_file)

        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.src_rgb_files[train_set_id]
        train_poses = self.src_poses[train_set_id]
        train_intrinsics = self.src_intrinsics[train_set_id]
        camera = self.create_camera(rgb, intrinsics, render_pose)

        if self.mode == 'train':
            if rgb_file in train_rgb_files:
                id_render = train_rgb_files.index(rgb_file)
            else:
                id_render = None
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views + np.random.randint(low=-2, high=self.num_select_high)
            id_render = id_render
        else:
            id_render = None
            subsample_factor = 1
            num_select = self.num_source_views

        nearest_pose_ids = self.get_nearest_pose_ids(render_pose, depth_range, train_poses, subsample_factor, id_render)
        nearest_pose_ids = self.choose_views(nearest_pose_ids, num_select, id_render)

        src_rgbs = []
        src_cameras = []
        for src_id in nearest_pose_ids:
            if src_id is None:
                # print(self.render_rgb_files[idx])
                src_rgb = self.read_image(self.render_rgb_files[idx])
                train_pose = self.render_poses[idx]
                train_intrinsics_ = self.render_intrinsics[idx]
            else:
                # print(train_rgb_files[src_id])
                src_rgb = self.read_image(train_rgb_files[src_id])
                train_pose = train_poses[src_id]
                train_intrinsics_ = train_intrinsics[src_id]

            src_rgbs.append(src_rgb)
            src_camera = self.create_camera(src_rgb, train_intrinsics_, train_pose)

            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        rgb, camera, src_rgbs, src_cameras = self.apply_transform(rgb, camera, src_rgbs, src_cameras)

        try:
            scene = rgb_file.parent.parent.stem
            gt_depth_file = (LOG_DIR / "pretraining____clean__l2" / self.name / "same" / f"{scene}_255000")
            gt_depth_file = list(gt_depth_file.glob("*"))[0] / f"{rgb_file.stem}_depth_fine.png"
            gt_depth = imageio.imread(gt_depth_file).__array__() / 1000
        except (FileNotFoundError, IndexError):
            gt_depth = 0

        depth_range = self.final_depth_range(depth_range)
        return self.create_batch_from_numpy(rgb, camera, rgb_file, src_rgbs, src_cameras, depth_range, gt_depth=gt_depth)

    def get_nearest_pose_ids(self, render_pose, depth_range, train_poses, subsample_factor, id_render):
        return get_nearest_pose_ids(render_pose,
                                    train_poses,
                                    min(self.num_source_views * subsample_factor, self.min_nearest_pose),
                                    tar_id=id_render,
                                    angular_dist_method='dist')

    def add_single_scene(self, i, scene_path):
        _, poses, bds, render_poses, i_test, rgb_files = self.load_scene(scene_path, self.args.factor)
        near_depth = bds.min()
        far_depth = bds.max()
        intrinsics, c2w_mats = batch_parse_llff_poses(poses)

        i_test = self.get_i_test(poses.shape[0])
        i_train = self.get_i_train(poses.shape[0], i_test, self.mode)

        if self.mode == 'train':
            i_render = i_train
        else:
            i_render = i_test

        # Source images
        self.src_intrinsics.append(intrinsics[i_train])
        self.src_poses.append(c2w_mats[i_train])
        self.src_rgb_files.append([rgb_files[i] for i in i_train])

        # Target images
        num_render = len(i_render)
        self.render_rgb_files.extend([rgb_files[i] for i in i_render])
        self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
        self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
        self.render_depth_range.extend([[near_depth, far_depth]] * num_render)
        self.render_train_set_ids.extend([i] * num_render)
        self.render_ids.extend(i_render)


class LLFFTestDataset(COLMAPDataset):
    name = 'llff_test'
    dir_name = 'nerf_llff_data'
    num_select_high = 2
    min_nearest_pose = 28

    def apply_transform(self, rgb, camera, src_rgbs, src_cameras):
        if self.mode == 'train' and self.random_crop:
            crop_h = np.random.randint(low=250, high=750)
            crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
            crop_w = int(400 * 600 / crop_h)
            crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w
            rgb, camera, src_rgbs, src_cameras = random_crop(rgb, camera, src_rgbs, src_cameras,
                                                             (crop_h, crop_w))

        if self.mode == 'train' and np.random.choice([0, 1]):
            rgb, camera, src_rgbs, src_cameras = random_flip(rgb, camera, src_rgbs, src_cameras)

        return rgb, camera, src_rgbs, src_cameras

    def final_depth_range(self, depth_range):
        return torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6])


class LLFFDataset(LLFFTestDataset):
    name = 'llff'
    dir_name = 'real_iconic_noface'
    num_select_high = 3
    min_nearest_pose = 20

    def __len__(self):
        return len(self.render_rgb_files)

    @staticmethod
    def get_i_train(N, i_test, mode):
        if mode == 'train':
            return np.array(np.arange(N))
        else:
            return super().get_i_train(N, i_test, mode)


class RealWorldDataset(LLFFTestDataset):
    dir_name = 'real_world'

    def __init__(self, args, mode, **kwargs):
        args = copy(args)
        args.eval_gain = 0
        print(f"[*] RealWorld Data - running on factor {args.factor}")
        print("[*] RealWorld Data - changing to gain 0")
        super().__init__(args, mode, **kwargs)

    # @staticmethod
    # def read_image(filename, **kwargs):
    #     return imageio.imread(filename).astype(np.float32) / 65535.

    def create_batch_from_numpy(self, rgb, camera, rgb_file, src_rgbs, src_cameras, depth_range, gt_depth=0):
        rgb      = torch.from_numpy(rgb[..., :3])
        src_rgbs = torch.from_numpy(src_rgbs[..., :3])

        sig_read, sig_shot = self.read_noise_params(rgb_file)
        sigma_est = self.get_std(src_rgbs.clamp(0, 1), sig_read, sig_shot)

        # wl = 0.1
        wl = rgb.max()
        # wl = 1

        return {'rgb_clean'     : rgb / wl,
                'rgb'           : rgb / wl,
                'gt_depth'      : gt_depth,
                'camera'        : torch.from_numpy(camera),
                'rgb_path'      : str(rgb_file),
                'src_rgbs_clean': src_rgbs / wl,
                'src_rgbs'      : src_rgbs / wl,
                'src_cameras'   : torch.from_numpy(src_cameras),
                'depth_range'   : depth_range,
                'sigma_estimate': sigma_est,
                'white_level'   : 1}

    @classmethod
    def read_noise_params(cls, rgb_file: Path, wl, sn):
        # From https://ipolcore.ipol.im/demo/clientApp/demo.html?id=336#

        dng_file = rgb_file.parent.parent / 'dng' / f"{rgb_file.stem}.dng"
        if not dng_file.exists():
            # raise FileNotFoundError(f"dng file {dng_file} was not found")
            print(f"[***] dng file {dng_file} was not found")
            return 0.01, 0.01

        with open(dng_file, 'rb') as raw_file:
            tags = exifread.process_file(raw_file)
        with rawpy.imread(str(dng_file)) as ref_rawpy:
            black_level = ref_rawpy.black_level_per_channel.copy()
            white_level = ref_rawpy.white_level

        if 'Image Tag 0xC761' not in tags:
            raise IOError(f"EXIF for {rgb_file} doesn't contain noise parameters")

        # more explanation in
        # https://www.adobe.com/content/dam/acom/en/products/photoshop/pdfs/dng_spec_1.4.0.0.pdf
        # https://www.exiv2.org/tags.html
        noise_profile = np.squeeze(tags['Image Tag 0xC761'].values)
        if len(noise_profile) == 2:
            lambda_sn = noise_profile[0]
            lambda_rn = noise_profile[1]
        else:  # if noiseProfile has one value per CFA color
            assert len(noise_profile) == 6
            # assert noise_profile[0] == noise_profile[2] == noise_profile[4], 'NoiseProfile tag is different for each channel'
            # assert noise_profile[1] == noise_profile[3] == noise_profile[5], 'NoiseProfile tag is different for each channel' # TODO Naama
            lambda_sn = min(noise_profile[::2])
            lambda_rn = min(noise_profile[1::2])

        # lambdaSn and lambdaRn are noise curve parameters for an image with values between 0 and 1
        # unnormalize: var(k*x) = k**2*var(x) = k**2*(lambdaSn*x+lambdaRn) = k*lambdaSn*(k*x) + k**2*lambdaRn
        k = (1 / wl) * sn  #
        # k = 1  #
        lambda_s = lambda_sn * k
        lambda_r = lambda_rn * k ** 2
        print(f'Noise curve parameters: {lambda_r=},{lambda_s=}')

        # TODO Naama
        #  I think that noise parameters in Adobe and HDR doesnt has the power
        #  eq (7) in http://www.ipol.im/pub/art/2021/336/
        sig_r = lambda_r ** 0.5
        sig_s = lambda_s ** 0.5
        print(f'Noise curve parameters: {sig_r=}, {sig_s=}')
        return sig_r, sig_s

    def choose_views(self, possible_views, num_views, target_view):
        # TODO doesnt work if llff_hold is different from num_source_views
        return [None] + list(range(num_views - 1))

    def __len__(self):
        return 1


# class RealWorldAgisoft(RealWorldDataset):
#     @staticmethod
#     def load_scene(scene_path, factor):
#         images, poses, bds, render_poses, i_test, imgfiles


if __name__ == '__main__':
    sys.argv = sys.argv[:1] + ['--config', str(EVAL_CONFIG), '--factor', str(4), '--eval_gain', str(0)]
    parser = CustomArgumentParser.config_parser()
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.local_rank}")

    scene = 'naamaDark2'
    dataset = RealWorldDataset(args, 'test', scenes=[scene])
    sample = dataset[0]
    rgb = sample['rgb']

    plt.figure()
    plt.imshow((rgb * 5) ** (1 / 2.4))
    plt.show()

    depth = torch.ones_like(rgb[..., 0])

    K0 = sample['camera'][2:18].reshape((-1, 4, 4))
    Rt0 = sample['camera'][18:34].reshape((-1, 4, 4))
    Ki = sample['src_cameras'][:, 2:18].reshape((-1, 4, 4))
    Rti = sample['src_cameras'][:, 18:34].reshape((-1, 4, 4))

    Ks = torch.cat((K0, Ki), dim=0).unsqueeze(0)  # add batch dimension
    Rts = torch.cat((Rt0, Rti), dim=0).unsqueeze(0)  # add batch dimension
    images = torch.cat((sample['rgb'].unsqueeze(0).unsqueeze(0).permute((0, 1, 4, 2, 3)), sample['src_rgbs'].unsqueeze(0).permute((0, 1, 4, 2, 3))), dim=1)

    plane_sweeping = []

    for d in tqdm(torch.linspace(sample['depth_range'][0], sample['depth_range'][1], 200)):
        warped_images = warp_KRt_wrapper(images.to(device), Ks.to(device), Rts.inverse().to(device), (1 / (depth * d)).to(device))
        warped_images = warped_images[0].mean().permute((1, 2, 0))
        warped_images_rgb = to_uint(warped_images.cpu().numpy().squeeze())
        plane_sweeping.append(warped_images_rgb ** (1 / 2.4))

    writer = imageio.get_writer(f'D:\\Naama\\Projects\\IBRNet\\data\\nerf_llff_data\\{scene}.mp4', fps=10)
    for im in plane_sweeping:
        writer.append_data(im)
    writer.close()





