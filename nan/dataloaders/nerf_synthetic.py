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
import numpy as np
import imageio
import torch
import json

from nan.dataloaders.basic_dataset import NoiseDataset, Mode
from nan.dataloaders.data_utils import get_nearest_pose_ids


def read_cameras(pose_file):
    basedir = os.path.dirname(pose_file)
    with open(pose_file, 'r') as fp:
        meta = json.load(fp)

    camera_angle_x = float(meta['camera_angle_x'])
    rgb_files = []
    c2w_mats = []

    img = imageio.imread(os.path.join(basedir, meta['frames'][0]['file_path'] + '.png'))
    H, W = img.shape[:2]
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    intrinsics = get_intrinsics_from_hwf(H, W, focal)

    for i, frame in enumerate(meta['frames']):
        rgb_file = os.path.join(basedir, meta['frames'][i]['file_path'][2:] + '.png')
        rgb_files.append(rgb_file)
        c2w = np.array(frame['transform_matrix'])
        w2c_blender = np.linalg.inv(c2w)
        w2c_opencv = w2c_blender
        w2c_opencv[1:3] *= -1
        c2w_opencv = np.linalg.inv(w2c_opencv)
        c2w_mats.append(c2w_opencv)
    c2w_mats = np.array(c2w_mats)
    return rgb_files, np.array([intrinsics] * len(meta['frames'])), c2w_mats


def get_intrinsics_from_hwf(h, w, focal):
    return np.array([[focal, 0, 1.0 * w / 2, 0],
                     [0, focal, 1.0 * h / 2, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


class NerfSyntheticDataset(NoiseDataset):
    dir_name = 'nerf_synthetic'

    def __init__(self, args, mode, scenes=(), **kwargs):
        self.testskip = args.testskip
        self.render_rgb_files = []
        self.render_poses = []
        self.render_intrinsics = []
        super().__init__(args, mode, scenes=scenes, **kwargs)

    def get_all_scenes(self):
        return [self.folder_path / scene for scene in ('chair', 'drums', 'lego', 'hotdog', 'materials', 'mic', 'ship')]

    def add_single_scene(self, _, scene_path):
        pose_file = os.path.join(scene_path, f'transforms_{self.mode}.json')
        rgb_files, intrinsics, poses = read_cameras(pose_file)
        if self.mode != Mode.train:
            rgb_files = rgb_files[::self.testskip]
            intrinsics = intrinsics[::self.testskip]
            poses = poses[::self.testskip]
        self.render_rgb_files.extend(rgb_files)
        self.render_poses.extend(poses)
        self.render_intrinsics.extend(intrinsics)

    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.render_rgb_files[idx]
        render_pose = self.render_poses[idx]
        render_intrinsics = self.render_intrinsics[idx]

        train_pose_file = os.path.join('/'.join(rgb_file.split('/')[:-2]), 'transforms_train.json')
        train_rgb_files, train_intrinsics, train_poses = read_cameras(train_pose_file)

        if self.mode == Mode.train:
            id_render = int(os.path.basename(rgb_file)[:-4].split('_')[1])
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.3, 0.5, 0.2])
        else:
            id_render = -1
            subsample_factor = 1

        rgb = self.read_image(rgb_file)
        rgb = rgb[..., [-1]] * rgb[..., :3] + 1 - rgb[..., [-1]]
        camera = self.create_camera_vector(rgb, render_intrinsics, render_pose)

        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                int(self.num_source_views * subsample_factor),
                                                tar_id=id_render,
                                                angular_dist_method='vector')
        nearest_pose_ids = self.choose_views(nearest_pose_ids, self.num_source_views, id_render)

        src_rgbs = []
        src_cameras = []
        for idx in nearest_pose_ids:
            src_rgb = self.read_image(train_rgb_files[idx])
            src_rgb = src_rgb[..., [-1]] * src_rgb[..., :3] + 1 - src_rgb[..., [-1]]
            train_pose = train_poses[idx]
            train_intrinsics_ = train_intrinsics[idx]

            src_rgbs.append(src_rgb)
            src_camera = self.create_camera_vector(src_rgb, train_intrinsics_, train_pose)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        depth_range = self.final_depth_range()

        return self.create_batch_from_numpy(rgb, camera, rgb_file, src_rgbs, src_cameras, depth_range)

    def final_depth_range(self):
        near_depth = 2.
        far_depth = 6.

        depth_range = torch.tensor([near_depth, far_depth])
        return depth_range
