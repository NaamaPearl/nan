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
import numpy as np
import torch

from nan.dataloaders.basic_dataset import NoiseDataset, Mode
from nan.dataloaders.data_utils import deepvoxels_parse_intrinsics, get_nearest_pose_ids


class DeepVoxelsDataset(NoiseDataset):
    dir_name = 'deepvoxels'

    def folder_path(self) -> Path:
        return super().folder_path / self.mode

    def __init__(self, args, mode, scenes='vase', **kwargs):
        self.testskip = args.testskip
        self.all_rgb_files = []
        self.all_depth_files = []
        self.all_pose_files = []
        self.all_intrinsics_files = []
        super().__init__(args, mode, scenes=scenes, **kwargs)

    def add_single_scene(self, _, scene_path):
        rgb_files = sorted((scene_path / 'rgb').glob("*"))
        if self.mode != Mode.train:
            rgb_files = rgb_files[::self.testskip]
        depth_files = [Path(str(f).replace('rgb', 'depth')) for f in rgb_files]
        pose_files = [Path(str(f).replace('rgb', 'pose').replace('png', 'txt')) for f in rgb_files]
        intrinsics_file = scene_path / 'intrinsics.txt'
        self.all_rgb_files.extend(rgb_files)
        self.all_depth_files.extend(depth_files)
        self.all_pose_files.extend(pose_files)
        self.all_intrinsics_files.extend([intrinsics_file]*len(rgb_files))

    def __len__(self):
        return len(self.all_rgb_files)

    def __getitem__(self, idx):
        idx = idx % len(self.all_rgb_files)
        rgb_file = self.all_rgb_files[idx]
        pose_file = self.all_pose_files[idx]
        intrinsics_file = self.all_intrinsics_files[idx]
        intrinsics = deepvoxels_parse_intrinsics(intrinsics_file, 512)[0]
        scene_path = rgb_file.parent.parent
        train_rgb_files = list((Path(str(scene_path).replace(f'/{self.mode}/', '/train/')) / 'rgb').glob('*'))
        train_poses_files = [Path(str(f).replace('rgb', 'pose').replace('png', 'txt')) for f in train_rgb_files]
        train_poses = np.stack([np.loadtxt(str(file)).reshape(4, 4) for file in train_poses_files], axis=0)

        if self.mode == Mode.train:
            id_render = train_poses_files.index(pose_file)
            subsample_factor = np.random.choice(np.arange(1, 5))
            num_source_views = np.random.randint(low=self.num_source_views-4, high=self.num_source_views+2)
        else:
            id_render = None
            subsample_factor = 1
            num_source_views = self.num_source_views

        rgb = self.read_image(rgb_file)
        render_pose = np.loadtxt(pose_file).reshape(4, 4)
        camera = self.create_camera_vector(rgb, intrinsics, render_pose)

        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                min(num_source_views*subsample_factor, 40),
                                                tar_id=id_render,
                                                angular_dist_method='vector')

        nearest_pose_ids = self.choose_views(nearest_pose_ids, num_source_views, id_render)

        src_rgbs = []
        src_cameras = []
        for idx in nearest_pose_ids:
            src_rgb = self.read_image(train_rgb_files[idx])
            train_pose = train_poses[idx]

            src_rgbs.append(src_rgb)
            src_camera = self.create_camera_vector(src_rgb, intrinsics, train_pose)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        depth_range = self.final_depth_range(render_pose=render_pose, rgb_file=rgb_file)

        return self.create_batch_from_numpy(rgb, camera, rgb_file, src_rgbs, src_cameras, depth_range)

    def final_depth_range(self, render_pose, rgb_file):
        origin_depth = np.linalg.inv(render_pose.reshape(4, 4))[2, 3]

        if 'cube' in rgb_file:
            near_depth = origin_depth - 1.
            far_depth = origin_depth + 1
        else:
            near_depth = origin_depth - 0.8
            far_depth = origin_depth + 0.8

        depth_range = torch.tensor([near_depth, far_depth])
        return depth_range
