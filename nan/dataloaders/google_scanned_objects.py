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
from nan.dataloaders.basic_dataset import NoiseDataset
from nan.dataloaders.data_utils import get_nearest_pose_ids


# only for training
class GoogleScannedDataset(NoiseDataset):
    dir_name = 'google_scanned_objects'

    def __init__(self, args, mode, **kwargs):
        self.all_rgb_files = []
        self.all_pose_files = []
        self.all_intrinsics_files = []
        super().__init__(args, mode, **kwargs)

        index = np.arange(len(self.all_rgb_files))
        self.all_rgb_files = np.array(self.all_rgb_files)[index]
        self.all_pose_files = np.array(self.all_pose_files)[index]
        self.all_intrinsics_files = np.array(self.all_intrinsics_files)[index]

    def add_single_scene(self, _, scene_path):
        num_files = 250
        rgb_files = list((scene_path / 'rgb').glob("*"))
        pose_files = [Path(str(f).replace('rgb', 'pose').replace('png', 'txt')) for f in rgb_files]
        intrinsics_files = [Path(str(f).replace('rgb', 'intrinsics').replace('png', 'txt')) for f in rgb_files]

        if np.min([len(rgb_files), len(pose_files), len(intrinsics_files)]) < num_files:
            print(scene_path)
            return

        self.all_rgb_files.append(rgb_files)
        self.all_pose_files.append(pose_files)
        self.all_intrinsics_files.append(intrinsics_files)

    def __len__(self):
        return len(self.all_rgb_files)

    def final_depth_range(self, render_pose):
        # get depth range
        min_ratio = 0.1
        origin_depth = np.linalg.inv(render_pose)[2, 3]
        max_radius = 0.5 * np.sqrt(2) * 1.1
        near_depth = max(origin_depth - max_radius, min_ratio * origin_depth)
        far_depth = origin_depth + max_radius
        depth_range = torch.tensor([near_depth, far_depth])
        return depth_range

    def __getitem__(self, idx):
        rgb_files = self.all_rgb_files[idx]
        pose_files = self.all_pose_files[idx]
        intrinsics_files = self.all_intrinsics_files[idx]

        id_render = np.random.choice(np.arange(len(rgb_files)))
        train_poses = np.stack([np.loadtxt(file).reshape(4, 4) for file in pose_files], axis=0)
        render_pose = train_poses[id_render]
        subsample_factor = np.random.choice(np.arange(1, 6), p=[0.3, 0.25, 0.2, 0.2, 0.05])

        id_feat_pool = get_nearest_pose_ids(render_pose,
                                            train_poses,
                                            self.num_source_views*subsample_factor,
                                            tar_id=id_render,
                                            angular_dist_method='vector')
        id_feat = self.choose_views(id_feat_pool, self.num_source_views, id_render)

        rgb = self.read_image(rgb_files[id_render])

        intrinsics = np.loadtxt(intrinsics_files[id_render])
        camera = self.create_camera_vector(rgb, intrinsics, render_pose)

        depth_range = self.final_depth_range(render_pose=render_pose)

        src_rgbs = []
        src_cameras = []
        for idx in id_feat:
            src_rgb = self.read_image(rgb_files[idx])
            pose = np.loadtxt(pose_files[idx])

            src_rgbs.append(src_rgb)
            intrinsics = np.loadtxt(intrinsics_files[idx])
            src_camera = self.create_camera_vector(src_rgb, intrinsics, pose)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs)
        src_cameras = np.stack(src_cameras)

        return self.create_batch_from_numpy(rgb, camera, rgb_files[id_render], src_rgbs, src_cameras, depth_range)
