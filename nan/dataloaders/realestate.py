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
from nan.dataloaders import NoiseDataset
from torch.nn import functional as F


class Camera:
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.intrinsics = np.array([[fx, 0, cx, 0],
                                    [0, fy, cy, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def unnormalize_intrinsics(intrinsics, h, w):
    intrinsics[0] *= w
    intrinsics[1] *= h
    return intrinsics


def parse_pose_file(file):
    f = open(file, 'r')
    cam_params = {}
    for i, line in enumerate(f):
        if i == 0:
            continue
        entry = [float(x) for x in line.split()]
        idx = int(entry[0])
        cam_params[idx] = Camera(entry)
    return cam_params


# only for training
class RealEstateDataset(NoiseDataset):
    dir_name = 'RealEstate10K-subset'

    def folder_path(self) -> Path:
        return super().folder_path / self.mode / 'frames'

    def __init__(self, args, mode, **kwargs):
        self.target_h, self.target_w = 450, 800
        self.all_rgb_files = []
        self.all_timestamps = []
        super().__init__(args, mode, **kwargs)

        index = np.arange(len(self.all_rgb_files))
        self.all_rgb_files = np.array(self.all_rgb_files)[index]
        self.all_timestamps = np.array(self.all_timestamps)[index]

    def add_single_scene(self, _, scene_path):
        print(scene_path)
        rgb_files = list(scene_path.glob("*"))
        if len(rgb_files) < 10:
            print('omitting {}, too few images'.format(os.path.basename(scene_path)))
            return
        timestamps = [int(os.path.basename(rgb_file).split('.')[0]) for rgb_file in rgb_files]
        sorted_ids = np.argsort(timestamps)
        self.all_rgb_files.append(np.array(rgb_files)[sorted_ids])
        self.all_timestamps.append(np.array(timestamps)[sorted_ids])

    def __len__(self):
        return len(self.all_rgb_files)

    def final_depth_range(self):
        return torch.tensor([1., 100.])

    @staticmethod
    def read_image(filename, **kwargs):
        h = kwargs['h']
        w = kwargs['w']
        rgb = super().read_image(**kwargs)
        # resize the image to target size
        rgb = F.interpolate(torch.from_numpy(rgb).unsqueeze(0).permute(0, 3, 1, 2),
                            size=(h, w)).permute(0, 2, 3, 1)[0].numpy()
        assert rgb.shape[:2] == (h, w)
        return rgb

    def __getitem__(self, idx):
        rgb_files = self.all_rgb_files[idx]
        timestamps = self.all_timestamps[idx]

        assert (timestamps == sorted(timestamps)).all()
        num_frames = len(rgb_files)
        window_size = 32
        shift = np.random.randint(low=-1, high=2)
        id_render = np.random.randint(low=4, high=num_frames - 4 - 1)

        right_bound = min(id_render + window_size + shift, num_frames - 1)
        left_bound = max(0, right_bound - 2 * window_size)
        candidate_ids = np.arange(left_bound, right_bound)
        # remove the query frame itself
        candidate_ids = candidate_ids[candidate_ids != id_render]
        id_feat = self.choose_views(candidate_ids, self.num_source_views, id_render)

        rgb_file = rgb_files[id_render]
        rgb = self.read_image(rgb_files[id_render], h=self.target_h, w=self.target_w)

        camera_file = os.path.dirname(rgb_file).replace('frames', 'cameras') + '.txt'
        cam_params = parse_pose_file(camera_file)
        cam_param = cam_params[timestamps[id_render]]

        camera = self.create_camera_vector(rgb,
                                           unnormalize_intrinsics(cam_param.intrinsics, self.target_h, self.target_w),
                                           cam_param.c2w_mat)

        # get depth range
        depth_range = self.final_depth_range()

        src_rgbs = []
        src_cameras = []
        for idx in id_feat:
            src_rgb = self.read_image(rgb_files[idx], h=self.target_h, w=self.target_w)
            src_rgbs.append(src_rgb)
            cam_param = cam_params[timestamps[idx]]
            src_camera = self.create_camera_vector(src_rgb,
                                                   unnormalize_intrinsics(cam_param.intrinsics, self.target_h, self.target_w),
                                                   cam_param.c2w_mat)

            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs)
        src_cameras = np.stack(src_cameras)

        return self.create_batch_from_numpy(rgb, camera, rgb_file, src_rgbs, src_cameras, depth_range)
