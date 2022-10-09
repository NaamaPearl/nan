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

# #### Modified version of IBRNetCollectedDataset dataset code
# #### see https://github.com/googleinterns/IBRNet for original

from functools import reduce
from pathlib import Path
from typing import Tuple
import numpy as np
import torch

from configs.local_setting import DATA_DIR
from nan.dataloaders import COLMAPDataset
from nan.dataloaders.data_utils import get_nearest_pose_ids


class IBRNetCollectedDataset(COLMAPDataset):
    dir_name = ('ibrnet_collected_1', 'ibrnet_collected_2')
    num_select_high = 3
    min_nearest_pose = 22

    @property
    def folder_path(self) -> Tuple[Path, ...]:
        return tuple((DATA_DIR / name for name in self.dir_name))

    def get_specific_scenes(self, scenes):
        # in this case you should determine which are the scenes from which directory
        raise NotImplementedError

    def get_all_scenes(self):
        return reduce(lambda a, b: a + list(b.glob("*")), self.folder_path, [])

    def load_scene(self, scene_path, _):
        dir_name = scene_path.parent.stem
        if dir_name == 'ibrnet_collected_2':
            factor = 8
        else:
            factor = 2
        return super().load_scene(scene_path, self.args.factor)

    def __len__(self):
        return len(self.render_rgb_files)

    def get_nearest_pose_ids(self, render_pose, depth_range, train_poses, subsample_factor, id_render):
        mean_depth = np.mean(depth_range)
        world_center = (render_pose.dot(np.array([[0, 0, mean_depth, 1]]).T)).flatten()[:3]
        return get_nearest_pose_ids(render_pose,
                                    train_poses,
                                    min(self.num_source_views * subsample_factor, self.min_nearest_pose),
                                    tar_id=id_render,
                                    angular_dist_method='dist',
                                    scene_center=world_center)

    def final_depth_range(self, depth_range):
        return torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])
