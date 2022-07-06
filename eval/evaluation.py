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

from eval.evaluate_images import evaluate_images
from eval.evaluate_rays import evaluate_rays
from eval.experiment_list import LLFF_SCENES_LIST


def eval_multi_scenes(ckpt=None, differ_from_train_args=None, scene_list=None, rerun=True, post='', images=True, rays=True):
    if scene_list is None:
        scene_list = LLFF_SCENES_LIST
    if differ_from_train_args is None:
        differ_from_train_args = []
    for scene in scene_list:
        add_args = ['--eval_scenes', scene]
        if ckpt is not None:
            add_args += ['--ckpt_path', str(ckpt)]

        if images:
            print("********** evaluate images ***********")
            evaluate_images(add_args, differ_from_train_args, rerun, post=post)
        if rays:
            print("********** evaluate rays   ***********")
            evaluate_rays(add_args, differ_from_train_args)
