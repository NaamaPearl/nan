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
import sys

from configs.global_setting import DEFAULT_GAIN_LIST
from configs.local_setting import EVAL_CONFIG
from eval.image_evaluator import SceneEvaluator
from eval.evaluate_rays import evaluate_rays
from eval.experiment_list import LLFF_SCENES_LIST
from nan.utils.io_utils import print_link


# TODO Naama create default evaluation config
def eval_multi_scenes(ckpt=None, differ_from_train_args=None, scene_list=None, rerun=True, post='', images=True, rays=True):
    if scene_list is None:
        scene_list = LLFF_SCENES_LIST
    if differ_from_train_args is None:
        differ_from_train_args = []
    for scene in scene_list:
        additional_eval_args = ['--eval_scenes', scene]
        if ckpt is not None:
            additional_eval_args += ['--ckpt_path', str(ckpt)]

        if images:
            print("********** evaluate images ***********")
            SceneEvaluator.scene_evaluation(additional_eval_args, differ_from_train_args, rerun, post=post)
        if rays:
            print("********** evaluate rays   ***********")
            evaluate_rays(additional_eval_args, differ_from_train_args)


def main():
    print("\n")
    print("************************************************************")
    print_link(EVAL_CONFIG, "Start evaluation from config file: ")
    print("************************************************************")
    print("\n")

    for gain in [0]:
        print(f'{gain=}')
        differ_from_train_args = [('factor', 4), ('eval_gain', gain), ('num_source_views', 8)]
        eval_multi_scenes(differ_from_train_args=differ_from_train_args, scene_list=['fern'])


if __name__ == '__main__':
    sys.argv += ['--config', str(EVAL_CONFIG)]
    main()

