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

from configs.experiment_setting import DEFAULT_GAIN_LIST
from configs.local_setting import EVAL_CONFIG
from eval.scene_evaluator import SceneEvaluator
from eval.experiment_list import LLFF_SCENES_LIST
from nan.utils.io_utils import print_link


def eval_multi_scenes(ckpt=None, differ_from_train_args=(), scene_list=LLFF_SCENES_LIST):
    for scene in scene_list:
        additional_eval_args = ['--eval_scenes', scene]
        if ckpt is not None:
            additional_eval_args += ['--ckpt_path', str(ckpt)]

        SceneEvaluator.scene_evaluation(add_args=additional_eval_args,
                                        differ_args=differ_from_train_args)


def main():
    print("\n")
    print("************************************************************")
    print_link(EVAL_CONFIG, "Start evaluation from config file: ")
    print("************************************************************")
    print("\n")

    for gain in DEFAULT_GAIN_LIST:
        print(f'{gain=}')
        differ_from_train_args = [('factor', 4), ('eval_gain', gain), ('num_source_views', 8)]
        eval_multi_scenes(differ_from_train_args=differ_from_train_args)


if __name__ == '__main__':
    sys.argv += ['--config', str(EVAL_CONFIG)]
    main()
