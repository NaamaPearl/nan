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

# #### Modified version
# #### see https://github.com/googleinterns/IBRNet for original

import sys
import numpy as np
import torch.utils.data.distributed

from configs.config import CustomArgumentParser
from configs.experiment_setting import DEFAULT_GAIN_LIST
from configs.local_setting import EVAL_CONFIG, TRAIN_CONFIG
from eval.evaluate import eval_multi_scenes
from eval.summary_BD import summary_multi_gains
from nan.trainer import Trainer
import torch.distributed as dist

from nan.utils.io_utils import print_link


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def main():
    print("\n")
    print("************************************************************")
    print_link(TRAIN_CONFIG, "Start training from config file: ")
    print("************************************************************")
    print("\n")

    # Create training args
    parser = CustomArgumentParser.config_parser()
    train_args = parser.parse_args(verbose=True)

    for key, value in sorted(vars(train_args).items()):
        print(f"{key:<30}: {value}")

    # Set distributed options
    if train_args.distributed:
        torch.cuda.set_device(train_args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # Call to train and return the last saved checkpoint
    trainer = Trainer(train_args)
    last_ckpt = trainer.train()

    # Evaluation of last ckpt saved using the default evaluation config EVAL_CONFIG
    sys.argv = sys.argv[:1] + ['--config', str(EVAL_CONFIG)]
    if last_ckpt is not None:
        # Run evaluation on each gain level (defined as in KPN paper)
        for gain in DEFAULT_GAIN_LIST:
            # To use the same configuration as in the training config set args.same = True (set in the default config)
            # Additional args which differ from the training scheme should be set here. (factor is the image resolution)
            # init_eval.py create the evaluation params from the training params
            eval_additional_args = [('factor', 4), ('eval_gain', gain)]
            eval_multi_scenes(ckpt=last_ckpt, differ_from_train_args=eval_additional_args)
        summary_multi_gains({last_ckpt.parent.name: (last_ckpt.parent.name, '')})


if __name__ == '__main__':
    # Training using the default training config TRAIN_CONFIG
    sys.argv = sys.argv + ['--config', str(TRAIN_CONFIG)]
    print(sys.argv)
    main()
