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


import math
from functools import reduce
from pathlib import Path
import configargparse
import sys

from configs.local_setting import OUT_DIR, TRAIN_CONFIG, ROOT_DIR
from nan.utils.io_utils import tuple_str, get_latest_file

if sys.gettrace() is None:
    DEBUG = False
else:
    DEBUG = True


def arg2expname(name, value):
    if isinstance(value, bool):
        return f"__{name}" if value else ""
    elif value is None:
        return ""
    else:
        return f"__{name}_{value}"


def std_str(std):
    if isinstance(std, (list, tuple)):
        if std == [-3, -1.5, -2, -1]:
            return "__low_noise"
        elif std == [-2.25, -0.5, -1.4, -0.5]:
            return "__high_noise"
        elif std == [-3, -0.5, -2, -0.5]:
            return "__full"
        return f"__std_{tuple_str(std)}"
    else:
        return "__clean"


def kernel_size_str(args):
    kernel_size = args.kernel_size
    rgb_weights = args.rgb_weights
    if kernel_size != (1, 1):
        if rgb_weights:
            return f"__{tuple_str(kernel_size)}_3"
        else:
            return f"__{tuple_str(kernel_size)}"
    else:
        return ""


def loss_str(args):
    losses = list(filter(lambda l: l[1] > 0, zip(args.losses, args.losses_weights)))

    def single_loss_str(l):
        if l[1] == 1:
            return f"{l[0]}"
        else:
            return f"{l[0]}_{l[1]}"

    return reduce(lambda a, b: f"{a}__{single_loss_str(b)}", losses, "")


class CustomArgumentParser(configargparse.ArgumentParser):
    @classmethod
    def config_parser(cls):
        parser = cls()
        # general
        parser.add_argument('--config', is_config_file=True,
                            help='config file path')
        parser.add_argument("--expname", type=str,
                            help='experiment name', default='')
        parser.add_argument('--distributed', action='store_true',
                            help='if use distributed training')
        parser.add_argument("--local_rank", type=int, default=0,
                            help='rank for distributed training')
        parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                            help='number of data loading workers (default: 8)')

        # ########## dataset options ##########
        # ## train and eval dataset
        parser.add_argument("--train_dataset", type=str, default='ibrnet_collected',
                            help='the training dataset, should either be a single dataset, '
                                 'or multiple datasets connected with "+", for example, ibrnet_collected+llff+spaces')
        parser.add_argument("--dataset_weights", nargs='+', type=float, default=[],
                            help='the weights for training datasets, valid when multiple datasets are used.')
        parser.add_argument("--train_scenes", nargs='+', default=[],
                            help='optional, specify a subset of training scenes from training dataset')
        parser.add_argument('--eval_dataset', type=str, default='llff_test',
                            help='the dataset to evaluate')
        parser.add_argument('--eval_scenes', nargs='+', default=[],
                            help='optional, specify a subset of scenes from eval_dataset to evaluate')
        # ## others
        parser.add_argument("--testskip", type=int, default=8, help='will load 1/N images from test/val sets, '
                                                                    'useful for large datasets like deepvoxels or '
                                                                    'nerf_synthetic')

        # ########## model options ##########
        # ## ray sampling options
        parser.add_argument('--sample_mode', type=str, default='uniform',
                            help='how to sample pixels from images for training: uniform|center')
        parser.add_argument('--center_ratio', type=float, default=0.8,
                            help='the ratio of center crop to keep')
        parser.add_argument("--N_rand", type=int, default=32 * 16,
                            help='batch size (number of random rays per gradient step)')
        parser.add_argument("--chunk_size", type=int, default=1024 * 4,
                            help='number of rays processed in parallel (in testing), decrease if running out of memory')

        # ## model options
        parser.add_argument('--coarse_feat_dim', type=int, default=32,
                            help="2D feature dimension for coarse level")
        parser.add_argument('--fine_feat_dim', type=int, default=32,
                            help="2D feature dimension for fine level")
        parser.add_argument('--num_source_views', type=int, default=8,
                            help='the number of input source views for each target view')
        parser.add_argument('--coarse_only', action='store_true',
                            help='use coarse network only')
        parser.add_argument("--anti_alias_pooling", type=int, default=1,
                            help='if use anti-alias pooling')
        parser.add_argument("--kernel_size", nargs=2, type=int, default=None,
                            help='determine the size of the RGB blending kernels')
        parser.add_argument("--rgb_weights", action='store_true', help='whether the blending kernels are different '
                                                                       'for each color channel')
        parser.add_argument("--views_attn", action='store_true',
                            help='determine whether t o use the views attention or not')
        parser.add_argument("--pre_net", action='store_true',
                            help='whether to add the convolutional layer before the features net')
        parser.add_argument("--noise_feat", action='store_true',
                            help='whether to use the noise parameters')

        # ########## checkpoints ##########
        parser.add_argument("--no_reload", action='store_true',
                            help='do not reload weights from saved ckpt')
        parser.add_argument("--resume_training", action='store_true', help='continue training from latest ckpt')
        parser.add_argument("--ckpt_path", type=Path, default=None,
                            help='specific weights npy file to reload for network')
        parser.add_argument("--no_load_opt", action='store_true',
                            help='do not load optimizer when reloading')
        parser.add_argument("--no_load_scheduler", action='store_true',
                            help='do not load scheduler when reloading')
        parser.add_argument("--allow_weights_mismatch", action="store_true",
                            help='allow mismatch between loaded ckpt and model weights shape. '
                                 'Only agreed dimension will be loaded')

        # ########### iterations, loss & learning rate options ##########
        parser.add_argument("--n_iters", type=int, default=250000,
                            help='num of total iterations')
        parser.add_argument("--lrate_feature", type=float, default=1e-3,
                            help='learning rate for feature extractor')
        parser.add_argument("--lrate_mlp", type=float, default=5e-4,
                            help='learning rate for mlp')
        parser.add_argument("--lrate_decay_factor", type=float, default=0.5,
                            help='decay learning rate by a factor every specified number of steps')
        parser.add_argument("--lrate_decay_steps", type=int, default=50000,
                            help='decay learning rate by a factor every specified number of steps')
        parser.add_argument("--losses", nargs='+', type=str, default=['l1'],
                            help='list of losses to apply')
        parser.add_argument("--losses_weights", nargs='+', type=float, default=[1],
                            help='list of weights for the losses')
        parser.add_argument("--process_loss", action='store_true',
                            help='whether to apply the loss on the post processed prediction (white balance and gamma '
                                 'correction). I trained the paper results without processing, but maybe it is better '
                                 'to do this')

        # ########## rendering options ##########
        parser.add_argument("--N_samples", type=int, default=64,
                            help='number of coarse samples per ray')
        parser.add_argument("--N_importance", type=int, default=64,
                            help='number of important samples per ray (additional samples during the fine phase)')
        parser.add_argument("--inv_uniform", action='store_true',
                            help='if True, will uniformly sample inverse depths')
        parser.add_argument("--det", action='store_true',
                            help='deterministic sampling for coarse and fine samples')
        parser.add_argument("--white_bkgd", action='store_true',
                            help='apply the trick to avoid fitting to white background')
        parser.add_argument("--render_stride", type=int, default=1,
                            help='render with large stride for validation to save time')

        # ########## logging/saving options ##########
        parser.add_argument("--i_print", type=int, default=100,
                            help='frequency of terminal printout')
        parser.add_argument("--i_tb", type=int, default=20,
                            help='frequency of tensorboard logging (metrics)')
        parser.add_argument("--i_img", type=int, default=500,
                            help='frequency of tensorboard image logging')
        parser.add_argument("--i_weights", type=int, default=10000,
                            help='frequency of weight ckpt saving')

        # ########## evaluation options ##########
        parser.add_argument("--llffhold", type=int, default=8,
                            help='will take every 1/N images as LLFF test set, paper uses 8')
        parser.add_argument("--factor", type=int, default=4,
                            help='resolution of evaluated images. Default is 4. For LLFF dataset it means 1108x756')
        parser.add_argument("--same", action='store_true',
                            help='for testing: whether to load same config as in training. differ_from_training_args '
                                 'is than used for changing specific args.')
        parser.add_argument("--process_output", action='store_true',
                            help='whether to save processed output (white balance and gamma correction)')
        parser.add_argument("--eval_images", action='store_true', help='whether to evaluate the whole images')
        parser.add_argument("--eval_rays", action='store_true', help='whether to evaluate specific rays')
        parser.add_argument("--rerun", action='store_true', help='whether to rerun inference again of just calculate metrics')
        parser.add_argument("--post", type=str, default='', help='suffix of the images to load when rerun=False')

        # ### burst denoising simulation and training ###
        parser.add_argument("--std", nargs='+', type=float, default=[0],
                            help='noise parameters for generating simulation. This is the log10 of the std limits, '
                                 'used to generate the noise in training in'
                                 'nan.dataloaders.basic_dataset.NoiseDataset.get_noise_params_train')

        parser.add_argument("--eval_gain", type=int, default=None,
                            help='gain to apply in evaluation')
        parser.add_argument("--include_target", action='store_true',
                            help='whether to include the target image in the input to the algorithm (burst denoising '
                                 'task) or not (novel view synthesis)')
        parser.add_argument("--sup_clean", action='store_true', help='apply the loss against the ground truth. '
                                                                     'Surprisingly, the network can also learn by '
                                                                     'compare the prediction to the noisy input '
                                                                     'sample, which is what RawNeRF is doing.')

        return parser

    def parse_args(self, verbose=False, **kwargs):
        args = super().parse_args(**kwargs)

        # Arranging some arguments
        # ckpt_path
        if args.ckpt_path is not None and not args.ckpt_path.is_absolute():
            print(f"[*] changing args.ckpt_path={str(args.ckpt_path)} to absolute path {ROOT_DIR / args.ckpt_path}")
            args.ckpt_path = ROOT_DIR / args.ckpt_path

        # losses
        assert len(args.losses) == len(args.losses_weights)
        loss_dict = {loss: w for loss, w in zip(args.losses, args.losses_weights) if w > 0}
        if verbose:
            print(f"[*] losses: {loss_dict}")
        if 'ssim' in loss_dict:
            if verbose:
                print(f"[*] changing sample mode from {args.sample_mode=} to 'crop'")
                print(f"[*] changing N_rand from {args.N_rand=} to {math.ceil(args.N_rand ** 0.5) ** 2} for patch loss")
            args.N_rand = math.ceil(args.N_rand ** 0.5) ** 2
            # This is not exactly the value that will be used, since it change in nan.trainer.Trainer.training_loop to
            # N_rand = int(1.0 * self.args.N_rand * self.args.num_source_views / train_data['src_rgbs'][0].shape[0])
            args.sample_mode = 'crop'

        # kernel size
        if args.kernel_size is not None:
            args.kernel_size = tuple(args.kernel_size)
        else:
            args.kernel_size = (1, 1)

        if len(args.std) == 1:
            args.std = args.std[0]

        self.update_expname(args)

        return args

    def update_expname(self, args):
        # for testing with same training parameters
        if args.same:
            args.expname += "same"
        else:
            if args.expname:
                args.expname += '__'
            args.expname += self.get_expname(args)

    @classmethod
    def get_expname(cls, args):
        name = f"{'tar' if args.include_target else ''}" \
               f"{std_str(args.std)}" \
               f"{'__pre' if args.pre_net else ''}" \
               f"{kernel_size_str(args)}" \
               f"{arg2expname('views_attn', args.views_attn)}" \
               f"{arg2expname('noise_feat', args.noise_feat)}" \
               f"{loss_str(args)}" \
               f"{'__DEBUG' if DEBUG else ''}"

        if name.endswith('tar__full__pre__3_3_3__views_attn__noise_feat__l1'):
            return 'NAN'
        if name.endswith('tar__full__pre__3_3_3__views_attn__noise_feat__l1__DEBUG'):
            return 'NAN__DEBUG'
        else:
            return name


if __name__ == '__main__':
    sys.argv = sys.argv + ['--config', str(TRAIN_CONFIG)]
    test_parser = CustomArgumentParser.config_parser()
    train_args = test_parser.parse_args(verbose=True)
    print(train_args.expname)
