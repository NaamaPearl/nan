import sys
from copy import copy
from functools import reduce
from torch.utils.data import DataLoader

from configs.config import CustomArgumentParser
from configs.local_setting import LOG_DIR
from nan.dataloaders import NoiseDataset, dataset_dict
from nan.dataloaders.basic_dataset import Mode
from nan.model import NANScheme
from nan.utils.io_utils import print_link, open_file_explorer


def rearrange_args_for_eval(additional_eval_args, differ_from_train_args):
    if differ_from_train_args is None:
        differ_from_train_args = []
    if additional_eval_args is None:
        additional_eval_args = []

    # creating args based on eval config
    parser = CustomArgumentParser.config_parser()
    eval_args = parser.parse_args(args=sys.argv[1:] + additional_eval_args, verbose=False)
    eval_args.distributed = False

    # creating args based on config from training
    if not (eval_args.ckpt_path.parent / "config.yml").exists():
        raise FileNotFoundError(f"config file: {(eval_args.ckpt_path.parent / 'config.yml').absolute()}")
    curr_ckpt_train_args = ['--config', str(eval_args.ckpt_path.parent / "config.yml"),
                            '--ckpt_path', str(eval_args.ckpt_path)]
    if eval_args.force_latest_exp:
        curr_ckpt_train_args.append('--force_latest_exp')  # TODO need to check that force_latest_exp works
    curr_ckpt_train_args = parser.parse_args(args=curr_ckpt_train_args, verbose=False)
    curr_ckpt_train_args.no_reload = False  # make sure to reload the ckpt weights
    curr_ckpt_train_args.local_rank = eval_args.local_rank

    # Evaluate in same configuration as in training
    if eval_args.same:
        # Copy training args to eval args, but update eval only parameters
        new_eval_args = copy(curr_ckpt_train_args)
        new_eval_args.chunk_size = eval_args.chunk_size
        new_eval_args.render_stride = eval_args.render_stride
        new_eval_args.eval_dataset = eval_args.eval_dataset
        new_eval_args.eval_scenes = eval_args.eval_scenes
        new_eval_args.distributed = eval_args.distributed
        new_eval_args.render_stride = eval_args.render_stride
        new_eval_args.expname = eval_args.expname
        new_eval_args.eval_gain = eval_args.eval_gain
        new_eval_args.no_reload = eval_args.no_reload
        new_eval_args.process_output = eval_args.process_output
        new_eval_args.local_rank = eval_args.local_rank

        # Change the specific args from training
        for arg_name, arg_val in differ_from_train_args:
            new_eval_args.__setattr__(arg_name, arg_val)
            if type(arg_val) == bool:
                new_eval_args.expname += f"__{arg_name}" if arg_val else ''
            elif type(arg_val) in [tuple, list]:
                new_eval_args.expname += f"__{arg_name}_{arg_val[0]}_{arg_val[1]}"
            else:
                new_eval_args.expname += f"__{arg_name}_{arg_val}"
        eval_args = new_eval_args

    return curr_ckpt_train_args, eval_args


# TODO Naama rearrange
def init_eval(additional_eval_args=None, open_dir=True, differ_from_train_args=None):
    train_args, eval_args = rearrange_args_for_eval(additional_eval_args, differ_from_train_args)

    # Create NAN model
    if eval_args.same:
        model = NANScheme(train_args)
    else:
        model = NANScheme(eval_args)

    # dataloader
    test_dataset: NoiseDataset = dataset_dict[eval_args.eval_dataset](eval_args, Mode.test, scenes=eval_args.eval_scenes)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # scene
    assert len(eval_args.eval_scenes) == 1, "only accept single scene"
    scene_name = eval_args.eval_scenes[0]

    # depth range, model step
    if train_args.expname in ['deeprep', 'bpn']:
        test_dataset.depth_range = 0
        model.start_step = 0
        depth_range_str = "0"
    else:
        depth_range_str = f"{reduce(lambda i, j: f'{i:.3f}_{j:.3f}', test_dataset.depth_range)}"

    # create dirs
    base_dir = LOG_DIR / train_args.expname / eval_args.eval_dataset
    if eval_args.eval_dataset == 'usfm':
        res_dir = base_dir / f"{scene_name}_{model.start_step:06d}" / depth_range_str / eval_args.expname
    else:
        res_dir = base_dir / eval_args.expname / f"{scene_name}_{model.start_step:06d}" / depth_range_str
    res_dir.mkdir(exist_ok=True, parents=True)

    # printing..
    print_link(res_dir, "saving results to", '...')
    if open_dir:
        open_file_explorer(res_dir)

    return test_loader, scene_name, res_dir, eval_args, model