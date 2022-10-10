from abc import ABC
from copy import copy
from pathlib import Path
import torch
from torch.utils.data import Dataset
from nan.dataloaders.data_utils import random_crop, random_flip
from configs.local_setting import DATA_DIR
import imageio
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np


class Mode(Enum):
    train = "train"
    validation = "validation"
    test = "test"


# From DeepRep
# https://github.com/goutamgmb/deep-rep/blob/master/data/postprocessing_functions.py#L73

t = 0.0031308
gamma = 2.4
a = 1. / (1. / (t ** (1 / gamma) * (1. - (1 / gamma))) - 1.)  # 0.055
# a = 0.055
k0 = (1 + a) * (1 / gamma) * t ** ((1 / gamma) - 1.)  # 12.92
# k0 = 12.92
inv_t = t * k0


def de_linearize(rgb, wl=1.):
    """
    Process the RGB values in the inverse process of the approximate linearization, in a differential format
    @param rgb:
    @param wl:
    @return:
    """
    rgb = rgb / wl
    srgb = torch.where(rgb > t, (1 + a) * torch.clamp(rgb, min=t) ** (1 / gamma) - a, k0 * rgb)

    k1 = (1 + a) * (1 / gamma)
    srgb = torch.where(rgb > 1, k1 * rgb - k1 + 1, srgb)
    return srgb


def de_linearize_np(rgb, wl=1.):
    rgb = rgb / wl
    srgb = np.where(rgb > t, (1 + a) * np.clip(rgb, a_min=t, a_max=np.inf) ** (1 / gamma) - a, k0 * rgb)

    # From deep-rep/data/postprocessing_functions.py  DenoisingPostProcess
    k1 = (1 + a) * (1 / gamma)
    srgb = np.where(rgb > 1, k1 * rgb - k1 + 1, srgb)
    return srgb


def re_linearize(rgb, wl=1.):
    """
    Approximate re-linearization of RGB values by revert gamma correction and apply white level
    Revert gamma correction
    @param rgb:
    @param wl:
    @return:
    """
    # return rgb
    return wl * (rgb ** 2.2)
    # degamma = torch.where(rgb > inv_t, ((torch.clamp(rgb, min=inv_t) + a) / (1 + a)) ** gamma, rgb / k0)


class BurstDataset(Dataset, ABC):
    @property
    def dir_name(self) -> str:
        raise NotImplementedError

    @property
    def folder_path(self) -> Path:
        return DATA_DIR / self.dir_name

    def __init__(self, args, mode: Mode, scenes=(), random_crop=True):
        assert type(Mode.train) is Mode
        self.args = copy(args)
        self.mode = mode
        self.num_source_views = args.num_source_views
        self.random_crop = random_crop
        self.scenes_dirs = self.pick_scenes(scenes)

        if len(self.scenes_dirs) == 1:
            print(f"loading {self.scenes_dirs[0].stem} scenes for {mode}")
        else:
            print(f"loading {len(self.scenes_dirs)} scenes for {mode}")
        print(f"num of source views {self.num_source_views}")

        for i, scene_path in enumerate(self.scenes_dirs):
            self.add_single_scene(i, scene_path)

    def pick_scenes(self, scenes):
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
            return self.get_specific_scenes(scenes)
        else:
            return self.get_all_scenes()

    def get_specific_scenes(self, scenes):
        return [self.folder_path / scene for scene in scenes]

    def get_all_scenes(self):
        return self.listdir(self.folder_path)

    @staticmethod
    def listdir(folder):
        return list(folder.glob("*"))

    @staticmethod
    def read_image(filename, **kwargs):
        return imageio.imread(filename).astype(np.float32) / 255.

    def apply_transform(self, rgb, camera, src_rgbs, src_cameras):
        if self.mode == Mode.train and self.random_crop:
            rgb, camera, src_rgbs, src_cameras = random_crop(rgb, camera, src_rgbs, src_cameras)

        if self.mode == Mode.train and np.random.choice([0, 1]):
            rgb, camera, src_rgbs, src_cameras = random_flip(rgb, camera, src_rgbs, src_cameras)

        return rgb, camera, src_rgbs, src_cameras

    def final_depth_range(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def create_camera_vector(rgb, intrinsics, pose):
        """
        Creating camera representative vector (used by IBRNet)
        :param rgb: (H, W, 3)
        :param intrinsics: (4, 4)
        :param pose: (4, 4)
        :return: camera vector (34) (H, W, K.flatten(), (R|t).flatten())
        """
        return np.concatenate((rgb.shape[:2], intrinsics.flatten(), pose.flatten())).astype(np.float32)


class NoiseDataset(BurstDataset, ABC):
    def __init__(self, args, mode, **kwargs):
        super().__init__(args, mode, **kwargs)
        if mode == Mode.train:
            assert len(self.args.std) == 4
            self.get_noise_params = self.get_noise_params_train
        else:
            if self.args.eval_gain == 0:
                sig_read, sig_shot = 0, 0
                print(f"Loading {mode} set without additional noise.")
            else:
                # load gain data from KPN paper https://bmild.github.io/kpn/index.html
                noise_data = np.load(DATA_DIR / 'synthetic_5d_j2_16_noiselevels6_wide_438x202x320x8.npz')
                sig_read_list = np.unique(noise_data['sig_read'])[2:]
                sig_shot_list = np.unique(noise_data['sig_shot'])[2:]

                log_sig_read = np.log10(sig_read_list)
                log_sig_shot = np.log10(sig_shot_list)

                d_read = np.diff(log_sig_read)[0]
                d_shot = np.diff(log_sig_shot)[0]

                gain_log = np.log2(self.args.eval_gain)

                sig_read = 10 ** (log_sig_read[0] + d_read * gain_log)
                sig_shot = 10 ** (log_sig_shot[0] + d_shot * gain_log)

                print(f"Loading {mode} set for gain {self.args.eval_gain}. "  
                      f"Max std {self.get_std(1, sig_read, sig_shot)}")

            def get_noise_params_test():
                return sig_read, sig_shot

            self.get_noise_params = get_noise_params_test
        self.depth_range = None

    def choose_views(self, possible_views, num_views, target_view):
        if self.mode == Mode.train:
            chosen_ids = np.random.choice(possible_views, min(num_views, len(possible_views)), replace=False)
        else:
            chosen_ids = possible_views[:min(num_views, len(possible_views))]

        assert target_view not in chosen_ids

        if self.args.include_target:
            # always include input image in the first idx - denoising task
            chosen_ids[0] = target_view
        else:
            # occasionally include input image in random index
            if np.random.choice([False, True], p=[0.995, 0.005]) and self.mode == Mode.train:
                chosen_ids[np.random.choice(len(chosen_ids))] = target_view

        return chosen_ids

    def get_noise_params_train(self):
        sigma_read_lim = self.args.std[:2]
        sigma_shot_lim = self.args.std[2:]

        sigma_read_log = np.random.default_rng().uniform(low=sigma_read_lim[0], high=sigma_read_lim[1], size=1).item()
        sigma_shot_log = np.random.default_rng().uniform(low=sigma_shot_lim[0], high=sigma_shot_lim[1], size=1).item()

        sigma_read = 10 ** sigma_read_log
        sigma_shot = 10 ** sigma_shot_log

        return sigma_read, sigma_shot

    @classmethod
    def get_std(cls, rgb, sig_read, sig_shot):
        return (sig_read ** 2 + sig_shot ** 2 * rgb) ** 0.5

    def add_noise(self, rgb):
        sig_read, sig_shot = self.get_noise_params()
        std = self.get_std(rgb, sig_read, sig_shot)
        noise = std * torch.randn_like(rgb)
        noise_rgb = rgb + noise
        sigma_estimate = self.get_std(noise_rgb.clamp(0, 1), sig_read, sig_shot)
        return noise_rgb, sigma_estimate

    def create_batch_from_numpy(self, rgb_clean, camera, rgb_file, src_rgbs_clean, src_cameras, depth_range,
                                gt_depth=None):
        if self.mode in [Mode.train, Mode.validation]:
            white_level = 10 ** -torch.rand(1)
        else:
            white_level = torch.Tensor([1])

        if rgb_clean is not None:
            rgb_clean = re_linearize(torch.from_numpy(rgb_clean[..., :3]), white_level)
            rgb, _ = self.add_noise(rgb_clean)
        else:
            rgb = None
        src_rgbs_clean = re_linearize(torch.from_numpy(src_rgbs_clean[..., :3]), white_level)
        src_rgbs, sigma_est = self.add_noise(src_rgbs_clean)

        batch_dict = {'camera'        : torch.from_numpy(camera),
                      'rgb_path'      : str(rgb_file),
                      'src_rgbs_clean': src_rgbs_clean,
                      'src_rgbs'      : src_rgbs,
                      'src_cameras'   : torch.from_numpy(src_cameras),
                      'depth_range'   : depth_range,
                      'sigma_estimate': sigma_est,
                      'white_level'   : white_level}

        if rgb_clean is not None:
            batch_dict['rgb_clean'] = rgb_clean
            batch_dict['rgb'] = rgb

        if gt_depth is not None:
            batch_dict['gt_depth'] = gt_depth

        return batch_dict


if __name__ == '__main__':
    v = torch.linspace(0, 1, 100)
    v_unproc = re_linearize(v, 1)
    v_proc = de_linearize(v_unproc, 1)

    plt.figure()
    plt.plot(v, v, label='linear')
    plt.plot(v, v_unproc, label='degamma')
    plt.plot(v, v_proc, label='gamma')
    plt.legend()
    plt.show()

    im_path = DATA_DIR / 'nerf_llff_data' / 'fern' / 'images_4' / 'image000.png'
    im = torch.from_numpy(imageio.imread(im_path) / 255)
    im = im + torch.randn_like(im) * 0.1
    print(im_path)

    plt.figure()
    plt.imshow(im.clamp(0, 1))
    plt.show()

    im_unprocessed = re_linearize(im)

    plt.figure()
    plt.imshow(im_unprocessed.clamp(0, 1))
    plt.show()

    im_processes = de_linearize(im_unprocessed)

    plt.figure()
    plt.imshow(im_processes.clamp(0, 1))
    plt.show()