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

import numpy as np
import torch
import torch.nn.functional as F

rng = np.random.RandomState(234)

########################################################################################################################
# ray batch sampling
########################################################################################################################


def parse_camera(params):
    """
    Parse camera vector (34) to H, W, K, R|t
    :param params:  camera vector (34)
    :return: W, H, K (Bx4x4), R|t (Bx4x4)
    """
    H = params[:, 0]
    W = params[:, 1]
    intrinsics = params[:, 2:18].reshape((-1, 4, 4))
    c2w = params[:, 18:34].reshape((-1, 4, 4))
    return W, H, intrinsics, c2w


class RaySampler:
    def __init__(self, data, device, resize_factor=1, render_stride=1):
        """
        Object that handle the sampling of the rays from the target images (pick the pixels to generate rays from).
        Only batch size 1 is supported (B = 1)
        :param data: dict {camera: (B, 34),
                                 src_rgbs_clean: (B, N, H, W, 3),
                                 src_rgbs: (B, N, H, W, 3),
                                 src_cameras: (B, N, 34),
                                 depth_range: (1, 2),
                                 sigma_estimate: (B, N, H, W, 3),
                                 white_level: (1, 1),
                                 rgb_clean: (B, H, W, 3), rgb: (B, H, W, 3),
                                 gt_depth: ,
                                 rgb_path: list(B)}
        :param device:
        :param resize_factor: float
        :param render_stride: int
        """
        super().__init__()
        self.render_stride                  = render_stride
        self.rgb_path                       = data['rgb_path']
        self.depth_range                    = data['depth_range']
        self.white_level                    = data['white_level']
        self.device                         = device

        self.rgb                            = data['rgb'] if 'rgb' in data.keys() else None
        self.src_rgbs                       = data['src_rgbs'] if 'src_rgbs' in data.keys() else None
        self.rgb_clean                      = data['rgb_clean'] if 'rgb_clean' in data.keys() else None
        self.src_rgbs_clean                 = data['src_rgbs_clean'] if 'src_rgbs_clean' in data.keys() else None

        self.sigma_estimate                 = data['sigma_estimate'] if 'sigma_estimate' in data.keys() else None

        self.camera                         = data['camera']
        self.src_cameras                    = data['src_cameras'] if 'src_cameras' in data.keys() else None

        W, H, self.intrinsics, self.c2w_mat = parse_camera(self.camera)
        self.batch_size                     = len(self.camera)

        assert H.shape[0] == 1
        self.H = int(H[0])
        self.W = int(W[0])

        # half-resolution output
        if resize_factor != 1:
            self.W = int(self.W * resize_factor)
            self.H = int(self.H * resize_factor)
            self.intrinsics[:, :2, :3] *= resize_factor
            if self.rgb is not None:
                self.rgb       = F.interpolate(self.rgb.permute(0, 3, 1, 2),
                                               scale_factor=resize_factor).permute(0, 2, 3, 1)
            if self.rgb_clean is not None:
                self.rgb_clean = F.interpolate(self.rgb_clean.permute(0, 3, 1, 2),
                                               scale_factor=resize_factor).permute(0, 2, 3, 1)

        self.rays_o, self.rays_d, self.xy = self.generate_all_rays(self.H, self.W, self.intrinsics, self.c2w_mat)

        if self.rgb is not None:
            self.rgb = self.rgb.reshape(-1, 3)

        if self.rgb_clean is not None:
            self.rgb_clean = self.rgb_clean.reshape(-1, 3)

    def generate_all_rays(self, H, W, intrinsics, c2w):
        """
        :param H: image height
        :param W: image width
        :param intrinsics: 4 by 4 intrinsic matrix
        :param c2w: 4 by 4 camera to world extrinsic matrix
        :return: all rays of a target image: rays origin (H*W, 3), rays direction (H*W, 3), pixels coordinate + (1,) (H*W, 3)
        """
        x, y = np.meshgrid(np.arange(W)[::self.render_stride], np.arange(H)[::self.render_stride])
        x = x.reshape(-1).astype(dtype=np.float32)  # + 0.5    # add half pixel
        y = y.reshape(-1).astype(dtype=np.float32)  # + 0.5
        pixels = np.stack((x, y, np.ones_like(x)), axis=0)  # (3, H*W)
        pixels = torch.from_numpy(pixels).to(intrinsics.device)
        batched_pixels = pixels.unsqueeze(0).expand(self.batch_size, 3, pixels.shape[-1])

        rays_d = (c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)).transpose(1, 2)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = c2w[:, :3, 3].unsqueeze(1).expand(1, rays_d.shape[0], 3).reshape(-1, 3)  # B x HW x 3
        batched_pixels = batched_pixels.transpose(1, 2).reshape(-1, 3)
        return rays_o, rays_d, batched_pixels.to(int)

    def get_all_rays_batch(self, device):
        ret = {'ray_o': self.rays_o.to(device),
               'ray_d': self.rays_d.to(device),
               'depth_range': self.depth_range.to(device),
               'camera': self.camera.to(device),
               'rgb': self.rgb.to(device) if self.rgb is not None else None,
               'rgb_clean': self.rgb_clean.to(device) if self.rgb_clean is not None else None,
               'src_rgbs': self.src_rgbs.to(device) if self.src_rgbs is not None else None,
               'src_rgbs_clean': self.src_rgbs_clean.to(device) if self.src_rgbs_clean is not None else None,
               'src_cameras': self.src_cameras.to(device) if self.src_cameras is not None else None,
               'xyz': self.xy.to(device)
        }
        return ret

    def sample_random_pixels(self, N_rand, sample_mode, center_ratio=0.8):
        if sample_mode == 'center':
            border_H = int(self.H * (1 - center_ratio) / 2.)
            border_W = int(self.W * (1 - center_ratio) / 2.)

            # pixel coordinates
            u, v = np.meshgrid(np.arange(border_H, self.H - border_H),
                               np.arange(border_W, self.W - border_W))
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = rng.choice(u.shape[0], size=(N_rand,), replace=False)
            select_inds = v[select_inds] + self.W * u[select_inds]

        elif sample_mode == 'uniform':
            # Random from one image
            select_inds = rng.choice(self.H*self.W, size=(N_rand,), replace=False)
        elif sample_mode == 'crop':
            crop_size = int(N_rand ** 0.5)
            top  = torch.randint(0, self.H - crop_size + 1, size=(1, )).item()
            left = torch.randint(0, self.W - crop_size + 1, size=(1, )).item()

            # pixel coordinates
            u, v = np.meshgrid(np.arange(self.H), np.arange(self.W))
            u = u[..., left:left+crop_size, top:top+crop_size].reshape(-1)
            v = v[..., left:left+crop_size, top:top+crop_size].reshape(-1)
            select_inds = v + self.W * u
        else:
            raise Exception("unknown sample mode!")

        return select_inds

    def random_ray_batch(self, N_rand, sample_mode, center_ratio=0.8, clean=False):
        """
        Select random N_rand rays.
        :param N_rand: number of rays in a batch (R)
        :param sample_mode: 'center', 'uniform', 'crop'
        :param center_ratio: arguments for sampling mode 'center'
        :param clean: if set to True, the rgb values in the rays batch will be clean,
                      otherwise it will have the noisy values.
                      This is determined by args.sup_clean.
                      It effects whether the loss function is WRT the clean values (supervied),
                       or WRT noisy values (unsupervised) (as in NeRF in the Dark)

        :return: batch of rays and more relevant data for the batch , Dict {ray_o: (R, 3),
                                                                           ray_d: (R, 3),
                                                                           camera: (1, 34),
                                                                           depth_range: (1, 2),
                                                                           src_cameras: (1, N, 34),
                                                                           selected_inds: (R,),
                                                                           xyz: (R, 3),
                                                                           rgb: (R, 3),
                                                                           white_level: (1, 1)}
        """

        select_inds = self.sample_random_pixels(N_rand, sample_mode, center_ratio)

        return self.specific_ray_batch(select_inds, clean=clean)

    def specific_ray_batch(self, select_inds, clean=False):
        """
        Select specific N_rand rays by select_inds
        :param select_inds: R indices of rays to choose
        :param clean: if set to True, the rgb values in the rays batch will be clean,
                      otherwise it will have the noisy values.
                      This is detemined by args.sup_clean.
                      It effect whether the loss function is WRT the clean values (supervied),
                       or WRT noisy values (unsupervies) (as in NeRF in the Dark)
        :return: batch of rays and more relevant data for the batch. Dict {ray_o: (R, 3),
                                                                           ray_d: (R, 3),
                                                                           camera: (1, 34),
                                                                           depth_range: (1, 2),
                                                                           src_cameras: (1, N, 34),
                                                                           selected_inds: (R,),
                                                                           xyz: (R, 3),
                                                                           rgb: (R, 3),
                                                                           white_level: (1, 1)}
        """
        rgb_inds = select_inds
        if clean:
            rgb = self.rgb_clean[rgb_inds, :].to(self.device) if self.rgb_clean is not None else None
        else:
            rgb = self.rgb[rgb_inds, :].to(self.device) if self.rgb is not None else None

        return {'ray_o'          : self.rays_o[select_inds].to(self.device),
                'ray_d'          : self.rays_d[select_inds].to(self.device),
                'camera'         : self.camera.to(self.device),
                'depth_range'    : self.depth_range.to(self.device),
                'src_cameras'    : self.src_cameras.to(self.device) if self.src_cameras is not None else None,
                'selected_inds'  : select_inds,
                'xyz'            : self.xy[select_inds],
                'rgb'            : rgb,
                'white_level'    : self.white_level.to(self.device)}

    def sample_ray_batch_from_pixel(self, pixel, clean=False):
        from random import shuffle
        select_inds = [x + self.W * y for y, x in pixel]
        shuffle(select_inds)
        return self.specific_ray_batch(select_inds, clean=clean)
