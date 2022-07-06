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

from math import prod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from nan.dataloaders.data_utils import TINY_NUMBER
from nan.transformer import MultiHeadAttention

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)


# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x * weight, dim=-2, keepdim=True)
    var = torch.sum(weight * (x - mean) ** 2, dim=-2, keepdim=True)
    return mean, var


@torch.jit.script
def kernel_fused_mean_variance(x, weight):
    mean = torch.sum(x * weight, dim=(-2, -3, -4), keepdim=True)
    var = torch.sum(weight * (x - mean) ** 2, dim=(-2, -3, -4), keepdim=True)

    return mean, var


def softmax3d(x, dim):
    R, S, k, _, V, C = x.shape
    return nn.functional.softmax(x.reshape((R, S, -1, C)), dim=-2).view(x.shape) ## TODO exchange to pytorch implementation
    exp_x = torch.exp(x)
    return exp_x / (exp_x.sum(dim=dim, keepdim=True) + TINY_NUMBER)


class KernelBasis(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.basis = nn.Parameter(torch.rand(args.kernel_size + (args.basis_size,)))


class NanMLP(nn.Module):
    activation_func = nn.ELU(inplace=True)

    def __init__(self, args, in_feat_ch=32, n_samples=64):
        super().__init__()
        self.args = args
        self.device = torch.device(f"cuda:{args.local_rank}")

        assert self.args.kernel_size[0] == self.args.kernel_size[1]
        self.k_mid = int(self.args.kernel_size[0] // 2)

        self.anti_alias_pooling = args.anti_alias_pooling
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)

        self.n_samples = n_samples

        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        self.activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        self.activation_func)

        base_input_channels = (in_feat_ch + 3) * 3
        if self.args.noise_feat:
            base_input_channels += 3

        self.base_fc = nn.Sequential(nn.Linear(base_input_channels, 64),
                                     self.activation_func,
                                     nn.Linear(64, 32),
                                     self.activation_func)

        if args.views_attn:
            input_channel = 35
            self.views_attention = MultiHeadAttention(5, input_channel, 7, 8)
            # self.spatial_views_attention = MultiHeadAttention(5, input_channel, 7, 8)
            self.spatial_pos_enc = self.posenc(n_samples=self.args.kernel_size[0] ** 2, d=input_channel, s=1)
        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    self.activation_func,
                                    nn.Linear(32, 33),
                                    self.activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     self.activation_func,
                                     nn.Linear(32, 1),
                                     torch.nn.Sigmoid()
                                     )

        self.geometry_fc = nn.Sequential(nn.Linear(32 * 2 + 1, 64),
                                         self.activation_func,
                                         nn.Linear(64, 16),
                                         self.activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             self.activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        if args.bpn:
            self.basis = KernelBasis(args)
        else:
            self.basis = None

        self.rgb_fc, self.rgb_out_channels = self.rgb_fc_factory()

        self.rgb_reduce_fn = self.rgb_reduce_factory()

        self.pos_enc_d = 16

        if type(args.pos_enc_scale) is int:
            self.pos_enc_s = args.pos_enc_scale
        elif args.pos_enc_scale is None:
            self.pos_enc_s = self.n_samples - 1
        else:
            raise IOError

        if args.pos_enc == 0:
            def apply_no_pos_enc(globalfeat, _):
                return globalfeat

            self.apply_pos_enc_fn = apply_no_pos_enc

        elif args.pos_enc == 1:
            self.pos_encoding_org = self.posenc(n_samples=self.n_samples, d=self.pos_enc_d, s=self.pos_enc_s)

            def apply_org_pos_enc(globalfeat, _):
                return globalfeat + self.pos_encoding_org

            self.apply_pos_enc_fn = apply_org_pos_enc

        elif args.pos_enc == 2:
            def apply_rel_pos_enc(globalfeat, norm_z_vals):
                return globalfeat + self.pos_encoding_rel(norm_z_vals, self.pos_enc_d, self.pos_enc_s, device)

            self.apply_pos_enc_fn = apply_rel_pos_enc

        self.apply(weights_init)

    def rgb_fc_factory(self):
        if not self.args.expand_rgb:
            rgb_out_channels = 1
            rgb_fc = nn.Sequential(nn.Linear(32 + 1 + 4, 16),
                                   self.activation_func,
                                   nn.Linear(16, 8),
                                   self.activation_func,
                                   nn.Linear(8, rgb_out_channels))

        else:
            rgb_out_channels = prod(self.args.kernel_size)
            rgb_pre_out_channels = prod(self.args.kernel_size)
            if self.args.rgb_weights:
                rgb_out_channels *= 3
                rgb_pre_out_channels *= 3
            if rgb_pre_out_channels < 16:
                rgb_pre_out_channels = 16

            rgb_fc = nn.Sequential(nn.Linear(32 + 1 + 4, rgb_pre_out_channels),
                                   self.activation_func,
                                   nn.Linear(rgb_pre_out_channels, rgb_out_channels),
                                   self.activation_func,
                                   nn.Linear(rgb_out_channels, rgb_out_channels))

        return rgb_fc, rgb_out_channels

    def posenc(self, n_samples, d, s):
        position = torch.linspace(0, 1, n_samples, device=self.device).unsqueeze(0)
        return self.pos_encoding_rel(position, d, s, self.device)

    @staticmethod
    def pos_encoding_rel(position, d, s, device):
        divider = (10000 ** (2 * torch.div(torch.arange(d, device=device),
                                           2, rounding_mode='floor') / d))
        sinusoid_table = (s * position.unsqueeze(-1) / divider.unsqueeze(0))
        sinusoid_table[..., 0::2] = torch.sin(sinusoid_table[..., 0::2])  # dim 2i
        sinusoid_table[..., 1::2] = torch.cos(sinusoid_table[..., 1::2])  # dim 2i+1

        return sinusoid_table

    def forward(self, rgb_feat, ray_diff, norm_pos, mask, org_rgb, sigma_est):
        """
        @param org_rgb:
        @param sigma_est:
        @param mask:
        @param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        @param ray_diff: ray direction difference [n_rays, n_samples, k, k, n_views, 4], first 3 channels are directions,
        last channel is inner product
        @param norm_pos: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        @return: rgb and density output, [n_rays, n_samples, 4]
        """

        # [n_rays, n_samples, n_views, 3*n_feat]
        num_valid_obs = mask.sum(dim=-2)

        if self.args.blend_src:
            rgb_in = org_rgb  # [n_rays, n_samples, k, k, n_views, 4]
        else:
            rgb_in = rgb_feat[..., :3]

        ext_feat, weight = self.compute_extended_features(ray_diff, rgb_feat, mask, num_valid_obs, sigma_est)

        x = self.base_fc(ext_feat)  # ((32 + 3) x 3) --> MLP --> (32)
        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1] - 1, 1], dim=-1)
        vis = torch.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask

        sigma_out, sigma_globalfeat = self.compute_sigma(x[:, :, 0, 0],
                                                         vis[:, :, 0, 0],
                                                         norm_pos, num_valid_obs[:, :, 0, 0])
        x = torch.cat([x, vis, ray_diff], dim=-1)
        rgb_out, w_rgb, rho = self.compute_rgb(x, mask, rgb_in)
        return rgb_out, sigma_out, rho, w_rgb, rgb_in, sigma_globalfeat

    def compute_extended_features(self, ray_diff, rgb_feat, mask, num_valid_obs, sigma_est):
        direction_feat = self.ray_dir_fc(ray_diff)  # [n_rays, n_samples, k, k, n_views, 35]
        rgb_feat  = rgb_feat[:, :, self.k_mid:self.k_mid + 1, self.k_mid:self.k_mid + 1] + direction_feat  # [n_rays, n_samples, 1, 1, n_views, 35]
        feat = rgb_feat

        if self.args.views_attn:
            r, s, k, _, v, f = feat.shape
            feat, _ = self.views_attention(feat, feat, feat, (num_valid_obs > 1).unsqueeze(-1))

        if self.args.noise_feat:
            feat = torch.cat([feat, sigma_est[:, :, self.k_mid:self.k_mid + 1, self.k_mid:self.k_mid + 1] ], dim=-1)

        weight = self.compute_weights(ray_diff, mask)

        # compute mean and variance across different views for each point
        mean, var = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean, var], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]
        globalfeat = globalfeat.expand(*rgb_feat.shape[:-1], globalfeat.shape[-1])

        ext_feat = torch.cat([globalfeat, feat], dim=-1)
        return ext_feat, weight

    def compute_weights(self, ray_diff, mask):
        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)  # [n_rays, n_samples, 1, 1, n_views, 1]
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=-2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=-2, keepdim=True) + 1e-8)
        else:
            weight = mask / (torch.sum(mask, dim=-2, keepdim=True) + 1e-8)
        weight = weight / prod(self.args.kernel_size)
        return weight

    def compute_sigma(self, x, vis, norm_pos, num_valid_obs):
        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight)
        sigma_globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)],
                                     dim=-1)  # [n_rays, n_samples, 32*2+1]
        globalfeat = self.geometry_fc(sigma_globalfeat)  # [n_rays, n_samples, 16]

        # positional encoding
        globalfeat = self.apply_pos_enc_fn(globalfeat, norm_pos)

        # ray transformer
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=num_valid_obs > 1)  # [n_rays, n_samples, 16]
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

        return sigma_out, sigma_globalfeat

    def compute_rgb(self, x, mask, rgb_in):
        x = self.rgb_fc(x)
        rgb_out, blending_weights_rgb, rho = self.rgb_reduce_fn(x, mask, rgb_in)
        rho = None
        return rgb_out, blending_weights_rgb, rho

    def rgb_reduce_factory(self):
        if self.args.expand_rgb:
            if self.args.rgb_weights:
                return self.expanded_rgb_weighted_rgb_fn
            else:
                return self.expanded_weighted_rgb_fn
        else:
            return self.weighted_rgb_fn

    @staticmethod
    def mean_rgb_fn(_, __, rgb_in):
        rgb_out = torch.mean(rgb_in, dim=2)
        blending_weights_valid = torch.ones_like(rgb_in[..., [0]]) / rgb_in.shape[-2]
        return rgb_out, blending_weights_valid, None

    @staticmethod
    def weighted_rgb_fn(x, mask, rgb_in):
        w = x[..., [0]].masked_fill(~mask, -1e9)
        blending_weights_valid = softmax3d(w, dim=(2, 3, 4))  # color blending
        rgb_out = torch.sum(rgb_in * blending_weights_valid, dim=(2, 3, 4))
        return rgb_out, blending_weights_valid, None

    @staticmethod
    def expanded_weighted_rgb_fn(x, mask, rgb_in):
        w = x.masked_fill((~mask), -1e9).squeeze().view(x.squeeze().shape[:-1] + rgb_in.shape[2:4])
        w = w.permute((0, 1, 3, 4, 2)).unsqueeze(-1)
        blending_weights_valid = softmax3d(w, dim=(2, 3, 4))  # color blending
        rgb_out = torch.sum(rgb_in * blending_weights_valid, dim=(2, 3, 4))
        return rgb_out, blending_weights_valid, None

    @staticmethod
    def expanded_rgb_weighted_rgb_fn(x, mask, rgb_in):
        R, S, k, _, V, C = rgb_in.shape
        w = x.masked_fill((~mask), -1e9).squeeze().view((R, S, V, k, k, C))
        w = w.permute((0, 1, 3, 4, 2, 5))
        blending_weights_valid = softmax3d(w, dim=(2, 3, 4))  # color blending
        rgb_out = torch.sum(rgb_in * blending_weights_valid, dim=(2, 3, 4))
        return rgb_out, blending_weights_valid, None

    @staticmethod
    def conv_weighted_rgb_fn(x, mask, rgb_in):
        # implementation from
        # https://discuss.pytorch.org/t/how-to-apply-different-kernels-to-each-example-in-a-batch-when-using-convolution/84848/3

        rays, samples, k1, k2, views, channels = rgb_in.shape

        # (rays, samples, views, k1, k2, channels)
        rho = torch.sigmoid(x.squeeze()[..., -1].mean(-1)).clamp(1e-3)
        w = x.masked_fill((~mask), -1e9).squeeze()[..., :-1].view((rays, samples, views, 3, 3, channels))

        # (rays*samples*channels)
        conv_weights = softmax3d(w, dim=(2, 3, 4)).permute((0, 1, 5, 2, 3, 4)).reshape((-1, views, 3, 3))
        rgb_out = torch.nn.functional.conv2d(input=rgb_in.permute((0, 1, 5, 4, 2, 3)).reshape(-1, views, k1, k2).reshape((1, -1, k1, k2)),
                                             weight=conv_weights,
                                             padding="valid",
                                             groups=rays*samples*channels)  # color
        rgb_out = rgb_out.reshape((rays, samples, channels, 3, 3)).permute((0, 1, 3, 4, 2))
        return rgb_out, conv_weights, rho

    def _forward_unimplemented(self, *input_: Any) -> None:
        pass

