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

from nan.utils.general_utils import TINY_NUMBER
from nan.attention import MultiHeadAttention

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
    return nn.functional.softmax(x.reshape((R, S, -1, C)), dim=-2).view(x.shape)


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

        self.rgb_fc = self.rgb_fc_factory()

        self.rgb_reduce_fn = self.rgb_reduce_factory()

        # positional encoding
        self.pos_enc_d = 16
        self.pos_encoding = self.pos_enc_generator(n_samples=self.n_samples, d=self.pos_enc_d)

        self.apply(weights_init)

    def rgb_fc_factory(self):
        kernel_numel = prod(self.args.kernel_size)
        if kernel_numel == 1:
            rgb_fc = nn.Sequential(nn.Linear(32 + 1 + 4, 16),
                                   self.activation_func,
                                   nn.Linear(16, 8),
                                   self.activation_func,
                                   nn.Linear(8, 1))

        else:
            rgb_out_channels = kernel_numel
            rgb_pre_out_channels = kernel_numel
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

        return rgb_fc

    def pos_enc_generator(self, n_samples, d):
        position = torch.linspace(0, 1, n_samples, device=self.device).unsqueeze(0) * n_samples
        divider = (10000 ** (2 * torch.div(torch.arange(d, device=self.device),
                                           2, rounding_mode='floor') / d))
        sinusoid_table = (position.unsqueeze(-1) / divider.unsqueeze(0))
        sinusoid_table[..., 0::2] = torch.sin(sinusoid_table[..., 0::2])  # dim 2i
        sinusoid_table[..., 1::2] = torch.cos(sinusoid_table[..., 1::2])  # dim 2i+1

        return sinusoid_table

    def forward(self, rgb_feat, ray_diff, mask, rgb_in, sigma_est):
        """
        :param rgb_feat: rgbs and image features [R, S, k, k, N, F]
        :param ray_diff: ray direction difference [R, S, 1, 1, N, 4], first 3 channels are directions, last channel is inner product
        :param mask: [R, S, 1, 1, N, 1]
        :param rgb_in:  # [R, S, k, k, N, 3]
        :param sigma_est: [R, S, k, k, N, 3]
        :return: rgb [R, S, 3], density [R, S, 1], rgb weights [R, S, k, k, N, 3].
                 For debug: rgb_in and features at the beggining of rho calculation [R, S, F*2+1]
        """

        # [n_rays, n_samples, n_views, 3*n_feat]
        num_valid_obs = mask.sum(dim=-2)
        ext_feat, weight = self.compute_extended_features(ray_diff, rgb_feat, mask, num_valid_obs, sigma_est)

        x = self.base_fc(ext_feat)  # ((32 + 3) x 3) --> MLP --> (32)
        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1] - 1, 1], dim=-1)
        vis = torch.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask

        rho_out, rho_globalfeat = self.compute_rho(x[:, :, 0, 0], vis[:, :, 0, 0], num_valid_obs[:, :, 0, 0])
        x = torch.cat([x, vis, ray_diff], dim=-1)
        rgb_out, w_rgb = self.compute_rgb(x, mask, rgb_in)
        return rgb_out, rho_out, w_rgb, rgb_in, rho_globalfeat

    def compute_extended_features(self, ray_diff, rgb_feat, mask, num_valid_obs, sigma_est):
        direction_feat = self.ray_dir_fc(ray_diff)  # [n_rays, n_samples, k, k, n_views, 35]
        rgb_feat = rgb_feat[:, :, self.k_mid:self.k_mid + 1,
                   self.k_mid:self.k_mid + 1] + direction_feat  # [n_rays, n_samples, 1, 1, n_views, 35]
        feat = rgb_feat

        if self.args.views_attn:
            r, s, k, _, v, f = feat.shape
            feat, _ = self.views_attention(feat, feat, feat, (num_valid_obs > 1).unsqueeze(-1))

        if self.args.noise_feat:
            feat = torch.cat([feat, sigma_est[:, :, self.k_mid:self.k_mid + 1, self.k_mid:self.k_mid + 1]], dim=-1)

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

    def compute_rho(self, x, vis, num_valid_obs):
        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight)
        rho_globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)],
                                   dim=-1)  # [n_rays, n_samples, 32*2+1]
        globalfeat = self.geometry_fc(rho_globalfeat)  # [n_rays, n_samples, 16]

        # positional encoding
        globalfeat = globalfeat + self.pos_encoding

        # ray attention
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=num_valid_obs > 1)  # [n_rays, n_samples, 16]
        rho = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        rho_out = rho.masked_fill(num_valid_obs < 1, 0.)  # set the rho of invalid point to zero

        return rho_out, rho_globalfeat

    def compute_rgb(self, x, mask, rgb_in):
        x = self.rgb_fc(x)
        rgb_out, blending_weights_rgb = self.rgb_reduce_fn(x, mask, rgb_in)
        return rgb_out, blending_weights_rgb

    def rgb_reduce_factory(self):
        if self.args.rgb_weights:
            return self.expanded_rgb_weighted_rgb_fn
        else:
            return self.expanded_weighted_rgb_fn

    @staticmethod
    def expanded_weighted_rgb_fn(x, mask, rgb_in):
        w = x.masked_fill((~mask), -1e9).squeeze().view(x.squeeze().shape[:-1] + rgb_in.shape[2:4])
        w = w.permute((0, 1, 3, 4, 2)).unsqueeze(-1)
        blending_weights_valid = softmax3d(w, dim=(2, 3, 4))  # color blending
        rgb_out = torch.sum(rgb_in * blending_weights_valid, dim=(2, 3, 4))
        return rgb_out, blending_weights_valid

    @staticmethod
    def expanded_rgb_weighted_rgb_fn(x, mask, rgb_in):
        R, S, k, _, V, C = rgb_in.shape
        w = x.masked_fill((~mask), -1e9).squeeze().view((R, S, V, k, k, C))
        w = w.permute((0, 1, 3, 4, 2, 5))
        blending_weights_valid = softmax3d(w, dim=(2, 3, 4))  # color blending
        rgb_out = torch.sum(rgb_in * blending_weights_valid, dim=(2, 3, 4))
        return rgb_out, blending_weights_valid
