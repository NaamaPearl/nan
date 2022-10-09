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
from typing import Dict

import torch
from collections import OrderedDict

from nan.model import NANScheme
from nan.projection import Projector
from nan.raw2output import RaysOutput


def sample_pdf(bins, weights, N_samples, det=False):
    """
    @param: bins: tensor of shape [N_rays, M+1], M is the number of bins
    @param: weights: tensor of shape [N_rays, M]
    @param: N_samples: number of samples along each ray
    @param: det: if True, will perform deterministic sampling
    @return: [N_rays, N_samples]
    """

    N_rays, M = weights.shape
    weights += 1e-5
    # Get pdf
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # [N_rays, M]
    cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, M]
    cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1)  # [N_rays, M+1]

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., N_samples, device=bins.device)
        u = u.unsqueeze(0).expand(bins.shape[0], -1)  # [N_rays, N_samples]
    else:
        u = torch.rand(bins.shape[0], N_samples, device=bins.device)

    # Invert CDF
    above_inds = torch.zeros_like(u, dtype=torch.long)  # [N_rays, N_samples]
    for i in range(M):
        above_inds += (u >= cdf[:, i:i + 1]).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds - 1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=2)  # [N_rays, N_samples, 2]

    cdf = cdf.unsqueeze(1).expand(N_rays, N_samples, M + 1)  # [N_rays, N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    bins = bins.unsqueeze(1).expand(N_rays, N_samples, M + 1)  # [N_rays, N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    # t = (u-cdf_g[:, :, 0]) / (cdf_g[:, :, 1] - cdf_g[:, :, 0] + TINY_NUMBER)  # [N_rays, N_samples]
    # fix numeric issue
    denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]  # [N_rays, N_samples]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[:, :, 0]) / denom

    samples = bins_g[:, :, 0] + t * (bins_g[:, :, 1] - bins_g[:, :, 0])

    return samples


class RayRender:
    """
    Object that handle the actual rendering:
    * Calculating features maps
    * Sampling 3D points along a ray
    * Projecting them to each of the other views
    * Feeding the network with the incident pixels
    * Rendering rho and RGB values to each 3D point along a ray
    * Aggregate the values from 3D points along a specific ray
    """
    def __init__(self, model: NANScheme, args, device, save_pixel=None):
        self.model = model
        self.device = device
        self.projector = Projector(device=device, args=args)

        self.N_samples = args.N_samples
        self.inv_uniform = args.inv_uniform
        self.N_importance = args.N_importance
        self.det = args.det
        self.white_bkgd = args.white_bkgd

        # For debug purposes
        if save_pixel is not None:
            y, x = tuple(zip(*save_pixel))
            self.save_pixels = torch.tensor((x, y, (1,) * len(x)))
        else:
            self.save_pixels = None

        self.fine_processing = args.N_importance > 0
        if self.fine_processing:
            assert self.model.net_fine is not None

    def pixel2index(self, ray_batch):
        if self.save_pixels is not None:
            row_detection = (ray_batch['xyz'].unsqueeze(-1) == self.save_pixels.unsqueeze(0)).all(1)
            idx_in_batch, idx_in_all = torch.where(row_detection)
            if len(idx_in_batch) > 0:
                return tuple(zip(*(idx_in_batch.tolist(), self.save_pixels[:2:, idx_in_all].T.flip(1).tolist())))

    def sample_along_ray_coarse(self, ray_o, ray_d, depth_range):
        """
        :param: ray_o: origin of the ray in scene coordinate system; tensor of shape [N_rays, 3]
        :param: ray_d: homogeneous ray direction vectors in scene coordinate system; tensor of shape [N_rays, 3]
        :param: depth_range: [near_depth, far_depth]
        :param: inv_uniform: if True, uniformly sampling inverse depth
        :param: det: if True, will perform deterministic sampling
        :return: 3D point: tensor of shape [N_rays, N_samples, 3], z_vals: tensor of shape [N_rays, N_samples, 3]
        """
        # will sample inside [near_depth, far_depth]
        # assume the nearest possible depth is at least (min_ratio * depth)
        near_depth_value = depth_range[0, 0]
        far_depth_value = depth_range[0, 1]
        assert 0 < near_depth_value < far_depth_value and far_depth_value > 0

        near_depth = near_depth_value * torch.ones_like(ray_d[..., 0])
        far_depth = far_depth_value * torch.ones_like(ray_d[..., 0])

        if self.inv_uniform:
            start = 1. / near_depth  # [N_rays,]
            step = (1. / far_depth - start) / (self.N_samples - 1)
            inv_z_vals = torch.stack([start + i * step for i in range(self.N_samples)], dim=1)  # [N_rays, N_samples]
            z_vals = 1. / inv_z_vals
        else:
            start = near_depth
            step = (far_depth - near_depth) / (self.N_samples - 1)
            z_vals = torch.stack([start + i * step for i in range(self.N_samples)], dim=1)  # [N_rays, N_samples]

        if not self.det:
            # get intervals between samples
            mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
            upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
            lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)
            # uniform samples in those intervals
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

        pts = z_vals.unsqueeze(2) * ray_d.unsqueeze(1) + ray_o.unsqueeze(1)  # [N_rays, N_samples, 3]
        return pts, z_vals

    def sample_along_ray_fine(self, coarse_out: RaysOutput, z_vals, ray_batch):
        # detach since we would like to decouple the coarse and fine networks
        weights = coarse_out.weights.clone().detach()  # [N_rays, N_samples]

        if self.inv_uniform:
            inv_z_vals = 1. / z_vals
            inv_z_vals_mid = .5 * (inv_z_vals[:, 1:] + inv_z_vals[:, :-1])  # [N_rays, N_samples-1]
            weights = weights[:, 1:-1]  # [N_rays, N_samples-2]
            inv_z_vals = sample_pdf(bins=torch.flip(inv_z_vals_mid, dims=[1]),
                                    weights=torch.flip(weights, dims=[1]),
                                    N_samples=self.N_importance, det=self.det)  # [N_rays, N_importance]
            z_samples = 1. / inv_z_vals
        else:
            # take mid-points of depth samples
            z_vals_mid = .5 * (z_vals[:, 1:] + z_vals[:, :-1])  # [N_rays, N_samples-1]
            weights = weights[:, 1:-1]  # [N_rays, N_samples-2]
            z_samples = sample_pdf(bins=z_vals_mid, weights=weights,
                                   N_samples=self.N_importance, det=self.det)  # [N_rays, N_importance]
        z_vals = torch.cat((z_vals, z_samples), dim=-1)  # [N_rays, N_samples + N_importance]

        # samples are sorted with increasing depth
        z_vals, _ = torch.sort(z_vals, dim=-1)
        N_total_samples = self.N_samples + self.N_importance
        N_rays = weights.shape[0]
        viewdirs = ray_batch['ray_d'].unsqueeze(1).expand(N_rays, N_total_samples, 3)
        ray_o = ray_batch['ray_o'].unsqueeze(1).expand(N_rays, N_total_samples, 3)
        pts = z_vals.unsqueeze(2) * viewdirs + ray_o  # [N_rays, N_samples + N_importance, 3]
        return pts, z_vals

    def render_batch(self, ray_batch, proc_src_rgbs, featmaps, org_src_rgbs,
                     sigma_estimate) -> Dict[str, RaysOutput]:
        """
        :param sigma_estimate: (1, N, H, W, 3)
        :param org_src_rgbs: (1, N, H, W, 3)
        :param proc_src_rgbs: org_src_rgbs after processing with pre_net, if exists (1, N, H, W, 3)
        :param featmaps: dict {'corase': (N, C, H', W'), 'fine': (N, C, H', W')}
        :param ray_batch: {ray_o: (R, 3),
                           ray_d: (R, 3),
                           camera: (1, 34),
                           depth_range: (1, 2),
                           src_cameras: (1, N, 34),
                           selected_inds: (R,),
                           xyz: (R, 3),
                           rgb: (R, 3),
                           white_level: (1, 1)}
        :return: {'coarse': {}, 'fine': {}}
        """

        # Find pixels for debug
        save_idx = self.pixel2index(ray_batch)

        # Create the output dictionary
        batch_out = {'coarse': None,
                     'fine': None}

        # pts:    [R, S, 3]
        # z_vals: [R, S]
        # Sample points along ray for coarse phase
        pts_coarse, z_vals_coarse = self.sample_along_ray_coarse(ray_o=ray_batch['ray_o'],
                                                                 ray_d=ray_batch['ray_d'],
                                                                 depth_range=ray_batch['depth_range'])

        # Process the rays and return the coarse phase output
        coarse_ray_out = self.process_rays_batch(ray_batch=ray_batch, pts=pts_coarse, z_vals=z_vals_coarse, save_idx=save_idx,
                                         level='coarse', proc_src_rgbs=proc_src_rgbs, featmaps=featmaps,
                                         org_src_rgbs=org_src_rgbs, sigma_estimate=sigma_estimate)
        batch_out['coarse'] = coarse_ray_out

        if self.fine_processing:
            # Sample points along ray for fine phase, based on the coarse output
            pts_fine, z_vals_fine = self.sample_along_ray_fine(coarse_out=coarse_ray_out,
                                                               z_vals=z_vals_coarse,
                                                               ray_batch=ray_batch)

            # Process the rays and return the fine phase output
            fine = self.process_rays_batch(ray_batch=ray_batch, pts=pts_fine, z_vals=z_vals_fine, save_idx=save_idx,
                                           level='fine', proc_src_rgbs=proc_src_rgbs, featmaps=featmaps,
                                           org_src_rgbs=org_src_rgbs, sigma_estimate=sigma_estimate)

            batch_out['fine'] = fine
        return batch_out

    def process_rays_batch(self, ray_batch, pts, z_vals, save_idx, level, proc_src_rgbs, featmaps,
                           org_src_rgbs, sigma_estimate):
        """
        :param sigma_estimate: (1, N, H, W, 3)
        :param org_src_rgbs: (1, N, H, W, 3)
        :param proc_src_rgbs: (1, N, H, W, 3)
        :param featmaps: (N, 3, H', W')
        :param level: str {'coarse', 'fine'}
        :param ray_batch: dictionary with rays origin, direction and relevant data for rendering
        :param pts: 3D points along the rays (R, S, 3)
        :param z_vals: z values from which the 3D points were calcuated (R, S)
        :param save_idx: indices for debug purposes
        :return: RaysOutput object of the rendered values
        """
        # Project the pts along the rays batch on all others views (src views)
        # based on the target camera and src cameras (intrinsics - K, rotation - R, translation - t)
        proj_out = self.projector.compute(pts, ray_batch['camera'], proc_src_rgbs, org_src_rgbs, sigma_estimate,
                                          ray_batch['src_cameras'],
                                          featmaps=featmaps[level])  # [N_rays, N_samples, N_views, x]
        rgb_feat, ray_diff, pts_mask, org_rgb, sigma_est = proj_out

        # [N_rays, N_samples, 4]
        # Process the feature vectors of all 3D points along each ray to predict density and rgb value
        rgb_out, rho_out, *debug_info = self.model.mlps[level](rgb_feat, ray_diff,
                                                               pts_mask.unsqueeze(-3).unsqueeze(-3),
                                                               org_rgb, sigma_est)
        ray_outputs = RaysOutput.raw2output(rgb_out, rho_out, z_vals, pts_mask, white_bkgd=self.white_bkgd)

        if save_idx is not None:
            debug_dict = {}
            for idx, pixel in save_idx:
                debug_dict[(tuple(pixel))] = OrderedDict([('z', ray_outputs.z_vals[idx].cpu()),
                                                          ('w', ray_outputs.weights[idx].cpu()),
                                                          ('w_rgb', debug_info[0][idx].cpu()),
                                                          ('feat', debug_info[1][idx].cpu()),
                                                          ('globalfeat_attention', debug_info[2][idx].cpu())])
            ray_outputs.debug = debug_dict

        return ray_outputs

    def calc_featmaps(self, src_rgbs):
        """
        Calculating the features maps of the source views
        :param src_rgbs: (1, N, H, W, 3)
        :return: src_rgbs after pre_net (if exists) (1, N, H, W, 3),
                 features maps: dict {'coarse': (N, C, H', W'), 'fine': (N, C, H', W')}
        """
        if self.model.pre_net is not None:
            src_rgbs = self.model.pre_net(src_rgbs.squeeze(0).permute(0, 3, 1, 2))  # (N, 3, H, W)
            featmaps = self.model.feature_net(src_rgbs)
            src_rgbs = src_rgbs.permute((0, 2, 3, 1)).unsqueeze(0)  # (1, N, H, W, 3)
        else:
            featmaps = self.model.feature_net(src_rgbs.squeeze(0).permute(0, 3, 1, 2))

        return src_rgbs, featmaps
