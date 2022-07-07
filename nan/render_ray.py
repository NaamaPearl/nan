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
from nan.raw2output import RaysOutput, rays_output_factory


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
        u = u.unsqueeze(0).expand(bins.shape[0], 1)  # [N_rays, N_samples]
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
    def __init__(self, model: NANScheme, args, device, save_pixel=None):
        self.model = model
        self.device = device
        self.projector = Projector(device=device, args=args)

        self.N_samples = args.N_samples
        self.ray_output = rays_output_factory(args)
        self.inv_uniform = args.inv_uniform
        self.N_importance = args.N_importance
        self.det = args.det
        self.white_bkgd = args.white_bkgd

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
        @param: ray_o: origin of the ray in scene coordinate system; tensor of shape [N_rays, 3]
        @param: ray_d: homogeneous ray direction vectors in scene coordinate system; tensor of shape [N_rays, 3]
        @param: depth_range: [near_depth, far_depth]
        @param: inv_uniform: if True, uniformly sampling inverse depth
        @param: det: if True, will perform deterministic sampling
        @return: tensor of shape [N_rays, N_samples, 3]
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

    def render_batch(self, ray_batch, src_rgbs, featmaps, org_src_rgbs,
                     sigma_estimate) -> Dict[str, RaysOutput]:
        """
        @param org_src_rgbs:
        @param src_rgbs:
        @param featmaps:
        @param ray_batch: {'ray_o': [N_rays, 3] , 'ray_d': [N_rays, 3], 'view_dir': [N_rays, 2]}
        @return: {'coarse': {}, 'fine': {}}
        """

        save_idx = self.pixel2index(ray_batch)

        batch_out = {'coarse': None,
                     'fine': None}

        # pts:    [N_rays, N_samples, 3]
        # z_vals: [N_rays, N_samples]

        pts_coarse, z_vals_coarse = self.sample_along_ray_coarse(ray_o=ray_batch['ray_o'],
                                                                 ray_d=ray_batch['ray_d'],
                                                                 depth_range=ray_batch['depth_range'])
        coarse = self.process_ray(ray_batch=ray_batch, pts=pts_coarse, z_vals=z_vals_coarse,
                                  model=self.model.net_coarse, save_idx=save_idx, level=0, src_rgbs=src_rgbs,
                                  featmaps=featmaps, org_src_rgbs=org_src_rgbs, sigma_estimate=sigma_estimate)
        batch_out['coarse'] = coarse

        if self.fine_processing:
            pts_fine, z_vals_fine = self.sample_along_ray_fine(coarse_out=coarse,
                                                               z_vals=z_vals_coarse,
                                                               ray_batch=ray_batch)
            fine = self.process_ray(ray_batch, pts_fine, z_vals_fine, self.model.net_fine, save_idx, 1,
                                    src_rgbs, featmaps, org_src_rgbs, sigma_estimate)

            batch_out['fine'] = fine
        return batch_out

    def ray2raw(self, pts, ray_batch, src_rgbs, org_src_rgbs, featmap, sigma_est, model, z_vals):
        proj_out = self.projector.compute(pts, ray_batch['camera'], src_rgbs, org_src_rgbs, sigma_est,
                                          ray_batch['src_cameras'], featmaps=featmap)  # [N_rays, N_samples, N_views, x]
        rgb_feat, ray_diff, mask, org_rgb, sigma_est = proj_out

        pixel_mask = mask[..., 0].sum(dim=2) > 1  # [N_rays, N_samples], should at least have 2 observations

        # [N_rays, N_samples, 4]
        rgb_out, sigma_out, rho, *debug_info = model(rgb_feat, ray_diff, mask.unsqueeze(-3).unsqueeze(-3),
                                                     org_rgb, sigma_est)
        return rgb_out, sigma_out, rho, pixel_mask, *debug_info

    def process_ray(self, ray_batch, pts, z_vals, model, save_idx, level, src_rgbs, featmaps,
                    org_src_rgbs, sigma_estimate):
        """
        @param src_rgbs:
        @param featmaps:
        @param level:
        @param ray_batch:
        @param pts:
        @param z_vals:
        @param model:
        @param save_idx:
        @return:
        """
        rgb_out, sigma_out, rho, pixel_mask, *debug_info = self.ray2raw(pts, ray_batch, src_rgbs, org_src_rgbs,
                                                                        featmaps[level], sigma_estimate, model, z_vals)
        return self.raw2output(rgb_out, sigma_out, rho, z_vals, pixel_mask, save_idx, *debug_info)

    def process_ray_clean(self, ray_batch, pts, z_vals, model, save_idx, level, src_rgbs, featmaps, src_rgbs_clean,
                          featmaps_clean,
                          org_src_rgbs, sigma_estimate):
        """
        @param _:
        @param __:
        @param src_rgbs_clean:
        @param featmaps_clean:
        @param level:
        @param ray_batch:
        @param pts:
        @param z_vals:
        @param model:
        @param save_idx:
        @return:
        """
        rgb_out, sigma_out, pixel_mask, *debug_info = self.ray2raw(pts, ray_batch, src_rgbs_clean,
                                                                   featmaps_clean[level], model,
                                                                   z_vals)
        return self.raw2output(rgb_out, sigma_out, z_vals, pixel_mask, save_idx, *debug_info)

    def process_ray_mixed(self, ray_batch, pts, z_vals, model, save_idx, level, src_rgbs, featmaps, src_rgbs_clean,
                          featmaps_clean, org_src_rgbs, sigma_estimate):
        rgb_out, _, pixel_mask, *debug_info = self.ray2raw(pts, ray_batch, src_rgbs, featmaps[level], model, z_vals)
        _, sigma_out_clean, *_ = self.ray2raw(pts, ray_batch, src_rgbs_clean, featmaps_clean[level], model, z_vals)
        return self.raw2output(rgb_out, sigma_out_clean, z_vals, pixel_mask, save_idx, *debug_info)

    def raw2output(self, rgb_out, sigma_out, rho, z_vals, pixel_mask, save_idx, *debug_info):
        outputs = self.ray_output.raw2output(rgb_out, sigma_out, rho, z_vals, pixel_mask, white_bkgd=self.white_bkgd)

        if save_idx is not None:
            debug_dict = {}
            for idx, pixel in save_idx:
                debug_dict[(tuple(pixel))] = OrderedDict([('z', outputs.z_vals[idx].cpu()),
                                                          ('w', outputs.weights[idx].cpu()),
                                                          ('w_rgb', debug_info[0][idx].cpu()),
                                                          ('feat', debug_info[1][idx].cpu()),
                                                          ('globalfeat_transformer', debug_info[2][idx].cpu())])
            outputs.debug = debug_dict
        return outputs

    def calc_featmaps(self, src_rgbs):
        src_rgbs = src_rgbs.to(self.device)
        if self.model.pre_net is not None:
            src_rgbs = self.model.pre_net(src_rgbs.squeeze(0).permute(0, 3, 1, 2)).permute(
                # TODO redundant permute calls
                (0, 2, 3, 1)).unsqueeze(0)
            featmaps = self.model.feature_net(src_rgbs.squeeze(0).permute(0, 3, 1, 2))
        else:
            featmaps = self.model.feature_net(src_rgbs.squeeze(0).permute(0, 3, 1, 2))

        return src_rgbs, featmaps
