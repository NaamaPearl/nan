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
import torch
import torch.nn.functional as F


class Projector:
    """
    Project that project 3D points to pixel locations in the other views and extract the relevant
    values in images and features maps.
    """
    def __init__(self, device, args):
        self.kernel_size = args.kernel_size
        self.kernel_enum = prod(args.kernel_size)
        self.expander, self.reshape_features = self.pixel_location_expander_factory(args.kernel_size, device)
        self.device = device
        self.args = args

    @staticmethod
    def inbound(pixel_locations, h, w):
        """
        check if the pixel locations are in valid range
        :param pixel_locations: [..., 2]
        :param h: height
        :param w: weight
        :return: mask, bool, [...]
        """
        return (pixel_locations[..., 0] <= w - 1.) & \
               (pixel_locations[..., 0] >= 0) & \
               (pixel_locations[..., 1] <= h - 1.) & \
               (pixel_locations[..., 1] >= 0)

    @staticmethod
    def normalize(pixel_locations, h, w):
        resize_factor = torch.tensor([w - 1., h - 1.]).to(pixel_locations.device)[None, None, :]
        normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
        return normalized_pixel_locations

    def compute_projections(self, xyz, src_cameras):
        """
        project 3D points into cameras
        :param xyz: [n_rays, n_samples, 3]   
        :param src_cameras: [n_views, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :return: pixel locations [..., 2], mask [...]
        """
        n_rays, n_samples = xyz.shape[:2]
        xyz               = xyz.reshape(-1, 3)  # [n_points, 3], n_points = n_rays * n_samples
        xyz_h             = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # [n_points, 4]

        num_views      = len(src_cameras)
        h, w           = src_cameras[0][:2]
        src_intrinsics = src_cameras[:, 2:18].reshape(-1, 4, 4)  # [n_views, 4, 4]
        src_poses      = src_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]

        projections = src_intrinsics.bmm(torch.inverse(src_poses)) \
            .bmm(xyz_h.t()[None, ...].expand(num_views, 4, xyz.shape[0]))  # [n_views, 4, n_points]
        projections = projections.permute(0, 2, 1)  # [n_views, n_points, 4]
        mask_in_front = projections[..., 2] > 0  # a point is invalid if behind the camera,  [n_views, n_points]

        uv = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)  # [n_views, n_points, 2]
        uv = torch.clamp(uv, min=-1e6, max=1e6)
        mask_inbound = self.inbound(uv, h, w)  # a point is invalid if out of the image
        uv = self.expander(uv)  # [n_views, n_points, kernel_size, 2]

        mask = mask_in_front * mask_inbound

        # split second dimension (n_points) into 2 dimensions: (n_rays, n_samples):
        # uv (because of F.grid_sample): [n_views, n_points, k, k, 2]  -->  [n_views, n_rays, n_samples*k*k, 2]
        # mask                         : [n_views, n_points]           -->  [n_views, n_rays, n_samples]
        return uv.view((num_views, n_rays, n_samples * self.kernel_enum, 2)), mask.view((num_views, n_rays, n_samples))
                                                                              
    @staticmethod
    def compute_angle(xyz, query_camera, train_cameras):
        """
        :param xyz: [n_rays, n_samples, 3]   
        :param query_camera: [34, ]
        :param train_cameras: [n_views, 34]
        :return: [n_views, ..., 4]; The first 3 channels are unit-length vector of the difference between
        query and target ray directions, the last channel is the inner product of the two directions.
        """
        original_shape = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)
        train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]
        num_views = len(train_poses)
        query_pose = query_camera[-16:].reshape(-1, 4, 4).expand(num_views, 4, 4)  # [n_views, 4, 4]
        ray2tar_pose = (query_pose[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0))
        ray2tar_pose /= (torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6)
        ray2train_pose = (train_poses[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0))
        ray2train_pose /= (torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6)
        ray_diff = ray2tar_pose - ray2train_pose
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(ray2tar_pose * ray2train_pose, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
        ray_diff = ray_diff.reshape((num_views,) + original_shape + (4,))
        return ray_diff

    def compute(self, xyz, query_camera, src_imgs, org_src_imgs, sigma_estimate, src_cameras, featmaps):
        """ Given 3D points and the camera of the target and src views,
        computing the rgb values of the incident pixels and their kxk environment.

        :param src_imgs: (1, N, W, H, 3)
        :param xyz: 3D points along the rays [R, S, 3]
        :param query_camera: [1, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :param src_cameras: [1, N, 34]
        :param featmaps: [N, C, H', W']
        :return: rgb_feat_sampled: [R, S, k, k, N, 3+F],
                 ray_diff: [R, S, 1, 1, N, 4],
                 3D points mask: [R, S, N, 1], 3D point is valid only if it is not behind the frame
                                               and if the projected pixel is inbound the view frame
                 org_rgbs_sampled: [R, S, k, k, N, 3+F],
                 sigma_estimate: [R, S, k, k, N, 3+F]
        """
        assert (src_imgs.shape[0] == 1) \
               and (src_cameras.shape[0] == 1) \
               and (query_camera.shape[0] == 1), 'only support batch_size=1 for now'

        src_imgs       = src_imgs.squeeze(0)  # [n_views, h, w, 3]
        src_imgs       = src_imgs.permute(0, 3, 1, 2)  # [n_views, 3, h, w]

        org_src_imgs   = org_src_imgs.squeeze(0)  # [n_views, h, w, 3]
        org_src_imgs   = org_src_imgs.permute(0, 3, 1, 2)  # [n_views, 3, h, w]

        src_cameras  = src_cameras.squeeze(0)  # [n_views, 34]
        query_camera = query_camera.squeeze(0)  # [34, ]

        # compute the projection of the query points to each reference image
        xys, mask = self.compute_projections(xyz, src_cameras)  # [n_views, n_rays, n_samples * prod(kernel_size), 2]
        h, w = src_cameras[0][:2]
        norm_xys = self.normalize(xys, h, w)  # [n_views, n_rays, n_samples, 2]

        # noise features sampling
        if self.args.noise_feat:
            sigma_estimate = sigma_estimate.squeeze(0)  # [n_views, h, w, 3]
            sigma_estimate = sigma_estimate.permute(0, 3, 1, 2)  # [n_views, 3, h, w]
            sigma_estimate = F.grid_sample(sigma_estimate, norm_xys, align_corners=True)
            sigma_estimate = sigma_estimate.permute(2, 3, 0, 1)  # [n_rays, n_samples, n_views, 3]
            sigma_estimate = self.reshape_features(sigma_estimate)
        else:
            sigma_estimate = None

        # rgb sampling
        rgbs_sampled = F.grid_sample(src_imgs, norm_xys, align_corners=True)
        rgbs_sampled = rgbs_sampled.permute(2, 3, 0, 1)  # [n_rays, n_samples, n_views, 3]

        org_rgbs_sampled = F.grid_sample(org_src_imgs, norm_xys, align_corners=True)
        org_rgbs_sampled = org_rgbs_sampled.permute(2, 3, 0, 1)  # [n_rays, n_samples, n_views, 3]
        org_rgbs_sampled = self.reshape_features(org_rgbs_sampled)

        # deep feature sampling
        feat_sampled = F.grid_sample(featmaps, norm_xys, align_corners=True)
        feat_sampled = feat_sampled.permute(2, 3, 0, 1)  # [n_rays, n_samples, n_views, d]
        rgb_feat_sampled = torch.cat([rgbs_sampled, feat_sampled], dim=-1)  # [n_rays, n_samples, n_views, d+3]
        rgb_feat_sampled = self.reshape_features(rgb_feat_sampled)

        # ray_diff
        ray_diff = self.compute_angle(xyz, query_camera, src_cameras)
        ray_diff = ray_diff.permute(1, 2, 0, 3).unsqueeze(-3).unsqueeze(-3)

        # mask
        mask = mask.permute(1, 2, 0)[..., None]  # [n_rays, n_samples, n_views, 1]
        return rgb_feat_sampled, ray_diff, mask, org_rgbs_sampled, sigma_estimate

    @staticmethod
    def pixel_location_expander_factory(kernel_size, device):
        kx, ky = kernel_size
        assert kx % 2 == 1 and ky % 2 == 1

        x_range = torch.arange(-(kx // 2), kx // 2 + 1)
        y_range = torch.arange(-(ky // 2), ky // 2 + 1)
        kernel_expander = torch.stack(torch.meshgrid(x_range, y_range, indexing='ij')[::-1]).permute((1, 2, 0)).to(device)

        def expander(pixel_locations):
            expanded_pixels = pixel_locations.unsqueeze(-2).unsqueeze(-2) + kernel_expander
            return expanded_pixels

        def reshape_features(pixel_locations):
            return pixel_locations.reshape((pixel_locations.shape[0],) +
                                           (-1,) + kernel_size + pixel_locations.shape[2:])

        return expander, reshape_features
