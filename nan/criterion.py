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


import torch
import torch.nn as nn
import torch.nn.functional as F

from nan.dataloaders.basic_dataset import process_fn
from nan.raw2output import RaysOutput
from utils import l2_loss, bayesian_img2mse, noise2stats_loss, onehot_penalty, TINY_NUMBER, img2mse_expanded, l1_loss, \
    ssim_loss, mean_with_mask, gen_loss


class RGBCriterion(nn.Module):
    name = 'rgb_loss'

    def __init__(self, args):
        super().__init__()
        self.args = args

    def loss_fn(self, pred, gt, mask):
        pass

    def forward(self, outputs, ray_batch, scalars_to_log):
        """
        training criterion
        """

        pred_rgb  = outputs.rgb
        pred_mask = outputs.mask.float()
        gt_rgb    = ray_batch['rgb']

        if self.args.process_loss:
            pred_rgb = process_fn(pred_rgb, ray_batch['white_level'])
            gt_rgb   = process_fn(gt_rgb,   ray_batch['white_level'])

        loss = self.loss_fn(pred_rgb, gt_rgb, pred_mask)
        scalars_to_log[self.name] = loss
        return loss

    @staticmethod
    def patch_view(x):
        assert x.shape[-1] in [1, 3]
        assert int(x.shape[0] ** 0.5) ** 2 == x.shape[0]
        crop_size = int(x.shape[0] ** 0.5)
        x = x.reshape((crop_size, crop_size, -1))
        return x


class L2Loss(RGBCriterion):
    name = 'rgb_l2'

    def loss_fn(self, pred, gt, mask):
        return l2_loss(pred, gt, mask)


class L1Loss(RGBCriterion):
    name = 'rgb_l1'

    def loss_fn(self, pred, gt, mask):
        return l1_loss(pred, gt, mask)


class SSIMLoss(RGBCriterion):
    name = 'ssim'

    def loss_fn(self, pred, gt, mask):
        return ssim_loss(pred, gt, mask)


class SmoothnessCriterion(RGBCriterion):
    name = 'smooth_loss'


class L1GradLoss(SmoothnessCriterion):
    name  = 'grad_l1'
    alpha = 2

    def loss_fn(self, pred, gt, mask):
    #     # TODO gradient loss https://discuss.pytorch.org/t/how-to-calculate-the-gradient-of-images/1407/6
    #     crop_size = int(pred.shape[-2] ** 0.5)
    #     pred_patch = pred.reshape((crop_size, crop_size, 3))
    #     mask_patch = mask.reshape((crop_size, crop_size, 1))[:-1, :-1]
    #     dy = (pred_patch[:-1, : ]  - pred_patch[1:, : ])[:, :-1]
    #     dx = (pred_patch[:  , :-1] - pred_patch[: , 1:])[:-1, :]
    #     d = (dx ** 2 + dy ** 2) ** 0.5
    #     if mask is None:
    #         return torch.mean(d)
    #     else:
    #         return mean_with_mask(d, mask_patch)
        # gradient
        pred = self.patch_view(pred)
        gt   = self.patch_view(gt)
        mask = self.patch_view(mask.unsqueeze(-1))[..., 0]

        pred_dx, pred_dy = self.gradient(pred)
        gt_dx,   gt_dy   = self.gradient(gt)
        #
        grad_diff_x = torch.abs(gt_dx - pred_dx)
        grad_diff_y = torch.abs(gt_dy - pred_dy)

        # condense into one tensor and avg
        return gen_loss((grad_diff_x ** self.alpha + grad_diff_y ** self.alpha + TINY_NUMBER) ** (1 / self.alpha), mask)

    @staticmethod
    def gradient(x):
        # From https://discuss.pytorch.org/t/how-to-calculate-the-gradient-of-images/1407/6
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (h, w, c), float32 or float64
        # dx, dy: (h, w, c)

        left   = x
        right  = F.pad(x, [0, 0, 0, 1])[:, 1:, :]
        top    = x
        bottom = F.pad(x, [0, 0, 0, 0, 0, 1])[1:, :, :]

        # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        dx, dy = right - left, bottom - top
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[..., :, -1, :] = 0
        dy[..., -1, :, :] = 0

        return dx, dy



class BayesianCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch, scalars_to_log):
        """
        training criterion
        """
        pred_rgb = outputs['rgb']
        pred_mask = outputs['mask'].float()
        gt_rgb = ray_batch['rgb']
        beta = outputs['beta']

        loss = bayesian_img2mse(pred_rgb, gt_rgb, beta, pred_mask)
        # loss = bayesian_img2mse(pred_rgb, gt_rgb, beta, pred_mask) + bayesian_regularization(beta, pred_mask)
        scalars_to_log['rgb_loss'] = loss
        return loss


class NoiseCriterion(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, outputs, ray_batch, scalars_to_log):
        """
        training criterion
        """
        pred_rgb = outputs['rgb']
        pred_noise = outputs['noise']
        pred_target_noise = pred_noise[..., 0, :]
        pred_mask = outputs['mask'].float()
        gt_noisy_rgb = ray_batch['rgb']

        img_loss = l2_loss(pred_rgb + pred_target_noise, gt_noisy_rgb, pred_mask)

        noise_std = outputs['noise_std']
        loss_std, loss_mean = noise2stats_loss(pred_noise, noise_std, pred_mask)

        loss = img_loss + self.gamma * loss_std + self.gamma * loss_mean

        return loss


class DeltaCriterion(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, outputs, ray_batch, scalars_to_log):
        """
        training criterion
        """
        rgb_loss = l2_loss(outputs.rgb, ray_batch['rgb'], outputs.mask.float())
        weights_loss = onehot_penalty(outputs.weights)
        return rgb_loss - self.gamma * weights_loss


class EntropyCriterion(nn.Module):
    def __init__(self, gamma, h=64):
        super().__init__()
        self.gamma = gamma
        self.h = h

    def forward(self, outputs: RaysOutput, ray_batch, scalars_to_log):
        """
        training criterion
        """
        rgb_loss = l2_loss(outputs.rgb, ray_batch['rgb'], outputs.mask.float())
        w_hist = self.create_w_hist(outputs.weights, outputs.z_vals, ray_batch['depth_range'], self.h)
        weights_loss = self.entropy_penalty(w_hist)

        scalars_to_log['rgb_loss'] = rgb_loss
        scalars_to_log['entropy_loss'] = weights_loss

        return rgb_loss + self.gamma * weights_loss

    @staticmethod
    def create_w_hist(w, z, depth_range, h):
        assert depth_range.shape[0] == 1
        bins = torch.linspace(depth_range[0, 0], depth_range[0, 1], steps=h, device=z.device)
        tau = (bins[1] - bins[0]) / 2
        k = 1 / (1 + ((z.unsqueeze(-1) - bins) / tau) ** 2)
        H = (k * w.unsqueeze(-1)).sum(-2)
        H = H / (H.sum(1, keepdims=True) + TINY_NUMBER)
        return H

    @staticmethod
    def plot_hist(bins, z, H, w, tau):
        import matplotlib.pyplot as plt
        r = 8
        plt.figure(figsize=(4, 3))
        plt.plot(bins.detach().cpu().numpy(), H[r].detach().cpu().numpy(), label='calculated histogram')
        plt.plot(z[r].detach().cpu().numpy(), w[r].detach().cpu().numpy(), 'D', markersize=2, label='network output')

        k0 = 1 / (1 + ((bins - 7) / tau) ** 2)
        plt.plot(bins.detach().cpu().numpy(), k0.detach().cpu().numpy(), label='weighting kernel')
        plt.title("Differential histogram")
        plt.xlabel(r"$z$ [m]")
        plt.ylabel(r"$w(z)=T(z)\cdot\alpha(z)$")
        plt.legend()
        plt.subplots_adjust(top=0.91,
                            bottom=0.16,
                            left=0.145,
                            right=0.975,
                            hspace=0.2,
                            wspace=0.2)
        plt.show()


    @staticmethod
    def entropy_penalty(w_hist):
        b = -F.softmax(w_hist, dim=1) * F.log_softmax(w_hist, dim=1)
        return b.sum(1).mean()


class FixedConvCriterion(nn.Module):
    def __init__(self, ray_output):
        super().__init__()
        self.ray_output = ray_output

    def forward(self, outputs, ray_batch, scalars_to_log):
        """
        training criterion
        """
        pred_rgb  = outputs.rgb
        pred_mask = outputs.mask.float()
        gt_rgb    = ray_batch['rgb']

        if outputs.rho is None:
            w = self.ray_output.w
        else:
            w = self.ray_output.gaussian(outputs.rho)

        loss = img2mse_expanded(pred_rgb, gt_rgb, w, pred_mask)
        scalars_to_log['rgb_loss'] = loss
        return loss


loss_mapping = {'l2': L2Loss, 'l1': L1Loss, 'l1_grad': L1GradLoss, 'ssim': SSIMLoss}


class NANLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.losses_list = []
        self.weights_list = []
        for loss_type, weight in zip(args.losses, args.losses_weights):
            if weight == 0:
                continue
            self.losses_list.append(loss_mapping[loss_type](args))
            self.weights_list.append(weight)

    def forward(self, outputs, ray_batch, scalars_to_log):
        return sum([w * loss(outputs, ray_batch, scalars_to_log) for w, loss in zip(self.weights_list, self.losses_list)])


def criterion_factory(args, ray_output):
    # Create criterion
    if args.loss == 'entropy':
        raise NotImplementedError
        return EntropyCriterion(gamma=args.gamma)
    elif args.loss in ['l1', 'l2', 'ssim']:
        return loss_mapping[args.loss](args)
    elif args.loss in ['ssim']:
        pass
    else:
        raise IOError



















