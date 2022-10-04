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

from nan.dataloaders.basic_dataset import de_linearize
from nan.utils.general_utils import TINY_NUMBER
from nan.losses import l2_loss, l1_loss, gen_loss
from nan.utils.eval_utils import ssim_loss


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
            pred_rgb = de_linearize(pred_rgb, ray_batch['white_level'])
            gt_rgb   = de_linearize(gt_rgb, ray_batch['white_level'])

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


loss_mapping = {'l2': L2Loss,
                'l1': L1Loss,
                'l1_grad': L1GradLoss,
                'ssim': SSIMLoss}


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





















