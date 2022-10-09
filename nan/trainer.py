import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from configs.local_setting import OUT_DIR, LOG_DIR
from nan.criterion import NANLoss
from nan.dataloaders import dataset_dict
from nan.dataloaders.basic_dataset import de_linearize, Mode
from nan.dataloaders.create_training_dataset import create_training_dataset
from nan.dataloaders.data_utils import cycle
from nan.losses import l2_loss
from nan.model import NANScheme
from nan.render_image import render_single_image
from nan.render_ray import RayRender
from nan.sample_ray import RaySampler
from nan.utils.eval_utils import mse2psnr, img2psnr
from nan.utils.general_utils import img_HWC2CHW
from nan.utils.io_utils import print_link, colorize


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.local_rank}")

        # exp_out_dir will contain .pth weights and .yml config files for the experiment
        self.exp_out_dir = OUT_DIR / args.expname
        self.exp_out_dir.mkdir(exist_ok=True, parents=True)
        print_link(self.exp_out_dir, 'outputs will be saved to')

        # save the args and config files
        self.save_ymls(args, sys.argv[1:], self.exp_out_dir)

        # create training dataset
        self.train_dataset, self.train_sampler = create_training_dataset(args)
        # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
        # please use distributed parallel on multiple GPUs to train multiple target views per batch
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=1,
                                                        worker_init_fn=lambda _: np.random.seed(),
                                                        num_workers=args.workers,
                                                        pin_memory=True,
                                                        sampler=self.train_sampler,
                                                        shuffle=True if self.train_sampler is None else False)

        # create validation dataset
        self.val_dataset = dataset_dict[args.eval_dataset](args, Mode.validation, scenes=args.eval_scenes)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1)
        self.val_loader_iterator = iter(cycle(self.val_loader))

        # Create NAN scheme
        self.model = NANScheme.create(args)
        self.last_weights_path = None

        # Create ray render object
        self.ray_render = RayRender(model=self.model, args=args, device=self.device)

        # Create criterion
        self.criterion = NANLoss(args)

        # tb_dir will contain tensorboard files and later evaluation results
        tb_dir = LOG_DIR / args.expname
        if args.local_rank == 0:
            self.writer = SummaryWriter(str(tb_dir))
            print_link(tb_dir, 'saving tensorboard files to')
        # dictionary to store scalars to log in tb
        self.scalars_to_log = {}

    @staticmethod
    def save_ymls(args, additional_args, out_folder):
        with open(out_folder / 'args.yml', 'w') as f:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                f.write(f'{arg} : {attr}\n')

        if args.config is not None:
            with open(str(args.config)) as f:
                contents = yaml.safe_load(f)
        else:
            contents = {}
        for arg in filter(lambda s: s[:2] == '--', additional_args):
            val = vars(args)[arg[2:]]
            if isinstance(val, Path):
                val = str(val)
            contents[arg[2:]] = val

        with open(out_folder / 'config.yml', 'w') as f:
            yaml.safe_dump(contents, f, default_flow_style=None)

    def train(self):
        global_step = self.model.start_step + 1
        epoch = 0  # epoch is not consistent when loading ckpt, it affects train_sampler when distributed and prints

        while global_step < self.args.n_iters + 1:
            np.random.seed()
            for train_data in self.train_loader:
                time0 = time.time()
                if self.args.distributed:
                    self.train_sampler.set_epoch(epoch)

                # core optimization loop
                ray_batch_out, ray_batch_in = self.training_loop(train_data)
                dt = time.time() - time0

                # Logging and saving
                self.logging(train_data, ray_batch_in, ray_batch_out, dt, global_step, epoch)

                global_step += 1
                if global_step > self.model.start_step + self.args.n_iters + 1:
                    break
            epoch += 1
        return self.last_weights_path

    def training_loop(self, train_data):
        """

        :param train_data: dict {camera: (B, 34),
                                 src_rgbs_clean: (B, N, H, W, 3),
                                 src_rgbs: (B, N, H, W, 3),
                                 src_cameras: (B, N, 34),
                                 depth_range: (1, 2),
                                 sigma_estimate: (B, N, H, W, 3),
                                 white_level: (1, 1),
                                 rgb_clean: (B, H, W, 3), rgb: (B, H, W, 3),
                                 gt_depth: ,
                                 rgb_path: list(B)}
        :return:
        """
        # Create object that generate and sample rays
        ray_sampler = RaySampler(train_data, self.device)
        N_rand = int(1.0 * self.args.N_rand * self.args.num_source_views / train_data['src_rgbs'][0].shape[0])

        # Sample subset (batch) of rays for the training loop
        ray_batch = ray_sampler.random_ray_batch(N_rand,
                                                 sample_mode=self.args.sample_mode,
                                                 center_ratio=self.args.center_ratio,
                                                 clean=self.args.sup_clean)
        # Calculate the feature maps of all views.
        # This step is seperated because in evaluation time we want to do it once for each image.
        org_src_rgbs = ray_sampler.src_rgbs.to(self.device)
        proc_src_rgbs, featmaps = self.ray_render.calc_featmaps(src_rgbs=org_src_rgbs)

        # Render the rgb values of the pixels that were sampled
        batch_out = self.ray_render.render_batch(ray_batch=ray_batch, proc_src_rgbs=proc_src_rgbs, featmaps=featmaps,
                                                 org_src_rgbs=org_src_rgbs,
                                                 sigma_estimate=ray_sampler.sigma_estimate.to(self.device))

        # compute loss
        self.model.optimizer.zero_grad()
        loss = self.criterion(batch_out['coarse'], ray_batch, self.scalars_to_log)

        if batch_out['fine'] is not None:
            loss += self.criterion(batch_out['fine'], ray_batch, self.scalars_to_log)

        loss.backward()
        self.scalars_to_log['loss'] = loss.item()
        self.model.optimizer.step()
        self.model.scheduler.step()

        self.scalars_to_log['lr_features'] = self.model.scheduler.get_last_lr()[0]
        self.scalars_to_log['lr_mlp'] = self.model.scheduler.get_last_lr()[1]

        return batch_out, ray_batch

    def logging(self, train_data, ray_batch_in, ray_batch_out, dt, global_step, epoch, max_keep=3):
        if self.args.local_rank == 0:
            # log iteration values
            if global_step % self.args.i_tb == 0 or global_step < 10:
                self.log_iteration(ray_batch_out, ray_batch_in, dt, global_step, epoch)

            # save weights
            if global_step % self.args.i_weights == 0:
                print(f"Saving checkpoints at {global_step} to {self.exp_out_dir}...")
                self.last_weights_path = self.exp_out_dir / f"model_{global_step:06d}.pth"
                self.model.save_model(self.last_weights_path)
                files = sorted(self.exp_out_dir.glob("*.pth"), key=os.path.getctime)
                rm_files = files[0:max(0, len(files) - max_keep)]
                for f in rm_files:
                    f.unlink()

            # log images of training and validation
            if global_step % self.args.i_img == 0 or global_step == self.model.start_step + 1:
                self.log_images(train_data, global_step)

    def log_iteration(self, ray_batch_out, ray_batch_in, dt, global_step, epoch):
        # write mse and psnr stats
        mse_error = l2_loss(de_linearize(ray_batch_out['coarse'].rgb, ray_batch_in['white_level']),
                            de_linearize(ray_batch_in['rgb'], ray_batch_in['white_level'])).item()
        self.scalars_to_log['train/coarse-loss'] = mse_error
        self.scalars_to_log['train/coarse-psnr-training-batch'] = mse2psnr(mse_error)
        if ray_batch_out['fine'] is not None:
            mse_error = l2_loss(de_linearize(ray_batch_out['fine'].rgb, ray_batch_in['white_level']),
                                de_linearize(ray_batch_in['rgb'], ray_batch_in['white_level'])).item()
            self.scalars_to_log['train/fine-loss'] = mse_error
            self.scalars_to_log['train/fine-psnr-training-batch'] = mse2psnr(mse_error)

        logstr = f"{self.args.expname} Epoch: {epoch}  step: {global_step} "
        for k in self.scalars_to_log.keys():
            logstr += f" {k}: {self.scalars_to_log[k]:.6f}"
            self.writer.add_scalar(k, self.scalars_to_log[k], global_step)
        if global_step % self.args.i_print == 0:
            print(logstr)
            print(f"each iter time {dt:.05f} seconds")

    def log_view_to_tb(self, global_step, ray_sampler, gt_img, render_stride=1, prefix=''):
        self.model.switch_to_eval()
        with torch.no_grad():
            ret = render_single_image(ray_sampler=ray_sampler, model=self.model, args=self.args)

        average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))

        if self.args.render_stride != 1:
            gt_img = gt_img[::render_stride, ::render_stride]
            average_im = average_im[::render_stride, ::render_stride]

        rgb_gt = img_HWC2CHW(gt_img)
        average_im = img_HWC2CHW(average_im)

        rgb_pred = img_HWC2CHW(ret['coarse'].rgb.detach().cpu())

        h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
        w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
        rgb_im = torch.zeros(3, h_max, 3 * w_max)
        rgb_im[:, :average_im.shape[-2], :average_im.shape[-1]] = average_im
        rgb_im[:, :rgb_gt.shape[-2], w_max:w_max + rgb_gt.shape[-1]] = rgb_gt
        rgb_im[:, :rgb_pred.shape[-2], 2 * w_max:2 * w_max + rgb_pred.shape[-1]] = rgb_pred

        depth_im = ret['coarse'].depth.detach().cpu()
        acc_map = torch.sum(ret['coarse'].weights, dim=-1).detach().cpu()

        if ret['fine'] is None:
            depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
            acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))
        else:
            rgb_fine = img_HWC2CHW(ret['fine'].rgb.detach().cpu())
            rgb_fine_ = torch.zeros(3, h_max, w_max)
            rgb_fine_[:, :rgb_fine.shape[-2], :rgb_fine.shape[-1]] = rgb_fine
            rgb_im = torch.cat((rgb_im, rgb_fine_), dim=-1)
            rgb_im = rgb_im.clamp(min=0., max=1.)
            rgb_im = de_linearize(rgb_im, ray_sampler.white_level)
            depth_im = torch.cat((depth_im, ret['fine'].depth.detach().cpu()), dim=-1)
            depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
            acc_map = torch.cat((acc_map, torch.sum(ret['fine'].weights, dim=-1).detach().cpu()), dim=-1)
            acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))

        # write the pred/gt rgb images and depths
        self.writer.add_image(prefix + 'rgb_gt-coarse-fine', rgb_im, global_step)
        self.writer.add_image(prefix + 'depth_gt-coarse-fine', depth_im, global_step)
        self.writer.add_image(prefix + 'acc-coarse-fine', acc_map, global_step)

        # write scalar
        pred_rgb = ret['fine'].rgb if ret['fine'] is not None else ret['coarse'].rgb
        psnr_curr_img = img2psnr(de_linearize(pred_rgb.detach().cpu(), ray_sampler.white_level),
                                 de_linearize(gt_img, ray_sampler.white_level))
        self.writer.add_scalar(prefix + 'psnr_image', psnr_curr_img, global_step)

        self.model.switch_to_train()

    def log_images(self, train_data, global_step):
        print('Logging a random validation view...')
        val_data = next(self.val_loader_iterator)
        tmp_ray_sampler = RaySampler(val_data, self.device, render_stride=self.args.render_stride)
        H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
        gt_img = tmp_ray_sampler.rgb_clean.reshape(H, W, 3)
        self.log_view_to_tb(global_step, tmp_ray_sampler, gt_img, render_stride=self.args.render_stride, prefix='val/')
        torch.cuda.empty_cache()

        print('Logging current training view...')
        tmp_ray_train_sampler = RaySampler(train_data, self.device,
                                           render_stride=self.args.render_stride)
        H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
        gt_img = tmp_ray_train_sampler.rgb_clean.reshape(H, W, 3)
        self.log_view_to_tb(global_step, tmp_ray_train_sampler, gt_img, render_stride=self.args.render_stride, prefix='train/')


