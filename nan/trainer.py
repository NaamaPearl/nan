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
from nan.dataloaders.create_training_dataset import create_training_dataset
from nan.dataloaders.data_utils import cycle
from nan.model import NANScheme
from nan.render_ray import RayRender
from nan.sample_ray import RaySampler
from nan.utils.io_utils import print_link
from tb_logging import log_iteration, log_images


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.local_rank}")
        self.out_exp = OUT_DIR / args.expname
        self.out_exp.mkdir(exist_ok=True, parents=True)
        print_link(self.out_exp, 'outputs will be saved to')

        # save the args and config files
        self.save_ymls(args, sys.argv[1:], self.out_exp)

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
        self.val_dataset = dataset_dict[args.eval_dataset](args, 'validation', scenes=args.eval_scenes)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1)
        self.val_loader_iterator = iter(cycle(self.val_loader))

        # Create IBRNet model
        self.model = NANScheme.create(args)
        self.last_weights_path = None

        # Create raw2output
        self.ray_render = RayRender(model=self.model, args=args, device=self.device)

        # Create criterion
        self.criterion = NANLoss(args)

        tb_dir = LOG_DIR / args.expname
        if args.local_rank == 0:
            self.writer = SummaryWriter(str(tb_dir))
            print_link(tb_dir, 'saving tensorboard files to')

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
        epoch = 0
        scalars_to_log = {}

        while global_step < self.args.n_iters + 1:
            np.random.seed()
            for train_data in self.train_loader:
                time0 = time.time()
                if self.args.distributed:
                    self.train_sampler.set_epoch(epoch)

                # core optimization loop
                batch_out, ray_batch = self.training_loop(train_data, scalars_to_log)
                dt = time.time() - time0

                # Rest is logging
                self.logging(train_data, batch_out, ray_batch, dt, global_step, epoch, scalars_to_log)

                global_step += 1
                if global_step > self.model.start_step + self.args.n_iters + 1:
                    break
            epoch += 1
        return self.last_weights_path

    def training_loop(self, train_data, scalars_to_log):
        # load training rays
        ray_sampler = RaySampler(train_data, self.device)
        N_rand = int(1.0 * self.args.N_rand * self.args.num_source_views / train_data['src_rgbs'][0].shape[0])

        ray_batch = ray_sampler.random_ray_batch(N_rand,
                                                 sample_mode=self.args.sample_mode,
                                                 center_ratio=self.args.center_ratio,
                                                 clean=self.args.sup_clean)

        src_rgbs, featmaps = self.ray_render.calc_featmaps(src_rgbs=ray_sampler.src_rgbs)

        batch_out = self.ray_render.render_batch(ray_batch=ray_batch, src_rgbs=src_rgbs, featmaps=featmaps,
                                                 org_src_rgbs=ray_sampler.src_rgbs.to(self.device),
                                                 sigma_estimate=ray_sampler.sigma_estimate.to(self.device))

        # compute loss
        self.model.optimizer.zero_grad()
        loss = self.criterion(batch_out['coarse'], ray_batch, scalars_to_log)

        if batch_out['fine'] is not None:
            loss += self.criterion(batch_out['fine'], ray_batch, scalars_to_log)

        loss.backward()
        scalars_to_log['loss'] = loss.item()
        self.model.optimizer.step()
        self.model.scheduler.step()

        scalars_to_log['lr_features'] = self.model.scheduler.get_last_lr()[0]
        scalars_to_log['lr_mlp'] = self.model.scheduler.get_last_lr()[1]

        return batch_out, ray_batch

    def logging(self, train_data, ret, ray_batch, dt, global_step, epoch, scalars_to_log, max_keep=3):
        if self.args.local_rank == 0:
            # log iteration values
            if global_step % self.args.i_tb == 0 or global_step < 10:
                log_iteration(ret, ray_batch, self.writer, scalars_to_log, dt, global_step, epoch, self.args)

            # save weights
            if global_step % self.args.i_weights == 0:
                print(f"Saving checkpoints at {global_step} to {self.out_exp}...")
                self.last_weights_path = self.out_exp / f"model_{global_step:06d}.pth"
                self.model.save_model(self.last_weights_path)
                files = sorted(self.out_exp.glob("*.pth"), key=os.path.getctime)
                rm_files = files[0:max(0, len(files) - max_keep)]
                for f in rm_files:
                    f.unlink()

            # log images of training and validation
            if global_step % self.args.i_img == 0 or global_step == self.model.start_step + 1:
                log_images(train_data, self.model, self.val_loader_iterator, self.writer, global_step,
                           self.args, self.device)
