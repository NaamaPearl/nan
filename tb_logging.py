import torch

from nan.dataloaders.basic_dataset import process_fn
from nan.losses import l2_loss
from nan.utils.eval_utils import mse2psnr, img2psnr
from nan.utils.general_utils import img_HWC2CHW
from nan.render_image import render_single_image
from nan.sample_ray import RaySampler
from nan.utils.io_utils import colorize


def log_view_to_tb(writer, global_step, args, model, ray_sampler, gt_img, render_stride=1, prefix=''):
    model.switch_to_eval()
    with torch.no_grad():
        ret = render_single_image(ray_sampler=ray_sampler, model=model, args=args)

    average_im = ray_sampler.src_rgbs.cpu().mean()

    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        average_im = average_im[::render_stride, ::render_stride]

    rgb_gt = img_HWC2CHW(gt_img)
    average_im = img_HWC2CHW(average_im)

    rgb_pred = img_HWC2CHW(ret['coarse'].rgb.detach().cpu())

    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
    rgb_im = torch.zeros(3, h_max, 3*w_max)
    rgb_im[:, :average_im.shape[-2], :average_im.shape[-1]] = average_im
    rgb_im[:, :rgb_gt.shape[-2], w_max:w_max+rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, :rgb_pred.shape[-2], 2*w_max:2*w_max+rgb_pred.shape[-1]] = rgb_pred

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
        rgb_im = process_fn(rgb_im, ray_sampler.white_level)
        depth_im = torch.cat((depth_im, ret['fine'].depth.detach().cpu()), dim=-1)
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
        acc_map = torch.cat((acc_map, torch.sum(ret['fine'].weights, dim=-1).detach().cpu()), dim=-1)
        acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))

    # write the pred/gt rgb images and depths
    writer.add_image(prefix + 'rgb_gt-coarse-fine', rgb_im, global_step)
    writer.add_image(prefix + 'depth_gt-coarse-fine', depth_im, global_step)
    writer.add_image(prefix + 'acc-coarse-fine', acc_map, global_step)

    # write scalar
    pred_rgb = ret['fine'].rgb if ret['fine'] is not None else ret['coarse'].rgb
    psnr_curr_img = img2psnr(process_fn(pred_rgb.detach().cpu(), ray_sampler.white_level),
                             process_fn(gt_img, ray_sampler.white_level))
    writer.add_scalar(prefix + 'psnr_image', psnr_curr_img, global_step)

    model.switch_to_train()


def log_iteration(ret, ray_batch, writer, scalars_to_log, dt, global_step, epoch, args):
    # write mse and psnr stats
    mse_error = l2_loss(process_fn(ret['coarse'].rgb, ray_batch['white_level']),
                        process_fn(ray_batch['rgb'], ray_batch['white_level'])).item()
    scalars_to_log['train/coarse-loss'] = mse_error
    scalars_to_log['train/coarse-psnr-training-batch'] = mse2psnr(mse_error)
    if ret['fine'] is not None:
        mse_error = l2_loss(process_fn(ret['fine'].rgb, ray_batch['white_level']),
                            process_fn(ray_batch['rgb'], ray_batch['white_level'])).item()
        scalars_to_log['train/fine-loss'] = mse_error
        scalars_to_log['train/fine-psnr-training-batch'] = mse2psnr(mse_error)

    logstr = f"{args.expname} Epoch: {epoch}  step: {global_step} "
    for k in scalars_to_log.keys():
        logstr += f" {k}: {scalars_to_log[k]:.6f}"
        writer.add_scalar(k, scalars_to_log[k], global_step)
    if global_step % args.i_print == 0:
        print(logstr)
        print(f"each iter time {dt:.05f} seconds")


def log_images(train_data, model, val_loader_iterator, writer, global_step, args, device):
    print('Logging a random validation view...')
    val_data = next(val_loader_iterator)
    tmp_ray_sampler = RaySampler(val_data, device, render_stride=args.render_stride)
    H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
    gt_img = tmp_ray_sampler.rgb_clean.reshape(H, W, 3)
    log_view_to_tb(writer, global_step, args, model, tmp_ray_sampler, gt_img,
                   render_stride=args.render_stride, prefix='val/')
    torch.cuda.empty_cache()

    print('Logging current training view...')
    tmp_ray_train_sampler = RaySampler(train_data, device,
                                       render_stride=args.render_stride)
    H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
    gt_img = tmp_ray_train_sampler.rgb_clean.reshape(H, W, 3)
    log_view_to_tb(writer, global_step, args, model, tmp_ray_train_sampler, gt_img,
                   render_stride=args.render_stride, prefix='train/')
