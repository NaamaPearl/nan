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

from tqdm import tqdm

from nan.render_ray import RayRender
from nan.raw2output import RaysOutput
from nan.sample_ray import RaySampler


def render_single_image(ray_sampler: RaySampler,
                        model,
                        args,
                        save_pixel=None) -> Dict[str, RaysOutput]:
    """
    :param: save_pixel:
    :param: featmaps:
    :param: render_stride:
    :param: white_bkgd:
    :param: det:
    :param: ret_output:
    :param: projector:
    :param: ray_batch:
    :param: ray_sampler: RaySamplingSingleImage for this view
    :param: model:  {'net_coarse': , 'net_fine': , ...}
    :param: chunk_size: number of rays in a chunk
    :param: N_samples: samples along each ray (for both coarse and fine model)
    :param: inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param: N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :return: {'coarse': {'rgb': numpy, 'depth': numpy, ...}, 'fine': {}}
    """
    device = torch.device(f'cuda:{args.local_rank}')
    ray_render = RayRender(model=model, args=args, device=device, save_pixel=save_pixel)
    src_rgbs, featmaps = ray_render.calc_featmaps(ray_sampler.src_rgbs.to(device))

    all_ret = OrderedDict([('coarse', RaysOutput.empty_ret()),
                           ('fine', None)])
    if args.N_importance > 0:
        all_ret['fine'] = RaysOutput.empty_ret()
    N_rays = ray_sampler.rays_o.shape[0]

    for i in tqdm(range(0, N_rays, args.chunk_size)):
        # print('batch', i)
        ray_batch = ray_sampler.specific_ray_batch(slice(i, i + args.chunk_size, 1), clean=args.sup_clean)
        ret       = ray_render.render_batch(ray_batch=ray_batch,
                                            proc_src_rgbs=src_rgbs,
                                            featmaps=featmaps,
                                            org_src_rgbs=ray_sampler.src_rgbs.to(device),
                                            sigma_estimate=ray_sampler.sigma_estimate.to(device))

        all_ret['coarse'].append(ret['coarse'])
        if ret['fine'] is not None:
            all_ret['fine'].append(ret['fine'])
        # torch.cuda.empty_cache()

    # merge chunk results and reshape
    out_shape = torch.empty(ray_sampler.H, ray_sampler.W)[::args.render_stride, ::args.render_stride].shape
    all_ret['coarse'].merge(out_shape)
    if all_ret['fine'] is not None:
        all_ret['fine'].merge(out_shape)

    return all_ret



