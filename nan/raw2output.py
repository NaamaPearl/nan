import torch

def unsqueeze_like(x, y):
    return x.view(x.shape + (1,) * (y.dim() - x.dim()))

class RaysOutput:
    RGB_IDX   = slice(3)
    SIGMA_IDX = 3

    def __init__(self, rgb_map, depth_map, weights, mask, alpha, z_vals, sigma, rho, debug=None):
        self.rho     = rho
        self.rgb     = rgb_map
        self.depth   = depth_map
        self.weights = weights  # used for importance sampling of fine samples
        self.mask    = mask
        self.alpha   = alpha
        self.z_vals  = z_vals
        self.sigma   = sigma
        self.debug   = debug


    @classmethod
    def raw2output(cls, rgb, sigma, rho, z_vals, mask, white_bkgd=False):
        """
        :param raw: raw network output; tensor of shape [N_rays, N_samples, 4]
        :param z_vals: depth of point samples along rays; tensor of shape [N_rays, N_samples]
        :param ray_d: [N_rays, 3]
        :return: {'rgb': [N_rays, 3], 'depth': [N_rays,], 'weights': [N_rays,], 'depth_std': [N_rays,]}
        """
        # TODO
        # note: we did not use the intervals here, because in practice different scenes from COLMAP can have
        # very different scales, and using interval can affect the model's generalization ability.
        # Therefore we don't use the intervals for both training and evaluation.

        def sigma2alpha(sigma_, dist):
            return 1. - torch.exp(-sigma_)

        # point samples are ordered with increasing depth
        # interval between samples
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat((dists, dists[:, -1:]), dim=-1)  # [N_rays, N_samples]

        sigma = sigma.view(sigma.shape[:2])
        alpha = sigma2alpha(sigma, dists)  + 1e-10  # [N_rays, N_samples]

        # Eq. (3): T
        T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[:, :-1]  # [N_rays, N_samples-1]
        T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]

        # maths show weights, and summation of weights along a ray, are always inside [0, 1]
        weights = alpha * T # [N_rays, N_samples]
        weights[sigma.sum(1) < 1e-3] /= weights[sigma.sum(1) < 1e-3].sum(1, keepdim=True)
        mask = mask.sum(dim=1) > 8  # should at least have 8 valid observation on the ray,
        rgb_map = torch.sum(unsqueeze_like(weights, rgb) * rgb, dim=1)  # [N_rays, 3]

        if rho is not None:
            rho = (rho * weights).sum(1) + 1e-6
        if white_bkgd:
            rgb_map = rgb_map + (1. - torch.sum(weights, dim=-1, keepdim=True))

        # otherwise don't consider its loss
        depth_map = torch.sum(weights * z_vals, dim=-1)  # [N_rays,]

        return cls(rgb_map=rgb_map, depth_map=depth_map, weights=weights, mask=mask, alpha=alpha, z_vals=z_vals, sigma=sigma, rho=rho)

    @classmethod
    def parse_raw(cls, raw):
        rgb   = raw[:, :, cls.RGB_IDX]  # [N_rays, N_samples, 3]
        sigma = raw[:, :, cls.SIGMA_IDX]  # [N_rays, N_samples]

        return rgb, sigma

    @classmethod
    def empty_ret(cls):
        return cls(rgb_map=[], depth_map=[], weights=[], mask=[], alpha=[], z_vals=[], sigma=[], rho=[], debug={})

    def append(self, ret):
        for k, v in ret.__dict__.items():
            if v is None:
                continue
            if type(v) == torch.Tensor:
                self.__getattribute__(k).append(v.cpu())
            else:
                self.__getattribute__(k).append(v)

    def merge(self, shape):
        # merge chunk results and reshape
        for k, v in self.__dict__.items():
            if len(v) in [0, 1]:
                continue
            tmp = torch.cat(v, dim=0).reshape(shape + (-1,))
            self.__setattr__(k, tmp.squeeze())
        # self.rgb[self.mask == 0] = 1.


# class RaysOutputConv(RaysOutput):
#     w = kornia.filters.get_gaussian_kernel2d(kernel_size=(3, 3), sigma=(0.5, 0.5)).cuda().unsqueeze(-1)
#
#     def append(self, ret):
#         for k, v in ret.__dict__.items():
#             if v is None:
#                 continue
#             if type(v) == torch.Tensor:
#                 if k == 'rgb':
#                     if ret.rho is None:
#                         w = self.w
#                     else:
#                         w = self.gaussian(ret.rho)
#                     self.__getattribute__(k).append((v * w).cpu())
#
#                 else:
#                     self.__getattribute__(k).append(v.cpu())
#             else:
#                 self.__getattribute__(k).append(v)
#
#     def merge(self, shape):
#         # merge chunk results and reshape
#         for k, v in self.__dict__.items():
#             if len(v) in [0, 1]:
#                 continue
#             if k == 'rgb':
#                 tmp = torch.cat(v, dim=0).reshape(shape + v[0].shape[1:])
#                 tmp = tmp.permute((4, 2, 3, 0, 1)).reshape((1, 27, -1))
#                 tmp = torch.nn.functional.fold(tmp, shape, (3, 3), padding=(1,1))
#                 tmp = tmp[0].permute((1, 2, 0))
#             else:
#                 tmp = torch.cat(v, dim=0).reshape(shape + (-1,))
#             self.__setattr__(k, tmp.squeeze())
#         # self.rgb[self.mask == 0] = 1.
#
#     @staticmethod
#     def gaussian(rho):
#         window_size = 3
#         x = torch.arange(window_size, device=rho.device, dtype=rho.dtype) - window_size // 2
#         kernel_1d = torch.exp((-x ** 2 / (2 * rho.unsqueeze(-1) ** 2)))
#         kernel_2d = (kernel_1d.unsqueeze(-1) @ kernel_1d.unsqueeze(-2)).unsqueeze(-1)
#
#         return kernel_2d / kernel_2d.sum(dim=(-3, -2), keepdim=True)


def rays_output_factory(args):
    return RaysOutput


if __name__ == '__main__':
    ret = RaysOutput(1, 2, 3, 4, 5, 6, 7)