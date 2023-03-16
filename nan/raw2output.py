import torch


def unsqueeze_like(x, y):
    return x.view(x.shape + (1,) * (y.dim() - x.dim()))


class RaysOutput:
    RGB_IDX = slice(3)
    RHO_IDX = 3
    """
    Object that aggregate the network output along the rays.
    """
    def __init__(self, rgb_map, depth_map, weights=None, mask=None, alpha=None, z_vals=None, rho=None,
                 debug=None):
        self.rgb = rgb_map
        self.depth = depth_map
        self.weights = weights  # used for importance sampling of fine samples
        self.mask = mask
        self.alpha = alpha
        self.z_vals = z_vals
        self.rho = rho
        self.debug = debug

    @classmethod
    def raw2output(cls, rgb, rho, z_vals, pts_mask, white_bkgd=False):
        """
        :param rgb: rgb network output; tensor of shape [R, S, 3]
        :param rho: rho network output; tensor of shape [R, S, 1]
        :param z_vals: depth of point samples along rays; tensor of shape [R, S]
        :param pts_mask: [R, S, N, 1]
        :param white_bkgd: Used to handle scenes with white background
        :return: RaysOutput object: rgb: [N_rays, 3], depth: [N_rays,], weights: [N_rays,]
        """

        # IBRNet note: we did not use the intervals here, because in practice different scenes from COLMAP can have
        # very different scales, and using interval can affect the model's generalization ability.
        # Therefor we don't use the intervals for both training and evaluation.

        def rho2alpha(rho_, dist):
            return 1. - torch.exp(-rho_)

        # point samples are ordered with increasing depth
        # interval between samples
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat((dists, dists[:, -1:]), dim=-1)  # [N_rays, N_samples]

        rho = rho.view(rho.shape[:2])
        alpha = rho2alpha(rho, dists) + 1e-10  # [N_rays, N_samples]

        # Eq. (3): T
        T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[:, :-1]  # [N_rays, N_samples-1]
        T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]

        # maths show weights, and summation of weights along a ray, are always inside [0, 1]
        weights = alpha * T  # [N_rays, N_samples]
        rgb_map = torch.sum(unsqueeze_like(weights, rgb) * rgb, dim=1)  # [N_rays, 3]

        if white_bkgd:
            rgb_map = rgb_map + (1. - torch.sum(weights, dim=-1, keepdim=True))

        # otherwise don't consider its loss
        depth_map = torch.sum(weights * z_vals, dim=-1)  # [N_rays,]

        # mask calculation
        # Calculate the pixel mask in the target view, based on the mask of points along the ray
        pts_mask = pts_mask[..., 0].sum(
            dim=2) > 1  # [N_rays, N_samples], a 3D points is valid if it is valid for at least 2 of the src views
        pixel_mask = pts_mask.sum(dim=1) > 8  # should at least have 8 valid observation on the ray,

        return cls(rgb_map=rgb_map,
                   depth_map=depth_map,
                   weights=weights,
                   mask=pixel_mask,
                   alpha=alpha,
                   z_vals=z_vals,
                   rho=rho)

    @classmethod
    def parse_raw(cls, raw):
        rgb = raw[:, :, cls.RGB_IDX]  # [N_rays, N_samples, 3]
        rho = raw[:, :, cls.RHO_IDX]  # [N_rays, N_samples]

        return rgb, rho

    @classmethod
    def empty_ret(cls):
        return cls(rgb_map=[], depth_map=[], weights=[], mask=[], alpha=[], z_vals=[], rho=[], debug={})

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


if __name__ == '__main__':
    ret = RaysOutput(1, 2, 3, 4, 5, 6, 7)
