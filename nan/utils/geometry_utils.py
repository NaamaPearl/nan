import kornia
import torch
import torch.nn.functional as F


def get_grid(H, W, device):
    return kornia.create_meshgrid(H, W, normalized_coordinates=False, device=device)  # (1, H, W, 2)


def get_homogenous_grid(H, W, device):
    grid = get_grid(H, W, device)
    grid = torch.cat((grid, torch.ones_like(grid[..., :1])), dim=-1)  # (1, H, W, 3)
    return grid


def parse_Rts(Rts):
    R = Rts[:, :, :, :3]
    t = Rts[:, :, :, 3:]

    return R, t


def u2w_yuval(u, K, Rts):
    """
    $u = \alpha + \beta * w$
    :param u:
    :param K:
    :param Rts:
    :return:
    """
    B, V, H, W, C = u.shape
    assert C == 2
    device = u.device

    ux = u[..., 0]
    uy = u[..., 1]

    fx = K[..., 0, 0]
    fy = K[..., 1, 1]

    cx = K[..., 0, 2]
    cy = K[..., 1, 2]

    R, t = parse_Rts(Rts)
    r11, r12, r13 = R[..., 0, :][0][0]
    r21, r22, r23 = R[..., 1, :][0][0]
    r31, r32, r33 = R[..., 2, :][0][0]

    tx, ty, tz = t[0][0][:3]

    ref_grid = kornia.create_meshgrid(H, W, normalized_coordinates=False, device=device)  # (1, H, W, 2)
    x = ref_grid[..., 0]
    y = ref_grid[..., 1]

    x_bar = x - cx
    y_bar = y - cy

    alphax = fx * ( ((r11 - 1) * x_bar / fx)
                    + (r12 * y_bar / fy)
                    + r13
                    - (r31 * x_bar ** 2 / fx ** 2)
                    - (r32 * x_bar * y_bar / (fx * fy))
                    - ((r33 - 1) * x_bar / fx))

    betax = fx * tx - tz * x_bar

    alphay = fy * ( (r21 * x_bar / fx)
                    + ((r22 - 1) * y_bar / fy)
                    + r23
                    - (r31 * x_bar * y_bar / (fx * fy))
                    - (r32 * y_bar ** 2 / fy ** 2)
                    - ((r33 - 1) * y_bar / fy))

    betay = fy * ty - tz * y_bar

    wx = ((ux - alphax) / betax).clamp(min=0)
    wy = ((uy - alphay) / betay).clamp(min=0)
    w = torch.stack((wx, wy), dim=-1)
    return w


def u2w(u, Hs):
    """
    grid = R * xy0 + t * w
    u = grid - xy0 = (R - I) * xy0 + t * w
    w = (u - (R - I) * xy0) / t
    :param u:
    :param Hs:
    :return:
    """
    device = u.device
    B, V, H, W, C = u.shape
    assert C == 2
    u = u.clone()
    # u[..., 0] *= -1
    # u[..., 1] *= -1
    u0 = torch.cat((u, torch.zeros_like(u[..., :1])), dim=-1)  # (B, V, H, W, 3)

    R_tilde, t_tilde = parse_Rts(Hs)  # (B, V, 3, 3), (B, V, 3, 1)
    __R3__ = R_tilde[..., [2], :]  # (B, V, 1, 3)
    t_tilde = t_tilde[..., 0]  # (B, V, 3)
    t3 = t_tilde[..., 2]  # (B, V)
    xy1 = get_homogenous_grid(H, W, device)[0]  # (H, W, 3)
    gamma = torch.einsum("bvij,hwj->bvhw", __R3__, xy1)  # (B, V, H, W)

    gammaI = gamma.unsqueeze(-1).unsqueeze(-1) * torch.eye(3)  # (B, V, H, W, 3, 3)
    R_minus_I = R_tilde - gammaI
    alpha = torch.einsum("bvhwij,hwj->bvhwi", R_minus_I, xy1)  # (B, V, H, W, 3)
    beta = (t3 * (u0 + xy1) - t_tilde)
    w = (alpha - gamma.unsqueeze(-1) * u0) / beta
    w = w.clamp(min=0)
    return w


def normalize_grid_(grid, W, H):
    grid[..., 0] = 2 * grid[..., 0] / (W - 1) - 1  # scale to [-1, 1]
    grid[..., 1] = 2 * grid[..., 1] / (H - 1) - 1  # scale to [-1, 1]
    return grid


def warp_H(src_feat, Hs, w):
    """
    src_feat: (V, C, H, W)
    pre_Hs: (V, 3, 4) equal to "src_proj @ ref_proj_inv"
    depth_values: (V, D, H, W)
    out: (V, C, D, H, W)
    """
    B, V, C, H, W = src_feat.shape
    device = src_feat.device

    w = w.view((B, 1, 1, H, W, 1))

    R_tilde, t_tilde = parse_Rts(Hs)            # (B, V, 3, 3)  # R_tilde = K R K^{-1}
    t_tilde = t_tilde.view((B, V, 1, 1, 1, 3))  # (B, V, 3, 1)  # t_tilde = K t

    # create grid from the ref frame
    xy1_0 = get_homogenous_grid(H, W, device)

    xy1_1 = torch.einsum("bvij,chwj->bvchwi", R_tilde, xy1_0) + t_tilde * w

    # project negative w_dict pixels to somewhere outside the image
    negative_depth_mask = xy1_1[..., 2:] <= 1e-7
    xy1_1[..., 0:1][negative_depth_mask] = W
    xy1_1[..., 1:2][negative_depth_mask] = H
    xy1_1[..., 2:3][negative_depth_mask] = 1

    xy1_1 = xy1_1[..., :2] / xy1_1[..., 2:]  # divide by w (B, V, C, H, W, 2)
    normalize_grid_(xy1_1, W, H)

    warped_src_feat = F.grid_sample(src_feat.reshape((B * V, C, H, W)), xy1_1.reshape((B * V, H, W, 2)),
                                    mode='bicubic', padding_mode='zeros',
                                    align_corners=True)
    warped_src_feat = warped_src_feat.view((B, V, C, H, W))
    valid_mask = warped_src_feat > 0

    return warped_src_feat, valid_mask


def torch_warp_flow(src_feat, u):
    B, V, C, H, W = src_feat.shape
    device = src_feat.device
    ref_grid = get_grid(H, W, device)
    src_grid = ref_grid + u
    normalize_grid_(src_grid, W, H)

    warped_src_feat = F.grid_sample(src_feat.reshape((B * V, C, H, W)), src_grid.reshape((B * V, H, W, 2)),
                                    mode='bicubic', padding_mode='zeros',
                                    align_corners=True)
    warped_src_feat = warped_src_feat.view((B, V, C, H, W))
    valid_mask = warped_src_feat > 0

    return warped_src_feat, valid_mask


def warp_H_wrapper(src_feat, Ms, depth):
    src_feat_warped, _ = warp_H(src_feat[:, 1:], Ms, depth)
    src_feat_warped = torch.cat([src_feat[:, [0]], src_feat_warped], dim=1)
    return src_feat_warped


def split_to_views(M, V):
    return torch.split(M, [1, V - 1], dim=1)


def build_H(Ks, Rts, levels=False):
    if levels:
        B, V, L, _, _ = Ks.shape
        proj_shape = (B, V - 1, L, 4, 4)
        Rts = Rts.unsqueeze(2)
    else:
        B, V, _, _ = Ks.shape
        proj_shape = (B, V - 1, 4, 4)

    Rt0, Rti = split_to_views(Rts, V)
    K0,  Ki  = split_to_views(Ks[..., :3, :3], V)

    M0 = init_proj_mat(proj_shape, Rts)
    M0[..., :3, :] = K0 @ Rt0[..., :3, :]  # K * [R0 | t0]
    M0_inv = torch.inverse(M0)

    Mi = init_proj_mat(proj_shape, Rts)
    Mi[..., :3, :] = Ki @ Rti[..., :3, :]

    H = (Mi @ M0_inv)[..., :3, :]  # K * [Ri | ti] * (K * [R0 | t0])^-1
    return H


def init_proj_mat(shape, vec_like):
    ref_proj = torch.zeros(shape, dtype=vec_like.dtype, device=vec_like.device)
    ref_proj[..., 3, 3] = 1
    return ref_proj


def warp_KRt(src_feat, Ks, Rts, w):
    """
    src_feat: (B, V-1, C, H, W)
    pre_Hs: (B, V-1, 3, 4) equal to "src_proj @ ref_proj_inv"
    depth_values: (B, H, W)
    out: (V, C, D, H, W)
    :type Ks: (B, V, 3, 3)
    :type Rts: (B, V, 4, 4)
    """
    B, _, _, H, W = src_feat.shape
    w = w.view((B, 1, 1, H, W, 1))
    Hs = build_H(Ks, Rts)
    return warp_H(src_feat, Hs, w)


def warp_KRt_wrapper(src_feat, Ks, Rts, w):
    src_feat_warped, _ = warp_KRt(src_feat[:, 1:], Ks, Rts, w)
    src_feat_warped = torch.cat([src_feat[:, [0]], src_feat_warped], dim=1)
    return src_feat_warped


def euler2mat(angles):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angles: rotation angle along 3 axis (in radians) -- size = [B, V-1, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, V-1, 3, 3]
    """
    B, V, _ = angles.shape
    x, y, z = angles[..., 0], angles[..., 1], angles[..., 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = torch.zeros_like(z)
    ones = torch.ones_like(z)
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=-1).reshape(B, V, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=-1).reshape(B, V, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=-1).reshape(B, V, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach() * 0 + 1, quat], dim=1)
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2Rts(vec, L, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, V-1, 6]
    Returns:
        A transformation matrix -- [B, V, 4, 4] (including reference image, [I | 0])
    """
    B, V, _ = vec.shape

    t = vec[..., 3:]  # [B, V-1, 3, 1]

    eul = vec[..., :3]
    R = euler2mat(eul)  # [B, V-1, 3, 3]

    Rts = init_proj_mat((B, V+1, 4, 4), vec)
    Rts[..., 0, :3, :3] = torch.eye(3)
    Rts[..., 1:, :3, :3]    = R
    Rts[..., 1:, :3,  3]    = t
    return Rts


def pose2Hs(pose, Ks):
    """
    convert pose vector (6 DoF) + intrinsic matrix K to homography matrix
    :param pose:
    :param Ks:
    :return:
    """
    L = Ks.shape[2]
    Rts = pose_vec2Rts(pose, L)
    return build_H(Ks, Rts, levels=True), Rts


if __name__ == '__main__':
    # P = torch.rand((5, 4, 3, 4))  # (B, V, 3, 4)
    # R = P[:, :, :, :3]  # (B, V, 3, 3)
    # grid = torch.rand(1, 100, 200, 3)  # (1, H, W, 3)
    # w_dict = torch.rand((5, 100, 200))
    # print(f"R shape: {R.shape}")
    # print(f"grid shape: {grid.shape}")
    #
    # res = torch.einsum("bvij,chwj->bvchwi", R, grid)
    # # res = torch.einsum("bvij,jhw->bvihw", R, grid)
    # print(f"res shape: {res.shape}")
    # print(f"single res: {R[2, 3] @ grid[:, 0, 1, :].T}")
    # print(f"einsum res: {res[2, 3, :, 0, 1, :]}")
    #
    # feat = torch.rand((5, 4, 1, 100, 200))
    # homo_warp_with_proj_mat(feat, P, w_dict)
    pass