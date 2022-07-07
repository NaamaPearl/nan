import numpy as np

HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6  # float32 only has 7 decimal digits precision
img_HWC2CHW = lambda x: x.permute(2, 0, 1)
gray2rgb = lambda x: x.unsqueeze(2).repeat(1, 1, 3)
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
rng = np.random.RandomState(234)
_EPS = np.finfo(float).eps * 4.0