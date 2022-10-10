from collections import defaultdict
from configs.config import CustomArgumentParser

DEFAULT_GAIN_LIST = [1, 2, 4, 8, 16, 20]

LLFF_SCENES_LIST = [
    'fern',
    'orchids',
    'flower',
    'horns',
    'leaves',
    'room',
    'trex',
    'fortress'
]

STEP_DICT = defaultdict(lambda: 255000)


class GeneralAblationTable:
    titles = ['PreNet', 'Views attention', '3D kernels', 'Noise parameters', 'Loss L1']
    rows = []
    base_args = ['--include_target', '--std', str(-3), str(-0.5), str(-2), str(-0.5), '--expname', 'reproduce',
                 '--losses', 'l2', '--losses_weights', str(1.)]
    loss_l1 = ['--losses', 'l1', '--losses_weights', str(1)]
    expand = ['--expand_rgb', '--kernel_size', str(3), str(3), '--rgb_weights']
    attn = ['--views_attn']
    pre_net = ['--pre_net', '--blend_src']
    noise_params = ['--noise_feat']
    args_list = [pre_net, attn, expand, noise_params, loss_l1]

    def __init__(self, rows=None):
        if rows is not None:
            self.__class__.rows = rows

    default_eval_expname_fmt = 'same__factor_4__eval_gain_{gain}'

    @classmethod
    def as_dict(cls, verbose=False) -> dict:
        res = {}
        parser = CustomArgumentParser.config_parser()
        for row in cls.rows:
            if type(row) is list and len(row) == len(cls.titles):
                additional_args = [arg for flag, arg in zip(row, cls.args_list) if flag]
                row_args = parser.parse_args(args=cls.base_args + [a for sublist in additional_args for a in sublist])
                initial = '  '.join(map(lambda a: ' + ' if a else '   ', row[:-1]))
                res[initial] = (row_args.expname, '', cls.default_eval_expname_fmt)
            else:
                if len(row[1]) == 2:
                    res[row[0]] = (*row[1], cls.default_eval_expname_fmt)
                elif len(row[1]) == 3:
                    res[row[0]] = row[1]
                else:
                    raise IOError(row)

        if verbose:
            for initial, exp in res.items():
                print(f"{initial:<20}, {exp}")
        return res

    @classmethod
    def as_list(cls, verbose=False) -> list:
        res = []
        parser = CustomArgumentParser.config_parser()
        for row in cls.rows:
            if type(row) is list and len(row) == len(cls.titles):
                additional_args = [arg for flag, arg in zip(row, cls.args_list) if flag]
                row_args = parser.parse_args(args=cls.base_args + [a for sublist in additional_args for a in sublist])
                res.append(row_args.expname)
            else:
                res.append(row[1][0])

        if verbose:
            print("******************************************")
            print("************ ablation list ***************")
            print("******************************************")
            for exp in res:
                print(exp)

        return res


class Ablation(GeneralAblationTable):
    rows = [
        # ['IBRNet', ('pretraining____clean__l2', '')],
        # [False, False, False, False, False, ],
        [False, False, False, False, True, ],
        #
        [True, False, False, False, True, ],
        [False, True, False, False, True, ],
        [False, False, True, False, True, ],
        #
        [False, True, True, False, True, ],
        [True, False, True, False, True, ],
        [True, True, False, False, True, ],

        [True, True, True, False, True, ],
        [True, True, True, True, True, ]
    ]


class BurstDenoising(GeneralAblationTable):
    rows = [
        # ['DeepRep', ('deeprep', '')],
        # ['BPN', ('bpn', '')],
        # ['IBRNet-N', ('IBRNet', '')],
        ['NAN', ('reproduce__NAN', '')],
        # ['NAN + bilateral', ('reproduce_NAN', 'bi_0.1_')],
    ]


COLORS_DICT = {
    'DeepRep': 'darkorchid',
    'IBRNet': 'limegreen',
    'BPN': 'darkorange',
    'IBRNet-N': 'lightskyblue',
    'NAN': 'dodgerblue'
}

LS_DICT = {
    'DeepRep': ':',
    'IBRNet': (0, (1, 5)),
    'BPN': '-.',
    'IBRNet-N': '--',
    'NAN': '-'
}