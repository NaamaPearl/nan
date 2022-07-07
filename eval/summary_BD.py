import numpy as np

from configs.global_setting import DEFAULT_GAIN_LIST
from eval.eval_utils import table_width, get_float_fmt, get_string_fmt, get_separator
from configs.local_setting import LOG_DIR, FIG_DIR
from nan.utils.io_utils import print_link

table_column = 3
float_fmt = get_float_fmt(table_width, table_column)
string_fmt = get_string_fmt(table_width, table_column)
separator = get_separator(string_fmt, table_width, table_column)


def print_exp(label, name, mean_result_exp):
    print(f"{label}, {STEP_DICT[name]}")
    print(float_fmt.format("", *(mean_result_exp.tolist())))
    print(separator)


def summary_results_per_gain(training, data, gain, print_fn=print_exp):
    mean_result_per_gain = []
    for i, (label, (name, post, eval_expname_fmt)) in enumerate(training.items()):
        exp_dir = LOG_DIR / name / data / eval_expname_fmt.format(gain=gain)
        result_summary = []
        npy_name = "psnr"
        if name not in ['deeprep', 'bpn']:
            npy_name = post + npy_name
        for res in exp_dir.glob(f"*{STEP_DICT[name]}.npy"):
            if not res.stem.startswith(("with_depth_" + npy_name, npy_name)):
                continue
            if res.stem.startswith(npy_name) and (exp_dir / ("with_depth_" + res.name)).exists():
                continue
            result_summary.append(np.load(res))

        if data == 'llff_test':
            if len(result_summary) != 8:
                print(len(result_summary), print_link(exp_dir, ""))
                continue
        else:
            raise NotImplemented

        mean_result_exp = np.concatenate(result_summary).mean(0)[:8]
        mean_result_exp = mean_result_exp[1::2]
        mean_result_per_gain.append(mean_result_exp)

        print_fn(label, name, mean_result_exp)
    return mean_result_per_gain


def summary_multi_gains(training, gains=None, eval_data='llff_test', print_fn=print_exp):
    if gains is None:
        gains = DEFAULT_GAIN_LIST
    mean_results = []
    for gain in gains:
        print(f"**************** gain {gain} *****************************")
        mean_results.append(
            summary_results_per_gain(training, eval_data, gain=gain, print_fn=print_fn))
        print(f"***************************************************")
        print("\n")
        print("\n")
    return mean_results



