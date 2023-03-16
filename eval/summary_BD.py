import numpy as np
from configs.experiment_setting import DEFAULT_GAIN_LIST, BurstDenoising, LS_DICT, COLORS_DICT, STEP_DICT
from eval.eval_utils import table_width, get_float_fmt, get_string_fmt, get_separator
from configs.local_setting import LOG_DIR, FIG_DIR
from nan.utils.io_utils import print_link
from visualizing.plotting import *

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
        for res in exp_dir.glob(f"*/*{STEP_DICT[name]}.npy"):
            if not res.stem.startswith(("with_depth_" + npy_name, npy_name)):
                continue
            if res.stem.startswith(npy_name) and (exp_dir / ("with_depth_" + res.name)).exists():
                continue
            result_summary.append(np.load(res))

        if data.startswith('llff_test'):
            if len(result_summary) != 8:
                print(len(result_summary), print_link(exp_dir, ""))
                continue
        else:
            raise NotImplementedError

        mean_result_exp = np.concatenate(result_summary).mean(0)[:8]
        mean_result_exp = mean_result_exp[4:]
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


# if __name__ == '__main__':
#     mean_results = summary_multi_gains({'reproduce_NAN': ('reproduce_NAN', '')})
metric_titles = [r"PSNR$\uparrow$", r"SSIM$\uparrow$", r"LPIPS$\downarrow$"]


def all_figure(mean_results_np, training_dict, gains, rearrange_legend=None):
    psnr, ssim, lpips, depth = mean_results_np
    fig, ax = plt.subplots(ncols=3, figsize=(11, 3))
    assert len(psnr) == len(training_dict)

    for psnr_exp, ssim_exp, lpips_exp, exp_name in zip(psnr, ssim, lpips, training_dict):
        ax[0].plot(gains, psnr_exp, label=exp_name, color=COLORS_DICT[exp_name], linestyle=LS_DICT[exp_name])
        ax[1].plot(gains, ssim_exp, label=exp_name, color=COLORS_DICT[exp_name], linestyle=LS_DICT[exp_name])
        ax[2].plot(gains, lpips_exp, label=exp_name, color=COLORS_DICT[exp_name], linestyle=LS_DICT[exp_name])

    for a, title in zip(ax, metric_titles):
        a.set_title(title)
        a.set_xlabel("gain")
        a.set_xticks(gains)
        a.grid(True)

    handles, labels = ax[1].get_legend_handles_labels()
    # sort both labels and handles by labels
    if rearrange_legend is not None:
        handles = [handles[i] for i in rearrange_legend]
        labels = [labels[i] for i in rearrange_legend]
    plt.legend(handles, labels, bbox_to_anchor=(1.04, 1), frameon=False)

    plt.subplots_adjust(top=0.89,
                        bottom=0.215,
                        left=0.042,
                        right=0.82,
                        hspace=0.26,
                        wspace=0.275)

    return fig, ax


def summary(training_dict, gains, fig_name, rearrange_legend=None):
    if gains is None:
        gains = DEFAULT_GAIN_LIST
    mean_results = [summary_multi_gains(training_dict, gains=gains)]
    mean_results = [sum(e, []) for e in zip(*mean_results)]
    mean_results_np = np.array(mean_results).transpose((2, 1, 0))

    all_figure(mean_results_np, training_dict, gains, rearrange_legend=rearrange_legend)
    plt.savefig(FIG_DIR / "metrics" / f"{fig_name}.pdf")
    plt.show()


if __name__ == '__main__':
    summary(BurstDenoising.as_dict(), DEFAULT_GAIN_LIST, 'BurstDenoising_8_views')
