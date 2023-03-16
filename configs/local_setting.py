import platform
from pathlib import Path


class LocalSettingDict(dict):
    def __missing__(self, key):
        self[key] = Path(__file__).parent.parent
        return self[key]


# local setting
PC_NAME = platform.uname()[1]
print(f"[*] Running on {PC_NAME}")

# root dir
ROOT_DIR_DICT = LocalSettingDict()

# config
CONFIG_TRAIN_DICT = LocalSettingDict()
CONFIG_TRAIN_DICT[PC_NAME] = 'train.yml'

CONFIG_EVAL_DICT = LocalSettingDict()
CONFIG_EVAL_DICT[PC_NAME] = 'eval.yml'

ROOT_DIR     = ROOT_DIR_DICT[PC_NAME]
OUT_DIR      = ROOT_DIR / 'out'
LOG_DIR      = ROOT_DIR / 'logs'
DATA_DIR     = ROOT_DIR.parent / 'data'
FIG_DIR      = ROOT_DIR / 'figures'

FIG_DIR.mkdir(exist_ok=True)
(FIG_DIR / 'metrics').mkdir(exist_ok=True)

TRAIN_CONFIG = ROOT_DIR / 'configs' / CONFIG_TRAIN_DICT[PC_NAME]
EVAL_CONFIG  = ROOT_DIR / 'configs' / CONFIG_EVAL_DICT[PC_NAME]
VIDEO_CONFIG  = ROOT_DIR / 'configs' / 'render_videos.yml'

if __name__ == '__main__':
    print(ROOT_DIR)

