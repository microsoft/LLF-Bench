import shutil
import os
from pathlib import Path
import yaml

def get_parser():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--src_dir', required=True)
    parser.add_argument('--tgt_dir', required=True)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    config_name = 'exp_config.yaml'
    # Load all results
    results = {}
    for root, dirs, files in os.walk(args.src_dir, topdown=True):
        if config_name in files:  # leaf
            path = root.split(os.sep)
            exp_config = yaml.safe_load(open(os.path.join(root,config_name)))
            path_list = root.split(os.sep)
            exp_name = path_list[-1]

            ## NOTE Modify here for you need
            env_name = path_list[-2]  + \
                      f"_syl_{exp_config['env_config']['syllable']}" + \
                      f"_fb_{exp_config['env_config']['feedback']}"

            dir_name = (os.sep).join([args.tgt_dir, env_name, exp_name])
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(root, f), os.path.join(dir_name,f))