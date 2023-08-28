import yaml, os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from verbal_gym.utils.benchmark_utils import set_nested_value
import collections

def main(args):
    data_dir = args.data_dir
    log_dir = args.log_dir if args.log_dir is not None else args.data_dir
    config_filename = args.config_filename
    stats_filename = args.stats_filename

    # traverse root directory, and list directories as dirs and files as files
    results = {}
    for root, dirs, files in os.walk(data_dir):
        if config_filename in files:  # reach the end
            config = yaml.safe_load(open(os.path.join(root,config_filename), 'r'))
            stats = yaml.safe_load(open(os.path.join(root,stats_filename), 'r'))
            set_nested_value(results, [config['env_name'], config['agent_config']['agent_name']], stats)

    if not args.separate_plots:
        plt.figure()
        n_fig_per_row = 3
        fig, axs = plt.subplots(int(np.ceil(len(results)/n_fig_per_row)), n_fig_per_row, figsize=(15,15))
        i = 0
        matplotlib.rcParams.update({'font.size': 10})

        for env_name in results:
            result = collections.OrderedDict(sorted(results[env_name].items()))
            print(env_name)

            means, errors = [], []
            for agent_name in result:
                mean, std, n = result[agent_name]['mean'], result[agent_name]['std'], len(result[agent_name]['scores'])
                print(f"\t {agent_name}: mean {mean:.2f} std {std:.2f}")
                means.append(mean)
                errors.append(std/np.sqrt(n))
            # Plot bar plot
            ax = axs[i//n_fig_per_row, i%n_fig_per_row]

            x = np.arange(len(means))
            ax.bar(x, means, width=0.6,#color = 'blue', edgecolor = 'black',
                    yerr=errors, capsize=7, label='poacee')

            x_ticks = [name.replace('full_info', 'fullinfo').replace('_','\n') for name in result.keys()]
            ax.set_xticks(x, x_ticks)
            ax.tick_params(axis='x', labelsize=12)
            ax.set_title(env_name.replace('verbal-',''))
            i+=1

        fig.tight_layout(pad=3.0)
        plt.savefig(os.path.join(log_dir, 'plots.png'))

    else:
        for env_name in results:
            print(env_name)
            plt.figure()
            means, errors = [], []
            for agent_name in result:
                mean, std, n = result[agent_name]['mean'], result[agent_name]['std'], len(result[agent_name]['scores'])
                print(f"\t {agent_name}: mean {mean:.2f} std {std:.2f}")
                means.append(mean)
                errors.append(std/np.sqrt(n))
            # Plot bar plot
            x = np.arange(len(means))
            plt.figure()
            plt.bar(x, means, width=0.3,#color = 'blue', edgecolor = 'black',
                    yerr=errors, capsize=7, label='poacee')
            plt.xticks(x, result.keys())
            plt.savefig(os.path.join(log_dir, f'{env_name}.png'))



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--config_filename', type=str, default='exp_config.yaml')
    parser.add_argument('--stats_filename', type=str, default='stats.yaml')
    parser.add_argument('--separate_plots', action='store_true')
    main(parser.parse_args())