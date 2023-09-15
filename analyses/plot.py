import yaml, os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from verbal_gym.utils.benchmark_utils import set_nested_value, get_nested_value
import collections

def parse_label(config, name, seperator=':'):
    # Read the config file based on nested_key1+nested_key2+.... format and return the value as val1_val2_...
    return '_'.join([ str(get_nested_value(config,n.split(seperator)))  for n in name.split('+')])

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
            plot_name = parse_label(config, args.plot_name)
            legend_name = parse_label(config, args.legend_name)
            set_nested_value(results, [plot_name, legend_name], stats)

    # if not args.separate_plots:
    plt.figure()
    n_fig_per_row = min(args.max_n_fig_per_row, len(results))
    n_fig_per_col = int(np.ceil(len(results)/n_fig_per_row))
    fig, axs = plt.subplots(n_fig_per_col, n_fig_per_row, figsize=(15,15))
    i = 0
    matplotlib.rcParams.update({'font.size': 10})
    for plot_name in results:
        result = collections.OrderedDict(sorted(results[plot_name].items()))
        print(plot_name)
        means, errors = [], []
        for agent_name in result:
            mean, std, n = result[agent_name]['mean'], result[agent_name]['std'], len(result[agent_name]['scores'])
            print(f"\t {agent_name}: mean {mean:.2f} std {std:.2f}")
            means.append(mean)
            errors.append(std/np.sqrt(n))

        # Get axis
        if n_fig_per_col==1:
            if n_fig_per_row==1:
                ax = axs
            else:
                ax = axs[i%n_fig_per_row]
        else:
            ax = axs[i//n_fig_per_row, i%n_fig_per_row]

        # Plot bar plot
        x = np.arange(len(means))
        ax.bar(x, means, width=0.6,#color = 'blue', edgecolor = 'black',
                yerr=errors, capsize=7, label='poacee')

        x_ticks = [name.replace('full_info', 'fullinfo').replace('_','\n') for name in result.keys()]
        ax.set_xticks(x, x_ticks)
        ax.tick_params(axis='x', labelsize=12)
        ax.set_title(plot_name.replace('verbal-',''))
        i+=1

    fig.tight_layout(pad=3.0)
    plt.savefig(os.path.join(log_dir, 'plots.png'))

    # Print average over envs
    print('\n')
    scores = {}
    for plot_name in results:
        for agent_name in results[plot_name]:
            if agent_name not in scores:
                scores[agent_name] = []
            scores[agent_name].append(results[plot_name][agent_name]['mean'])
    for agent_name in scores:
        print(f"{agent_name}: mean {np.mean(scores[agent_name]):.2f} std {np.std(scores[agent_name]):.2f}")



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--config_filename', type=str, default='exp_config.yaml')
    parser.add_argument('--stats_filename', type=str, default='stats.yaml')
    parser.add_argument('--separate_plots', action='store_true')
    parser.add_argument('--plot_name', type=str, default='env_config:env_name')
    parser.add_argument('--legend_name', type=str, default='agent_config:agent_name')
    parser.add_argument('--max_n_fig_per_row', type=int, default=4)
    main(parser.parse_args())