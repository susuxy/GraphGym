from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
import os


def simplify(line):
    ret = {}
    ret['trained_score'] = float(re.findall("train_score is (-*\d+.\d+)", line[0])[0])
    ret['denas_score'] = float(re.findall("denas_score is (-*\d+.\d+)", line[0])[0])
    # ret['arch'] = line[0]
    # ret['act'] = re.findall("\'act\': (\d)", line[3])[0]
    # ret['concat'] = re.findall("\'concat\': (\d)", line[3])[0]
    return ret

def parse(filename, args):
    f = open(filename)
    lines = f.readlines()

    arch_list = []
    for idx, line in enumerate(lines):
        if line.startswith('train_score'):
            arch_list.append(lines[idx:idx+1])

    simplified_arch_list = []
    for line in arch_list:
        simplified_arch_list.append(simplify(line))

    x = np.array([v['trained_score'] for v in simplified_arch_list])
    y = np.array([v['denas_score'] for v in simplified_arch_list])
    if 'syncflow' in args.log_name:
        y = np.log(y*(-1))
    elif 'grad_norm' in args.log_name or 'zen' in args.log_name:
        y = np.log(y)
    else:
        raise ValueError

    rank_x = stats.rankdata(x)
    rank_y = stats.rankdata(y)
    spearman_corr, pvalue = stats.spearmanr(rank_x, rank_y)

    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("trained_acc")
    plt.ylabel("nas_score")
    plt.title(f"Spearman corr is {spearman_corr}, num_samples is {len(rank_x)}")
    plt.show()
    plt.savefig(os.path.join('denas_fig', args.log_name + '_score.png'))

    plt.figure()
    plt.scatter(rank_x, rank_y)
    plt.xlabel("rank_trained_acc")
    plt.ylabel("rank_nas_score")
    plt.title(f"Spearman corr is {spearman_corr}, num_samples is {len(rank_x)}")
    plt.show()
    plt.savefig(os.path.join('denas_fig', args.log_name + '_rank.png'))
    print(f"Spearman corr is {spearman_corr}, num_samples is {len(rank_x)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw Correlation')
    parser.add_argument('--log_name', type=str, default='arxiv_node', help='log file name')
    args = parser.parse_args()
    log_file_path = os.path.join('denas_output', 'log_file', args.log_name + '.log')
    parse(log_file_path, args)