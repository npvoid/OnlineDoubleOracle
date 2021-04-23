import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import os
import pickle
from pathlib import Path
np.set_printoptions(precision=3)

mult = 2
SMALL_SIZE = 8*mult
MEDIUM_SIZE = 10*mult*1.6
BIGGER_SIZE = 12*mult*1.3

plt.rcParams['lines.linewidth'] = 3
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=int(SMALL_SIZE*1.8))    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title




def plot_error(data, label=''):
    data_mean = np.mean(np.array(data), axis=0)
    error_bars = stats.sem(np.array(data))
    plt.plot(data_mean, label=label)
    plt.fill_between([i for i in range(data_mean.size)],
                     np.squeeze(data_mean - error_bars),
                     np.squeeze(data_mean + error_bars), alpha=0.4)

def plt_experiments(rounds, brs, game, time_str):
    PATH_ODO = f"results_{game}/{game}_poker-online_oracle-vs-online_oracle--n_rounds-{rounds}--n_brs-{brs}--time-{time_str}"

    fig_handle = plt.figure(figsize=(15, 8), dpi=200)

    # Read ODO data
    data = pickle.load(open(os.path.join(PATH_ODO, 'recordings.p'), 'rb'))
    mean_exp_history_a = data['exp_history_a']
    mean_exp_history_b = data['exp_history_b']
    # Ensure all trajectories have same length
    min_len = min( np.min(np.array([len(a) for a in mean_exp_history_a])), np.min(np.array([len(b) for b in mean_exp_history_b])) )
    exp_oo = np.zeros((len(mean_exp_history_a), min_len))
    for i, (a, b) in enumerate(zip(mean_exp_history_a, mean_exp_history_b)):
        exp_oo[i] = (np.array(a[:min_len]) + np.array(b[:min_len]) ) / 2.
    exp_oo = np.repeat(exp_oo, 2, axis=1)  # Two best responses computed every iter

    # Plot online oracle    
    plot_error(exp_oo, label='ODO')
    plt.legend()
    plt.title(f'{game.capitalize()} Poker')
    plt.xlabel('Number of BR computed')
    plt.ylabel('Exploitability')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_ODO, f'{game}_results.pdf'))


if __name__=="__main__":
    plt_experiments()
    plt.show(block=True)
































































