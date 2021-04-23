import pickle
import os
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from utils import print_args
from leduc_poker_exploitability.online_learning_leduc import online_learning_leduc
from plotting import plt_experiments

dt_string = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")

def main(row_num_pol_init, col_num_pol_init, n_brs, n_rounds, eps, lp_nash_iter, refractory, whole_time_average, ref_func=None):

    exp_history0 = []
    exp_history1 = []

    start_time = time.time()
    for i in range(n_rounds):
        print(f'round: {i}')
        exp_history_i = \
            online_learning_leduc(row_num_pol_init, col_num_pol_init, n_brs, eps, lp_nash_iter=lp_nash_iter, refractory=refractory,
                                  whole_time_average=whole_time_average, ref_func=ref_func)
        exp_history0.append(exp_history_i[0])
        exp_history1.append(exp_history_i[1])

        save_dir = Path(__file__).parent / "results_leduc" / f"leduc_poker-online_oracle-vs-online_oracle--n_rounds-{n_rounds}--n_brs-{n_brs}--time-{dt_string}"
        os.makedirs(save_dir, exist_ok=True)
        pickle.dump({
                      'exp_history_a': exp_history0,
                      'exp_history_b': exp_history1
                     }, open(os.path.join(save_dir, 'recordings.p'), 'wb'))

    end_time = time.time()
    avg_time_per_round = (end_time-start_time)/n_rounds
    print(f'Average Time per round for Online-Oracle vs Online-Oracle in Game Leduc Poker are {avg_time_per_round} seconds')
    plt_experiments(n_rounds, n_brs, 'leduc', dt_string)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--row_num_pol_init', '-rp', type=str, default=3)
    parser.add_argument('--col_num_pol_init', '-cp', type=str, default=3)
    parser.add_argument('--n_brs', '-ni', help='number of best responses', type=int, default=10)
    parser.add_argument('--n_rounds', '-nr', type=int, default=2)
    parser.add_argument('--eps', help='run with debug flag', type=float, default=0.01)
    parser.add_argument('--lp_nash_iter', default=False, action='store_true')
    parser.add_argument('--refractory', type=int, default=50)
    parser.add_argument('--whole_time_average', default=False, action='store_true')
    ref_func = lambda n:  2*(np.log(n))*(n**2) #  2 log(n)n^2 or 50 log(n)^2

    args = parser.parse_args()
    print_args(args)
    args = vars(args)
    main(**args, ref_func=ref_func)

