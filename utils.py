import numpy as np
from scipy.optimize import linprog


def nash_solver(matrix_a, col_solution):
    n, m = matrix_a.shape

    Aub = np.hstack((np.transpose(matrix_a), -1 * np.ones((m, 1))))
    bub = np.zeros((m, 1))
    Aeq = np.hstack((np.ones((1, n)), np.zeros((1, 1))))
    beq = 1
    c = np.append(np.zeros((n, 1)), 1)
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=None, method='interior-point')
    xoptimal = np.array([res.x[i] for i in range(n)])
    v = np.max(np.matmul(xoptimal, matrix_a))
    if not col_solution:
        return xoptimal, 'x', v
    else:
        Aub = np.hstack((-matrix_a, np.ones((n, 1))))
        bub = np.zeros((n, 1))
        Aeq = np.hstack((np.ones((1, m)), np.zeros((1, 1))))
        beq = 1
        c = np.append(np.zeros((m, 1)), -1)
        res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=None, method='interior-point')
        yoptimal = np.array([res.x[i] for i in range(m)])

        return xoptimal, yoptimal, v


def get_losses(row_policy, col_policy, matrix_a):
    row_loss = np.matmul(matrix_a, col_policy)
    col_loss = -np.matmul(row_policy, matrix_a)
    expected_loss = np.matmul(row_policy, row_loss)
    return row_loss, col_loss, expected_loss


def print_args(args):
    """ Prints the argparse argmuments applied
    Args:
      args = parser.parse_args()
    """
    max_length = max([len(k) for k, _ in vars(args).items()])
    from collections import OrderedDict
    new_dict = OrderedDict((k, v) for k, v in sorted(vars(args).items(), key=lambda x: x[0]))
    for k, v in new_dict.items():
        print(' ' * (max_length-len(k)) + k + ': ' + str(v))
