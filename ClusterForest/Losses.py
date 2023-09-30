import numpy as np


def zero_one_loss(leaf, y):

    if leaf.pred != y:
        return 1
    else:
        return 0


def l1_loss(leaf, y):

    return 1 - leaf.probabilities[y]


def l2_loss(leaf, y):

    res = np.copy(leaf.probabilities)
    res[y] -= 1

    return np.linalg.norm(res)


def zero_one_loss_eval(test_y, y_hat):

    if y_hat is None or len(test_y) != len(y_hat):

        return 1

    zero_one = 0
    for i in range(len(y_hat)):

        if test_y[i] != y_hat[i]:
            zero_one += 1

    return zero_one / len(test_y)
