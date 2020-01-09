import numpy as np
from numba import jit


def adjust_dist(y_pred, tr_mean, tr_std):
    """Adjusts the prediction distribution following the paper."""
    return tr_mean + (y_pred - y_pred.mean()) / (y_pred.std() / tr_std)


def allocate_to_rate(y_pred):
    """Allocates raw predictions to rates."""
    rates = np.zeros(y_pred.size, dtype=int)
    for i in range(3):
        rates[y_pred >= i + 0.5] = i + 1
    return rates


@jit
def qwk(a1, a2):
    """
    Fast QWK computation.
    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168
    """
    max_rat = 3
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o += (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


def eval_qwk_lgb_regr(y_pred, train_data, tr_mean, tr_std):
    """
    Fast QWK eval function for lgb.
    """
    labels = train_data.get_label()
    y_pred = adjust_dist(y_pred, tr_mean, tr_std)
    y_pred = allocate_to_rate(y_pred)
    return 'kappa', qwk(labels, y_pred), True
