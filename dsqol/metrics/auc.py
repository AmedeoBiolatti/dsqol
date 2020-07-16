import numpy as np
from sklearn import metrics


def roc_auc_lowerbound(y, p, alpha=0.01):
    p0 = p[y <= 0.5].reshape(-1)
    p1 = p[y >= 0.5].reshape(-1)

    alpha = 1 - np.sqrt(1 - alpha)

    p0 = np.sort(p0)
    n0 = p0.shape[0]
    h0 = np.arange(n0 + 1) / n0
    e0 = (1 / (2 * n0) * np.log(2 / alpha)) ** 0.5
    l0 = np.clip(h0 - e0, 0, 1)

    p1 = np.sort(p1)
    n1 = p1.shape[0]
    h1 = np.arange(n1) / n1
    e1 = (1 / (2 * n1) * np.log(2 / alpha)) ** 0.5
    u1 = np.clip(h1 + e1, 0, 1)

    p = np.concatenate([np.zeros(1), p0, p1, np.ones(1)])
    p = np.sort(p)

    p_l0 = l0[np.maximum.accumulate(np.argmax(p.reshape(-1, 1) <= p0.reshape(1, -1), 1))]
    p_u1 = u1[np.maximum.accumulate(np.argmax(p.reshape(-1, 1) <= p1.reshape(1, -1), 1))]
    curve_y = np.minimum(1 - p_l0, p_u1)

    return 1.0 - metrics.auc(p, curve_y)
