import numpy as np
from scipy.stats import norm

def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i+j-1)+1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2

def fast_delong(y_true, y_scores):
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    m, n = len(pos), len(neg)
    all_scores = np.concatenate((pos, neg))
    ranks = compute_midrank(all_scores)
    auc = (np.sum(ranks[:m]) - m*(m+1)/2) / (m*n)
    return auc

def delong_roc_test(y_true, p1, p2):
    auc1 = fast_delong(y_true, p1)
    auc2 = fast_delong(y_true, p2)
    var = np.var(p1 - p2)
    z = (auc1 - auc2) / np.sqrt(var + 1e-8)
    return 2 * (1 - norm.cdf(abs(z)))
