import numpy as np
# from sklearn.metrics import confusion_matrix


# Compute the total computational time for the fusion algorithm
def compute_total_time(M, alg_times):
    is_used = np.zeros((14,))
    for i in range(15):
        for j in range(15):
            if (M[i, j] != 0):
                if (i >= 1):
                    is_used[i - 1] = 1
                if (j >= 1):
                    is_used[j - 1] = 1

    total_dur = np.dot(is_used, alg_times)
    return total_dur


# Evaluation metric
def compute_eval(fused_score):
    look_at_FAR = 0.0001
    sort = np.argsort(fused_score[:, 1])
    scores = fused_score[sort]

    totpos = sum(scores[:, 0])
    totneg = scores.shape[0] - totpos
    fa = (np.cumsum(scores[:, 0] - 1) + totneg) / totneg
    fr = np.cumsum(scores[:, 0]) / totpos

    i = 0
    while fa[i] > look_at_FAR:
        i += 1

    return scores[i][1], fa[i], fr[i]


# Score function
def score_func(y_true, y_score, verbose=True):
    fuse = np.stack([y_true, y_score], axis=1)

    # compute the FRR at FAR = 0.01%
    thr, fa, fr = compute_eval(fuse)
    if verbose:
        print ("threshold:", thr, "far:", fa, "frr:", fr)

    # y_pred = (y_score > thr) * 1  # predicted classes
    # acc = np.sum(y_true == y_pred) / len(y_true)  # accuracy
    # cm = confusion_matrix(y_true, y_pred)  # confusion matrix
    return fr


# Construct fusion matrix from coef.
def construct_fusion_matrix(coef_, col=range(0, 15), scaler=None, poly=False):
    # coef_: obtained from linear classifier
    # col: subset of algorithms 0,(1-14)

    if scaler is not None:  # scaling
        coef_ = np.insert(coef_[0, 1:] / scaler.scale_, 0, coef_[0, 0] -
                          np.sum(coef_[0, 1:] * scaler.mean_ / scaler.scale_))

    M = np.zeros((15, 15))

    if poly:  # quadratic
        M1 = np.zeros((len(col), len(col)))
        M1[np.triu_indices(len(col))] = coef_
        M[np.ix_(col, col)] = M1
    else:  # linear
        M[0, col] = coef_

    return M

