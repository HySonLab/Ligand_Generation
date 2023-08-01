import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score, \
    auc, accuracy_score, matthews_corrcoef
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, \
    mean_absolute_percentage_error, max_error, r2_score
from scipy.stats import pearsonr


def get_cindex(y, p):
    summ = 0
    pair = 0

    for i in range(1, len(y)):
        for j in range(0, i):
            if i is not j:
                if y[i] > y[j]:
                    pair += 1
                    summ += 1 * (p[i] > p[j]) + 0.5 * (p[i] == p[j])

    if pair != 0:
        return summ / pair
    else:
        return 0


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for _ in y_obs]
    y_pred_mean = [np.mean(y_pred) for _ in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for _ in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def get_metrics_reg(y_true, y_pred, with_rm2=False, with_ci=False):
    metrics = dict()
    metrics["mse"] = float(mean_squared_error(y_true, y_pred))
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["medae"] = float(median_absolute_error(y_true, y_pred))
    metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
    metrics["maxe"] = float(max_error(y_true, y_pred))
    metrics["r2"] = float(r2_score(y_true, y_pred))
    metrics["pearsonr"] = pearsonr(y_true.flatten(), y_pred.flatten())[0]

    if with_rm2:
        metrics["rm2"] = get_rm2(y_true.flatten().tolist(), y_pred.flatten().tolist())
    if with_ci:
        metrics["ci"] = get_cindex(y_true.flatten().tolist(), y_pred.flatten().tolist())

    return metrics


def get_metrics_cls(y_true, y_pred, transform=torch.sigmoid, threshold=0.5):
    if transform is not None:
        y_pred = transform(y_pred)
    y_pred_lbl = (y_pred >= threshold).type(torch.float32)

    metrics = dict()
    metrics["f1"] = float(f1_score(y_true, y_pred_lbl))
    metrics["precision"] = float(precision_score(y_true, y_pred_lbl, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred_lbl))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred_lbl))
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred_lbl))
    try:
        metrics["rocauc"] = float(roc_auc_score(y_true, y_pred))
    except ValueError:
        metrics["rocauc"] = np.nan
    try:
        precision_list, recall_list, thresholds = precision_recall_curve(y_true, y_pred)
        metrics["prauc"] = float(auc(recall_list, precision_list))
    except ValueError:
        metrics["prauc"] = np.nan

    return metrics