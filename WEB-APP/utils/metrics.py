import numpy as np


def right_count(y_pred, y_true):
    return (np.abs(y_pred - y_true) < 5).sum()


def mare_c(y_true, y_pred):
    mask = y_true > 5
    y_true_nonzero = y_true[mask]
    y_pred_nonzero = y_pred[mask]
    if len(y_true_nonzero) == 0:
        return float("nan")
    return np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero))


def smape_c(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred)
    return np.mean(diff / denominator) * 100


def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mare = mare_c(y_true, y_pred)
    smape = smape_c(y_true, y_pred)
    accuracy = right_count(y_pred, y_true) / len(y_pred) * 100
    return {
        "RMSE": rmse,
        "R2": r2,
        "MARE": mare,
        "SMAPE": smape,
        "accuracy": accuracy,
    }
