import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def calculate_metrics(y_true, y_pred_probs):
    y_pred = (y_pred_probs > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc = 0.0
    return acc, f1, auc
