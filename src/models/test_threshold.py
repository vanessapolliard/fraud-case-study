from sklearn.metrics import confusion_matrix
import numpy as np

def predict_threshold(threshold, pipeline, X, y):
    probas = pipeline.set_params().predict_proba(X)
    y_thresh = np.where(probas[:,1] > threshold, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y, y_thresh).ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2*tp / (2*tp + fp + fn)
    return recall, precision, f1