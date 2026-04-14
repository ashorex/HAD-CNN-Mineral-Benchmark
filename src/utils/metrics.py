from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_metrics(y_true, y_pred):

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    return acc, report, cm