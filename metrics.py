from sklearn import metrics

def accuracy(truth, pred):
    return metrics.accuracy_score(truth, pred)

def precision(y_true, y_pred):
    return metrics.precision_score(y_true, y_pred, average='weighted')

def recall(y_true, y_pred):
    return metrics.recall_score(y_true, y_pred, average='weighted')

def f1score(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='weighted')