# Yalala Mohit
# Dhruv Kamalesh Kumar

# Import libraries
from sklearn import metrics


# method to get the accuracy
def accuracy(truth, pred):
    return metrics.accuracy_score(truth, pred)


# method to get the precision
def precision(y_true, y_pred):
    return metrics.precision_score(y_true, y_pred, average='weighted')


# method to get the recall
def recall(y_true, y_pred):
    return metrics.recall_score(y_true, y_pred, average='weighted')


# method to get the f1 score
def f1score(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='weighted')
