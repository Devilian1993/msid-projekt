import sklearn.metrics


def print_metrics(scores):
    print("F1 mean:  %.4f" % scores.mean())
    print("F1 std:   %.4f" % scores.std())
