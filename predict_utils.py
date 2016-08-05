import math
import numpy as np


def compute_attr_prob(val, mean, stdev):
    exponent = math.exp(-(math.pow(val-mean, 2))/(2*math.pow(stdev, 2)))
    attr_prob = (1. / (math.sqrt(2*math.pi)*stdev)) * exponent
    return attr_prob


def compute_cls_prob(instance, features):
    cls_probs = {}
    for cls, feature in features.iteritems():
        cls_probs[cls] = 1.
        i = 0
        for mean, stdev in feature:
            cls_probs[cls] *= compute_attr_prob(instance[i], mean, stdev)
            i += 1
    return cls_probs


def predict(test_set, features):
    predictions = []
    for instance in test_set:
        cls_probs = compute_cls_prob(instance, features)
        max_cls_label, max_cls_prob = None, -1
        for cls, cls_prob in cls_probs.iteritems():
            if max_cls_label is None or cls_prob > max_cls_prob:
                max_cls_label = cls
                max_cls_prob = cls_prob
        predictions.append([max_cls_label, max_cls_prob])
    return np.array(predictions)


def accuracy(predictions, groundtruth):
    correct = 0
    for i in range(len(groundtruth)):
        if predictions[i] == groundtruth[i]:
            correct += 1
    return (float(correct) / len(groundtruth))

