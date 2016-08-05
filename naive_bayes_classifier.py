import csv
from feature_utils import extract_feature
from predict_utils import predict, accuracy
import numpy as np


def load_dataset(filename):
    lines = csv.reader(open(filename, 'rb'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def data_split(dataset, ratio=0.7):
    train_size = int(len(dataset) * ratio)
    dataset = list(dataset)
    train = dataset[:train_size]
    test = dataset[train_size:]
    return train, test

if __name__ == "__main__":
    # load dataset and prepare data
    filename = "../data/data.csv"
    dataset = load_dataset(filename)
    train, test = data_split(dataset, ratio=0.7)

    # extract features
    features = extract_feature(train)

    # make prediction and evaluation
    predictions = predict(test, features)
    test = np.array(test)
    accuracy = accuracy(predictions[:, 0], test[:, -1])
    print "accuracy: ", accuracy



