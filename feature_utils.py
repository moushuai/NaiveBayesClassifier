import math


def mean(data):
    return sum(data)/float(len(data))


def stdev(data):
    avg = mean(data)
    var = sum([pow(x-avg, 2) for x in data]) / float(len(data) - 1)
    return math.sqrt(var)


def split_by_cls(dataset):
        split_dataset = {}
        for instance in dataset:
            if instance[-1] not in split_dataset:
                split_dataset[instance[-1]] = []
            split_dataset[instance[-1]].append(instance)
        return split_dataset


def extract_feature(dataset):
    split_dataset = split_by_cls(dataset)
    features = {}

    for cls, instances in split_dataset.iteritems():
        feature = [(mean(attr), stdev(attr)) for attr in zip(*instances)]
        feature.pop(-1)
        features[cls] = feature
    return features




