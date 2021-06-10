import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.datasets import make_classification

def run_1():
    x, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2,
                               n_clusters_per_class=2, flip_y=0.01, class_sep=1.0)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    print(5)


if __name__ == '__main__':
    run_1()
