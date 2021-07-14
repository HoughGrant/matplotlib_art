import matplotlib.pyplot as plt
import numpy as np
import random


def run_1():
    fig = plt.figure(constrained_layout=False, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.87, 0.87, 0.85))

    X, Y = np.meshgrid(np.arange(-10, 10, .2), np.arange(-10, 10, .3))
    U = X * np.cos(X)
    V = Y * np.sin(Y)
    Q = ax.quiver(X, Y, U, V,
                  angles=np.random.uniform(-50, 50, X.shape[0] * X.shape[1]),
                  width=0.005)
    plt.show()


def run_2():
    fig = plt.figure(constrained_layout=False, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.87, 0.87, 0.85))

    X, Y = np.meshgrid(np.arange(-10, 10, .2), np.arange(-10, 10, .3))
    U = X * Y
    V = X * np.sin(Y) - X ** 2
    Q = ax.quiver(X, Y, U, V,
                  angles=np.random.uniform(-50, 50, X.shape[0] * X.shape[1]),
                  width=0.005)
    plt.show()


def run_3():
    fig = plt.figure(constrained_layout=False, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.87, 0.87, 0.85))

    X, Y = np.meshgrid(np.arange(-10, 10, .2), np.arange(-10, 10, .3))
    U = X * Y
    V = X * np.sin(Y) - X ** 2
    Q = ax.quiver(X, Y, U, V,
                  angles=np.random.uniform(-50, 50, X.shape[0] * X.shape[1]),
                  width=0.005)
    plt.show()


if __name__ == '__main__':
    # run_1()
    # run_2()
    run_3()
