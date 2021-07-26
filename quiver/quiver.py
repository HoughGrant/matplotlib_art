import matplotlib.pyplot as plt
import numpy as np
import uuid


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


def run_4():
    fig = plt.figure(constrained_layout=False, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.87, 0.87, 0.85))

    X, Y = np.meshgrid(np.arange(-10, 10, np.random.uniform(0.05, 0.3)),
                       np.arange(-10, 10, np.random.uniform(0.05, 0.3)))
    U = np.random.uniform(-50, 50)
    U += np.random.uniform(-5, 5) * X * Y * Y
    U += np.random.uniform(-25, 25) * X * X * Y

    V = np.random.uniform(-50, 50)
    V += np.random.uniform(-5, 5) * X * np.sin(Y)
    V += np.random.uniform(-30, 5) * X ** 2

    U *= np.random.randint(0, 2, U.shape) * np.random.randint(0, 2, U.shape)
    U = U % 100
    V *= np.random.randint(0, 2, V.shape) * np.random.randint(0, 2, V.shape)
    V = V % 100

    for _ in range(np.random.randint(1, 6)):
        ax.quiver(X, Y, U, V,
                  angles=np.random.uniform(-50, 50, X.shape[0] * X.shape[1]),
                  width=0.005,
                  # minshaft=np.random.uniform(1.1, 5),
                  alpha=np.random.beta(5, 8),
                  # headlength=np.random.uniform(1, 8),
                  )
    plt.savefig(str(uuid.uuid4()))
    plt.close()


if __name__ == '__main__':
    # run_1()
    # run_2()
    # run_3()
    for _ in range(5):
        run_4()
