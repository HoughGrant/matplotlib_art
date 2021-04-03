import matplotlib.pyplot as plt
import numpy as np


def run_1():
    def v_x(x: float, y: float) -> float:
        """ x_velocity vector field """
        return -y ** 2 - 5 * y + 0.2 * x * y ** 3

    def v_y(x: float, y: float) -> float:
        """ y_velocity vector field """
        return -x ** 2 + 2 * x + 5 * y * x

    def evolve_point(point: tuple, delta: float) -> tuple:
        """ moves point according to previously defined velocity vector fields """
        x_updated = point[0] + v_x(*point) * delta
        y_updated = point[1] + v_y(*point) * delta
        return x_updated, y_updated

    def create_path(initial_point: tuple, num_steps: int, delta: float) -> np.ndarray:
        """ Moves a point through a vector field """
        traveled_points = [initial_point]
        for _ in range(num_steps):
            new_point = evolve_point(traveled_points[-1], delta)
            traveled_points.append(new_point)

        return np.array(traveled_points)

    plt.figure()
    initial_point = (0, 0)
    all_clusters = {}
    for cluster_id, perturb in enumerate(np.linspace(0.1, 1, 10)):

        tmp_initial_point = (initial_point[0] + perturb, initial_point[1] + perturb)
        traveled_points = create_path(tmp_initial_point, num_steps=100000, delta=0.01)
        all_clusters[cluster_id] = traveled_points

        plt.scatter(traveled_points[:, 0], traveled_points[:, 1])
    plt.show()

    # plot vector field


if __name__ == '__main__':
    run_1()
