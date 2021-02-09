import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color


def naive_approach():
    # url = 'https://upload.wikimedia.org/wikipedia/commons/6/6d/Blue-and-Yellow-Macaw.jpg'
    url = 'https://i.pinimg.com/originals/a2/75/0c/a2750c2051f6c5eda339bf314d1075e4.jpg'
    # url = 'https://upload.wikimedia.org/wikipedia/en/d/d1/Plasticbeach452.jpg'
    # url = 'https://upload.wikimedia.org/wikipedia/en/a/a2/The_Fall_%28Gorillaz_album%29_cover.jpg'
    image_rgb = io.imread(url)

    plt.figure()
    plt.imshow(image_rgb)
    plt.show()

    image_hsv = color.rgb2hsv(image_rgb)
    buffer = 0.1
    hsv_index = 2
    for _ in range(10000):
        rand_x, rand_y = np.random.choice(image_hsv.shape[0]), np.random.choice(image_hsv.shape[1])
        orig_rgb = image_rgb[rand_x, rand_y]
        rand_value = image_hsv[rand_x, rand_y, hsv_index]

        x, y = np.where((rand_value * (1 + buffer) >= image_hsv[:, :, hsv_index]) & (
                rand_value * (1 - buffer) < image_hsv[:, :, hsv_index]))
        if len(x) == 0:
            continue
        idx = np.random.choice(len(x))
        update_rgb = image_rgb[x[idx], y[idx]]

        image_rgb[x[idx], y[idx]] = orig_rgb
        image_rgb[rand_x, rand_y] = update_rgb

    plt.figure()
    plt.imshow(image_rgb)
    plt.show()


if __name__ == '__main__':
    naive_approach()
