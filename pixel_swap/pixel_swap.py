import matplotlib.pyplot as plt
import numpy as np
import randomcolor
from scipy.ndimage import correlate
from skimage import io, color
from skimage.filters import gaussian
from skimage.filters import sato, hessian
from skimage.morphology import erosion
from skimage.segmentation import random_walker
from skimage.transform import rescale
from sklearn.cluster import KMeans


class ColorPallete:
    def __init__(self, url, n_clusters=8):
        self.url = url
        self.clustered_colors_lab = None
        self.clustered_colors_rgb = None
        self.n_clusters = n_clusters

    def extract_colors(self, remove_white_and_black=True, l_mod=1.0):
        image_rgb = io.imread(self.url)
        image_lab = color.rgb2lab(image_rgb)
        image_lab = rescale(image_lab, 0.05, anti_aliasing=False, multichannel=True)
        image_flattened = image_lab.reshape(-1, image_lab.shape[-1])
        kmeans_lab = KMeans(n_clusters=self.n_clusters, random_state=1).fit(image_flattened)
        self.clustered_colors_lab = kmeans_lab.cluster_centers_
        # gaussian_mixture_lab = GaussianMixture(n_components=self.n_clusters, random_state=1).fit(image_flattened)
        # self.clustered_colors_lab = gaussian_mixture_lab.means_
        if remove_white_and_black:
            self.clustered_colors_lab = self.clustered_colors_lab[(self.clustered_colors_lab[:, 0] > 10) &
                                                                  (self.clustered_colors_lab[:, 0] < 95)]
        self.change_l(l_mod)
        self.clustered_colors_rgb = color.lab2rgb(self.clustered_colors_lab)
        return self.clustered_colors_rgb

    def plot_rgb_colors(self, rgb_colors):
        all_colors = np.ones((rgb_colors.shape[0] * 10, 5, 3))
        for i, rgb in enumerate(rgb_colors):
            all_colors[i * 10:(i + 1) * 10] = rgb
        plt.figure()
        plt.imshow(all_colors)
        plt.show()

    def plot_available_colors(self):
        if self.clustered_colors_rgb is None:
            self.extract_colors()
        self.plot_rgb_colors(self.clustered_colors_rgb)

    def change_l(self, mod_coeff):
        modified_lab = self.clustered_colors_lab[:]
        modified_lab[:, 0] *= mod_coeff
        modified_colors_rgb = color.lab2rgb(modified_lab)
        self.clustered_colors_rgb = modified_colors_rgb

    def get_least_light_color(self):
        if self.clustered_colors_rgb is None:
            self.extract_colors()
        lightest_lab = self.clustered_colors_lab[np.argmin(self.clustered_colors_lab[0, :])]
        return color.lab2rgb(lightest_lab)

    def get_most_light_color(self):
        if self.clustered_colors_rgb is None:
            self.extract_colors()
        lightest_lab = self.clustered_colors_lab[np.argmax(self.clustered_colors_lab[0, :])]
        return color.lab2rgb(lightest_lab)


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


def strange_noise():
    image_rgb = np.random.uniform(0, 1, (100, 100, 3))
    labels = np.random.randint(5, size=(100, 100))

    for i in range(100):
        segments = random_walker(image_rgb,
                                 labels,
                                 multichannel=True,
                                 beta=150)

        for color_index in np.unique(segments):
            color = np.average(image_rgb * np.expand_dims(segments == color_index, axis=2), axis=(0, 1))
            image_rgb[segments == color_index] = np.random.uniform(0, 1, 3)

        if i % 10 == 0:
            plt.figure()
            plt.title(i)
            plt.imshow(image_rgb)
            plt.show()

def background_generator():

    size_x = 50
    size_y = 50

    n_labels = 2

    image_rgb = np.random.uniform(0.9, 1, (size_x, size_y, 3))
    labels = np.random.randint(n_labels + 1, size=(size_x, size_y)) * np.random.randint(0, 2, size=(size_x, size_y))

    # segment random image
    segments = random_walker(image_rgb,
                             labels,
                             multichannel=True,
                             beta=250,
                             copy=False,
                             spacing=[50, 10])
    for color_index in np.unique(segments):
        image_rgb[segments == color_index] = np.random.uniform(0, 1, 3)

    # transform segmented image so it is large, preserving blobs, and blurry
    image_rgb = rescale(image_rgb, 2, anti_aliasing=False, multichannel=True)
    image_rgb = gaussian(image_rgb, sigma=2, multichannel=True)
    image_hsv = color.rgb2hsv(image_rgb)
    buffer = 0.05
    hsv_index = 1
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

    return image_rgb


def noisy_mountains():

    size_x = 50
    size_y = 50

    n_labels = 2

    image_rgb = np.random.uniform(0.9, 1, (size_x, size_y, 3))
    labels = np.random.randint(n_labels + 1, size=(size_x, size_y)) * np.random.randint(0, 2, size=(size_x, size_y))

    import randomcolor
    rand_color = randomcolor.RandomColor()
    print(rand_color.generate(hue="blue", count=3, Format='rgb'))
    # segment random image
    segments = random_walker(image_rgb,
                             labels,
                             multichannel=True,
                             beta=250,
                             copy=False,
                             spacing=[50, 10])
    for color_index in np.unique(segments):
        image_rgb[segments == color_index] = np.random.uniform(0, 1, 3)

    # transform segmented image so it is large, preserving blobs, and blurry
    image_rgb = rescale(image_rgb, 2, anti_aliasing=False, multichannel=True)
    image_rgb = gaussian(image_rgb, sigma=2, multichannel=True)
    image_hsv = color.rgb2hsv(image_rgb)
    buffer = 0.05
    hsv_index = 2

    for pix_frac in [0.9]:
        total_pixels_switched = int(image_rgb.shape[0] * image_rgb.shape[0] * pix_frac)
        print(total_pixels_switched)
        for _ in range(total_pixels_switched):
            rand_x, rand_y = np.random.choice(image_hsv.shape[0]), np.random.choice(image_hsv.shape[1])
            orig_rgb = image_rgb[rand_x, rand_y]
            rand_value = image_hsv[rand_x, rand_y, hsv_index]

            x, y = np.where((rand_value * (1 + 0.5 * buffer) >= image_hsv[:, :, hsv_index]) & (
                    rand_value * (1 - 2 * buffer) < image_hsv[:, :, hsv_index]))
            if len(x) == 0:
                continue
            idx = np.random.choice(len(x))
            update_rgb = image_rgb[x[idx], y[idx]]

            image_rgb[x[idx], y[idx]] = orig_rgb
            image_rgb[rand_x, rand_y] = update_rgb

        plt.figure()
        plt.title(pix_frac)
        plt.imshow(image_rgb)
        plt.show()

    plt.figure()
    plt.imshow(image_rgb)
    plt.show()

    # filter =  np.array(1*[(((1.0, 5.0, 1.0),
    #                         (0.0, 0.0, 0.0),
    #                         (-2.0, -5.0, -2.0)))])

    filter = np.array(1 * [(((-5.0, -10.0, -5.0),
                             (0.0, 0.0, 0.0),
                             (2.0, 10.0, 2.0)))])

    # filter =  np.array(1*[(((-1.0, 0.0, 1.0),
    #                         (-1.0, 0.0, 1.0),
    #                         (-1.0, 0.0, 1.0)))])

    filter /= np.sum(np.abs(filter))
    filtered_img = correlate(image_rgb, filter)

    filter_average = np.average(filtered_img)
    filtered_img_binary = np.average(filtered_img[:], axis=2)
    # filtered_img_binary[filtered_img_binary < 1.0 * filter_average] = 0
    # filtered_img_binary[filtered_img_binary >= 1.0 * filter_average] = 1
    plt.figure()
    filtered_img_binary = (filtered_img_binary + np.abs(np.min(filtered_img_binary))) / np.max(
        filtered_img_binary + np.abs(np.min(filtered_img_binary)))
    # filtered_img_binary = 1 + filtered_img_binary
    filtered_img_binary *= 2
    plt.imshow(filtered_img_binary, cmap='gray')  # , vmin=0, vmax=1)
    plt.colorbar()
    plt.show()

    hsv_index = 2
    image_hsv = color.rgb2hsv(image_rgb)
    image_hsv[:, :, hsv_index] *= filtered_img_binary
    image_rgb_shadow_aug = color.hsv2rgb(image_hsv)
    plt.figure()
    plt.imshow(image_rgb_shadow_aug)
    plt.show()


def noisy_mountains_2():
    rand_color = randomcolor.RandomColor()

    size_x = 20
    size_y = 20
    upscale_factor = 10
    n_labels = 4
    sigma = 1
    buffer = 0.0005
    hsv_index = 1
    segment_spacing = [10, 10]
    image_rgb = np.random.uniform(0, 1, (size_x, size_y, 3))
    labels = np.random.randint(n_labels + 1, size=(size_x, size_y)) * np.random.randint(0, 2, size=(size_x, size_y))

    # segment random image
    segments = random_walker(image_rgb,
                             labels,
                             multichannel=True,
                             beta=250,
                             copy=False,
                             spacing=segment_spacing,
                             )

    all_colors = np.array(rand_color.generate(hue="purple", count=n_labels, format_='Array_rgb')) / 256.

    for color_index in np.unique(segments):
        color_hsv = color.rgb2hsv(all_colors[color_index - 1])
        color_hsv[2] = 0.3 + (color_index - 1) * 0.4 / n_labels
        color_rgb = color.hsv2rgb(color_hsv)
        image_rgb[segments == color_index] = color_rgb

    # transform segmented image so it is large, preserving blobs, and blurry
    image_rgb = rescale(image_rgb, upscale_factor, anti_aliasing=False, multichannel=True)
    image_rgb = gaussian(image_rgb, sigma=sigma, multichannel=True)
    image_hsv = color.rgb2hsv(image_rgb)

    plt.figure()
    plt.imshow(image_rgb)
    plt.show()

    for pix_frac in [0.9]:
        total_pixels_switched = int(image_rgb.shape[0] * image_rgb.shape[0] * pix_frac)
        print(total_pixels_switched)
        for _ in range(total_pixels_switched):
            rand_x, rand_y = np.random.choice(image_hsv.shape[0]), np.random.choice(image_hsv.shape[1])
            orig_rgb = image_rgb[rand_x, rand_y]
            rand_value = image_hsv[rand_x, rand_y, hsv_index]

            x, y = np.where((rand_value * (1 + 0.5 * buffer) >= image_hsv[:, :, hsv_index]) & (
                    rand_value * (1 - 2 * buffer) < image_hsv[:, :, hsv_index]))
            if len(x) == 0:
                continue
            idx = np.random.choice(len(x))
            update_rgb = image_rgb[x[idx], y[idx]]

            image_rgb[x[idx], y[idx]] = orig_rgb
            image_rgb[rand_x, rand_y] = update_rgb

        plt.figure()
        plt.title(pix_frac)
        plt.imshow(image_rgb)
        plt.show()

    plt.figure()
    plt.imshow(image_rgb)
    plt.show()


    # filtered_img = prewitt_v(color.rgb2hsv(image_rgb)[:, :, 0])
    # filtered_img = meijering(color.rgb2hsv(image_rgb)[:, :, 0])
    filtered_img = sato(color.rgb2hsv(image_rgb)[:, :, 0])
    # filtered_img = frangi(color.rgb2hsv(image_rgb)[:, :, 0])

    plt.figure()
    plt.imshow(filtered_img)
    plt.show()

    filtered_img = filtered_img
    filtered_img = (filtered_img + np.abs(np.min(filtered_img))) / np.max(
        filtered_img + np.abs(np.min(filtered_img)))
    filtered_img += 0.8

    plt.figure()
    plt.imshow(filtered_img, cmap='gray')
    plt.colorbar()
    plt.show()

    num_erosions = 5
    eroded_image = hessian(color.rgb2hsv(image_rgb)[:, :, 0])
    eroded_aug = np.zeros_like(eroded_image)
    for n in range(num_erosions):
        eroded_aug += 1 * eroded_image
        eroded_image = erosion(eroded_image)

    plt.figure()
    plt.imshow(image_rgb)
    plt.show()

    image_hsv = color.rgb2hsv(image_rgb)
    image_hsv[:, :, 2] *= filtered_img
    image_hsv[:, :, 2] *= 1 - (0.3 * eroded_aug / np.max(eroded_aug))
    image_rgb_shadow_aug = color.hsv2rgb(image_hsv)
    plt.figure()
    plt.imshow(image_rgb_shadow_aug)
    plt.show()


if __name__ == '__main__':
    # naive_approach()
    # strange_noise()
    noisy_mountains_2()