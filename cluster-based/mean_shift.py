import numpy as np
from scipy.spatial.distance import cdist
import cv2
import os
import matplotlib.pyplot as plt


def init_seeds(X, num_of_seeds):
    return X[np.random.choice(X.shape[0], num_of_seeds, replace=False)]


def get_point_inside_seed(X, seed, bandwidth):
    seed = seed.reshape(1, -1)
    dist = cdist(seed, X)
    check = (dist < bandwidth).flatten()
    return X[check]


def gaussian_kernel_bandwidth(points, seed, bandwidth):
    weights = np.exp(-1*np.linalg.norm((points - seed)/bandwidth, axis=1))
    mean = np.array(np.sum(weights[:, None] * points, axis=0) / np.sum(weights), dtype=np.int64)
    return mean


def visualize(image, predicted_mask):
    figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[0].set_axis_off()

    ax[1].imshow(predicted_mask)
    ax[1].set_title('Predicted mask')
    ax[1].set_axis_off()

    plt.show()


def eliminate_seed(shifted_seeds, bandwidth):
    flags = [1] * len(shifted_seeds)

    for i in range(len(shifted_seeds)):
        if flags[i] == 1:
            w = 1.0
            j = i + 1

            while j < len(shifted_seeds):
                d = np.linalg.norm(shifted_seeds[i] - shifted_seeds[j])
                if d < bandwidth:
                    shifted_seeds[i] = shifted_seeds[i] + shifted_seeds[j]
                    flags[j] = 0
                j += 1

            shifted_seeds[i] = shifted_seeds[i] / w

    converged_seeds = []
    for i in range(len(shifted_seeds)):
        if flags[i] == 1:
            converged_seeds.append(shifted_seeds[i])

    return np.array(converged_seeds)


if __name__ == '__main__':

    bandwidth = 40
    num_of_seeds = 20
    threshold = 2
    num_iters = 20

    images_directory = os.path.join('images')
    image_filenames = sorted(os.listdir(images_directory))

    num_test = 10
    test_idx = np.random.choice(len(image_filenames), num_test, replace=False)
    for i in test_idx:
        img = cv2.imread(os.path.join(images_directory, image_filenames[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X = img.reshape(-1, 3)

        seeds = init_seeds(X, num_of_seeds)
        shifted_seeds = []

        for i in range(len(seeds)):
            pos = seeds[i]
            for n in range(num_iters):
                points = get_point_inside_seed(X, pos, bandwidth)
                new_pos = gaussian_kernel_bandwidth(points, pos, bandwidth)

                dist = np.linalg.norm(new_pos - pos)
                pos = new_pos
                if dist < threshold:
                    break
                shifted_seeds.append(pos)

        final_seeds = eliminate_seed(shifted_seeds, bandwidth)
        segmented_image = []

        for i in range(X.shape[0]):
            dis = cdist(np.array([X[i]]), np.array(final_seeds))
            idx = np.argmin(dis, axis=1)
            segmented_image.append(final_seeds[idx])

        segmented_image = np.reshape(segmented_image, img.shape)
        segmented_image = segmented_image / 255.0

        visualize(img, segmented_image)