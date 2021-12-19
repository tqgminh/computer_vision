import numpy as np
from scipy.spatial.distance import cdist
import cv2
import os
import matplotlib.pyplot as plt


def init_seeds(X, num_of_seeds):
    return X[np.random.choice(X.shape[0], num_of_seeds, replace=False)]


def init_centers(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]

# advance initialization center k-means++


def advanced_init_centers(X, k):
    centers = [X[np.random.randint(X.shape[0]), :]]
    for c_id in range(k - 1):
        # calculate distance between each point with centers
        dist = cdist(X, centers)
        # select closet center to each point
        min_dist = np.min(dist, axis=1)
        # choose next center farthest
        next_center = X[np.argmax(min_dist), :]
        centers.append(next_center)

    return centers


def assign_labels(X, centers):
    d = cdist(X, centers)
    return np.argmin(d, axis=1)


def update_centers(X, labels, K):
    new_centers = np.zeros((K, X.shape[1]))

    for k in range(K):
        Xk = X[labels == k, :]
        new_centers[k, :] = np.mean(Xk, axis=0)

    return new_centers


def stop_training(centers, new_centers, eps):
    dist = np.linalg.norm(centers - new_centers, axis=1)
    total_dist = np.sum(dist)

    return total_dist < eps


def kmeans_clustering(X, k, method='kmeans++'):
    if method == 'kmeans':
        centers = init_centers(X, k)
    elif method == 'kmeans++':
        centers = advanced_init_centers(X, k)

    while True:
        label = assign_labels(X, centers)
        new_centers = update_centers(X, label, k)

        if stop_training(centers, new_centers, 1e-5):
            centers = new_centers
            break
        centers = new_centers

    return centers, label


def kmeans(src_path, k=4):
    img = cv2.imread(src_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X = img.reshape(-1, 3)

    center, label = kmeans_clustering(X, k)
    predicted_mask = center[label]
    predicted_mask = predicted_mask.reshape(img.shape)
    predicted_mask = np.array(predicted_mask)

    return predicted_mask


def get_point_inside_seed(X, seed, bandwidth):
    seed = seed.reshape(1, -1)
    dist = cdist(seed, X)
    check = (dist < bandwidth).flatten()
    return X[check]


def gaussian_kernel_bandwidth(points, seed, bandwidth):
    weights = np.exp(-1*np.linalg.norm((points - seed)/bandwidth, axis=1))
    mean = np.array(
        np.sum(weights[:, None] * points, axis=0) / np.sum(weights), dtype=np.int64)
    return mean


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


def mean_shift(src_path):
    bandwidth = 40
    num_iters = 20
    threshold = 2

    img = cv2.imread(src_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X = img.reshape(-1, 3)
    seeds = init_seeds(X, num_of_seeds=20)
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

    return segmented_image


if __name__ == '__main__':

    kmeans('dog.jpg')
    mean_shift('dog.jpg')
