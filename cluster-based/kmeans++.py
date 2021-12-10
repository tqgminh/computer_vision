import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


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


def kmeans(src_path, dst_path, k):
    img = cv2.imread(src_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X = img.reshape(-1, 3)

    center, label = kmeans_clustering(X, k)
    predicted_mask = center[label]
    predicted_mask = predicted_mask.reshape(img.shape)
    predicted_mask = np.array(predicted_mask)
    cv2.imwrite(dst_path, predicted_mask)


"""
def visualize(image, predicted_mask):

    figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[0].set_axis_off()

    ax[1].imshow(predicted_mask)
    ax[1].set_title('Predicted mask')
    ax[1].set_axis_off()

    plt.show()
"""

if __name__ == '__main__':
    """
    images_directory = os.path.join('images')
    image_filenames = sorted(os.listdir(images_directory))

    num_test = 100
    test_idx = np.random.choice(len(image_filenames), num_test, replace=False)

    for i in test_idx:
        img_filename = image_filenames[i]
        img = cv2.imread(os.path.join(images_directory, img_filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X = img.reshape(-1, 3)

        # choose method
        method = 'kmeans++'

        k = 3
        center, label = kmeans(X, k, method)
        new_center = np.array([[0, 0, 0],
                               [255, 0, 0],
                               [255, 255, 255]])
        predicted_mask = new_center[label]
        predicted_mask = predicted_mask.reshape(img.shape) / 255.0

        visualize(img, predicted_mask)
    """

    kmeans('test_original\Abyssinian_31.jpg', 'cluster-based/res.jpg')
