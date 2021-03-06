import cv2
import numpy as np


def distance(point1, point2):
    s1 = point1.astype(np.float)
    s2 = point2.astype(np.float)
    numerator = (
        (s1[0] - s2[0]) * (s1[0] - s2[0])
        + (s1[1] - s2[1]) * (s1[1] - s2[1])
        + (s1[2] - s2[2]) * (s1[2] - s2[2])
    )

    return np.sqrt(numerator)


def cal_similarity(hsv_img):
    [H, W, C] = hsv_img.shape
    sigma = np.zeros([H, W])
    for h in range(H):
        for w in range(W):
            min_x = max(0, h - 1)
            min_y = max(0, w - 1)
            max_x = min(H - 1, h + 1)
            max_y = min(W - 1, w + 1)
            S = (max_x - min_x + 1) * (max_y - min_y + 1)

            # get 8 neightbors of the selected pixel
            neighbor = hsv_img[min_x : max_x + 1, min_y : max_y + 1]

            # reshape to 2D array
            tmp = np.reshape(neighbor, newshape=(S, C))

            # calculate standard deviation of 3 channels of the pixel and its neighbors
            sigma_neighbor = np.std(tmp, axis=0)

            # get sum of standard deviation of 3 chanels
            sigma[h, w] = np.sum(sigma_neighbor)

    # normalize matrix in (0, 1)
    sigma = sigma / np.max(sigma)
    similarity = 1 - sigma

    # calculate threshold using the OTSU algorithm
    bin_sim = np.round(similarity * 255)  # convert to unsign 8-bit integer
    bin_sim = bin_sim.astype(np.uint8)
    threshold, _ = cv2.threshold(bin_sim, 0, 255, cv2.THRESH_OTSU)  # get threshold

    threshold = threshold / 255  # threshold for condition 1

    return similarity, threshold


def cal_max_distance(n_img):
    [H, W, C] = n_img.shape
    max_distance = np.zeros([H, W])
    for h in range(H):
        for w in range(W):
            min_x = max(0, h - 1)
            min_y = max(0, w - 1)
            max_x = min(H - 1, h + 1)
            max_y = min(W - 1, w + 1)
            S = (max_x - min_x + 1) * (max_y - min_y + 1)

            # get all neightbors of the selected pixel
            neighbor = n_img[min_x : max_x + 1, min_y : max_y + 1]

            # reshape to 2D array
            tmp = np.reshape(neighbor, newshape=(S, C))

            # the i-th dimension of vector d is distance of the i-th neighbor and selected pixel
            d = np.zeros(S)
            for i in range(S):
                d[i] = distance(n_img[h, w], tmp[i])

            # get max distance
            max_distance[h, w] = np.max(d)

    return max_distance


def select_seeds(
    height, width, similarity, max_distance, similar_threshold, maxdist_threshold=0.05
):
    # mark seed
    seeds = []
    labels = np.zeros([height, width], np.uint8)
    for h in range(height):
        for w in range(width):
            if (
                similarity[h, w] > similar_threshold
                and max_distance[h, w] < maxdist_threshold
            ):
                labels[h, w] = 1
                seeds.append([h, w])

    num_labels, labels = cv2.connectedComponents(labels)

    # list of seeds
    regions = [[] for i in range(num_labels)]

    for h in range(height):
        for w in range(width):
            label = labels[h, w]
            regions[label].append([h, w])

    print("Seeds have been selected!")

    return labels, regions, seeds


def cal_mean_region(n_img, regions):
    mean_region = []
    for region in regions:
        mr = [0, 0, 0]
        for seed in region:
            mr[0] += n_img[seed[0], seed[1], 0]
            mr[1] += n_img[seed[0], seed[1], 1]
            mr[2] += n_img[seed[0], seed[1], 2]

        l = len(region)
        mr[0] = mr[0] / l
        mr[1] = mr[1] / l
        mr[2] = mr[2] / l
        mean_region.append(mr)

    mean_region = np.array(mean_region)

    return mean_region


def region_growing(
    n_img, labels, regions, seeds, mean_region, max_iter=1000000, grow_threshold=0.05
):
    # region growing on the selected seeds
    idx = 0
    last = len(seeds)
    moves = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
    [H, W, _] = n_img.shape

    for n in range(max_iter):
        # print(len(seeds))
        ctr = 0
        for i in range(idx, last):
            seed = seeds[i]
            x = seed[0]
            y = seed[1]
            label = labels[x, y]
            p1 = n_img[x, y]

            for move in moves:
                xx = x + move[0]
                yy = y + move[1]

                if xx >= 0 and xx < H and yy >= 0 and yy < W:
                    p2 = n_img[xx][yy]
                    d = distance(p1, p2)

                    if d < grow_threshold and labels[xx, yy] == 0:
                        labels[xx, yy] = label
                        seeds.append([xx, yy])
                        l = len(regions[label])
                        mean_region[label] = np.add(mean_region[label] * l, p2) / (
                            l + 1
                        )
                        regions[label].append([xx, yy])
                        ctr = ctr + 1

        idx = last
        last = last + ctr
        if idx >= last:
            break

    print("Region growing is done!")

    return labels, regions, mean_region, seeds


def merge(region1, region2, mean_region1, mean_region2):
    l1 = len(region1)
    l2 = len(region2)
    p1 = mean_region1
    p2 = mean_region2

    mean_region1 = np.add(p1 * l1, p2 * l2) / (l1 + l2)
    region1 = [*region1, *region2]
    mean_region2 = np.zeros(3)
    region2 = []

    return region1, region2, mean_region1, mean_region2


def noise_region_merging(regions, mean_region, noise_merge_threshold):
    num_regions = len(regions)
    emp_row = []
    max_iter = 100
    for n in range(max_iter):
        for i in range(1, num_regions):
            if len(regions[i]) < noise_merge_threshold and len(regions[i]) > 0:
                assign_label = 1
                m = 100000
                p1 = mean_region[i]
                for j in range(1, num_regions):
                    if len(regions[j]) > 0 and j != i:
                        d = distance(p1, mean_region[j])
                        if d < m:
                            m = d
                            assign_label = j

                # print(assign_label)
                (
                    regions[assign_label],
                    regions[i],
                    mean_region[assign_label],
                    mean_region[i],
                ) = merge(
                    regions[assign_label],
                    regions[i],
                    mean_region[assign_label],
                    mean_region[i],
                )
                emp_row.append(i)

    while [] in regions:
        regions.remove([])

    mean_region = np.delete(mean_region, emp_row, axis=0)

    print("Noise region merging is done!")

    return regions, mean_region


def nearby_region_merging(regions, mean_region, nearby_merge_threshold=0.1):
    num_regions = len(regions)
    emp_row = []
    for i in range(1, num_regions):
        if len(regions[i]) == 0:
            continue
        else:
            for j in range(1, num_regions):
                if len(regions[j]) == 0 or j == i:
                    continue
                else:
                    p1 = mean_region[i]
                    p2 = mean_region[j]
                    if distance(p1, p2) < nearby_merge_threshold:
                        regions[i], regions[j], mean_region[i], mean_region[j] = merge(
                            regions[i], regions[j], mean_region[i], mean_region[j]
                        )
                        emp_row.append(j)

    while [] in regions:
        regions.remove([])

    mean_region = np.delete(mean_region, emp_row, axis=0)

    print("Nearby region merging is done!")

    return regions, mean_region


def assign_label(regions, H, W, bgr_img):

    mask = np.zeros((H, W, 3))
    for i in range(len(regions)):
        first_seed = regions[i][0]
        color = bgr_img[first_seed[0], first_seed[1]]
        for seed in regions[i]:
            x = seed[0]
            y = seed[1]
            mask[x, y] = color
    print(len(regions))
    return mask


def auto_seeded_region_growing(src_path):
    # read image to array
    bgr_img = cv2.imread(src_path)

    [original_h, original_w, C] = bgr_img.shape
    img = np.array(bgr_img)
    img = cv2.resize(img, (224, 224))
    n_img = img / 255
    [H, W, C] = img.shape  # get shape of image
    similarity, similar_threshold = cal_similarity(img)
    max_distance = cal_max_distance(n_img)
    labels, regions, seeds = select_seeds(
        H, W, similarity, max_distance, similar_threshold
    )
    mean_region = cal_mean_region(n_img, regions)
    labels, regions, mean_region, seeds = region_growing(
        n_img, labels, regions, seeds, mean_region
    )
    regions, mean_region = nearby_region_merging(regions, mean_region)
    regions, mean_region = noise_region_merging(regions, mean_region, H * W / 25)
    labeled_img = assign_label(regions, H, W, img)

    # label_hue = np.uint8(179*labels/np.max(labels))
    # blank_ch = 255*np.ones_like(label_hue)
    # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # labeled_img[label_hue == 0] = 0

    labeled_img = cv2.resize(labeled_img, (original_w, original_h))

    return labeled_img


if __name__ == "__main__":
    auto_seeded_region_growing("static/upload/cat.jpg")
