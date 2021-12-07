import cv2
import numpy as np

from func import assign_label, cal_max_distance, cal_mean_region, cal_similarity
from func import nearby_region_merging, noise_region_merging, region_growing, select_seeds


def region_based(src_path, dst_path):
    # read image to array
    bgr_img = cv2.imread(src_path)

    [original_h, original_w, C] = bgr_img.shape
    img = np.array(bgr_img)
    #img = cv2.resize(img, (256, 256))
    n_img = img/np.max(img)
    [H, W, C] = img.shape  # get shape of image

    similarity, similar_threshold = cal_similarity(img)

    max_distance = cal_max_distance(n_img)

    labels, regions, seeds = select_seeds(
        H, W, similarity, max_distance, similar_threshold)

    mean_region = cal_mean_region(n_img, regions)

    labels, regions, mean_region, seeds = region_growing(
        n_img, labels, regions, seeds, mean_region)

    regions, mean_region = nearby_region_merging(regions, mean_region)

    regions, mean_region = noise_region_merging(regions, mean_region, H*W/25)

    labels, regions = assign_label(labels, regions)

    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    labeled_img[label_hue == 0] = 0

    #labeled_img = cv2.resize(labeled_img, (original_w, original_h))

    cv2.imwrite(dst_path, labeled_img)


if __name__ == '__main__':
    region_based('test_original\Abyssinian_31.jpg', 'region-based/res.jpg')
