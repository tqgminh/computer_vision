import cv2
import numpy as np
import argparse
from pathlib import Path

from func import assign_label, cal_max_distance, cal_mean_region, cal_similarity
from func import nearby_region_merging, noise_region_merging, region_growing, select_seeds


def get_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='link to image',
                     default='Abyssinian_2.jpg')

    return arg.parse_args()


args = get_arguments()
img_path = Path(args.image_path)

# read image to array
bgr_img = cv2.imread(str(img_path))

# convert BGR color space to HSV color space
hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

hsv_img = np.array(hsv_img)
n_img = hsv_img/np.max(hsv_img)
[H, W, C] = hsv_img.shape  # get shape of image

similarity, similar_threshold = cal_similarity(hsv_img)

max_distance = cal_max_distance(n_img)

labels, regions, seeds = select_seeds(
    H, W, similarity, max_distance, similar_threshold)

mean_region = cal_mean_region(n_img, regions)

labels, regions, mean_region, seeds = region_growing(
    n_img, labels, regions, seeds, mean_region)

regions, mean_region = nearby_region_merging(regions, mean_region)

regions, mean_region = noise_region_merging(regions, mean_region, H*W/150)

labels, regions = assign_label(labels, regions)

label_hue = np.uint8(179*labels/np.max(labels))
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

labeled_img[label_hue == 0] = 0

cv2.imwrite('res.jpg', labeled_img)
cv2.imshow('img', labeled_img)
cv2.waitKey(0)
