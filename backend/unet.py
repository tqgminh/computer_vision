from operator import mod
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


def unet(src_path, dst_path):
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    H, W, _ = img.shape
    img = cv2.resize(img, (128, 128))
    img = img/255
    img = img.astype(np.float)

    model = load_model('model.h5')

    mask = model.predict(np.reshape(img, (1, 128, 128, 3)))
    mask = np.argmax(mask, axis=3)
    mask = mask[0]

    res = np.zeros((128, 128, 3))
    for h in range(mask.shape[0]):
        for w in range(mask.shape[1]):
            # black
            if mask[h, w] == 1:
                res[h, w, 0] = 0
                res[h, w, 1] = 0
                res[h, w, 2] = 255
            elif mask[h, w] == 2:
                res[h, w, 0] = 0
                res[h, w, 1] = 0
                res[h, w, 2] = 0
            else:
                res[h, w, 0] = 255
                res[h, w, 1] = 0
                res[h, w, 2] = 0

    # res = res.astype(np.int8)

    labeled_img = cv2.resize(res, (W, H))

    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    labeled_img = cv2.filter2D(src=labeled_img, ddepth=-1, kernel=kernel)
    cv2.imwrite(dst_path, labeled_img)


if __name__ == '__main__':

    unet('images/staffordshire_bull_terrier_102.jpg', 'unet/res.jpg')
