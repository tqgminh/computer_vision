import cv2
from tensorflow.keras.models import load_model
import numpy as np


def unet_segmentation(src_path):
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    H, W, _ = img.shape
    img = cv2.resize(img, (224, 224))
    img = img/255
    img = img.astype(np.float)

    model = load_model('model/densenet121_unet.h5')

    mask = model.predict(np.reshape(img, (1, 224, 224, 3)))
    mask = np.argmax(mask, axis=3)
    mask = mask[0]

    res = np.zeros((224, 224, 3))
    for h in range(mask.shape[0]):
        for w in range(mask.shape[1]):
            # black
            if mask[h, w] == 1:
                res[h, w, 0] = 0
                res[h, w, 1] = 0
                res[h, w, 2] = 255
            elif mask[h, w] == 0:
                res[h, w, 0] = 0
                res[h, w, 1] = 255
                res[h, w, 2] = 0

    img = img*255
    img = cv2.addWeighted(img, 1, res, 0.4, 0)
    labeled_img = cv2.resize(img, (W, H))

    return labeled_img


if __name__ == '__main__':

    unet_segmentation('dogcat.png', 'unet/res.jpg')
