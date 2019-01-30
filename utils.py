import cv2
import numpy as np


def read_image(path):
    img = cv2.imread(path)
    # opencv default color space is BGR, change it to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    p = np.percentile(img, 90)
    img = np.clip(img * 255.0 / p, 0, 255).astype(np.uint8)
    return img
