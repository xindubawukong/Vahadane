import cv2


def read_image(path):
    img = cv2.imread(path)
    # opencv default color space is BGR, change it to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
