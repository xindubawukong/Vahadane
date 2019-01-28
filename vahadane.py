import spams
import numpy as np
import cv2
from global_config import *


def getV(img):
    img_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    mask = img_LAB[:, :, 0] / 255 < 0.9
    I = img[mask].reshape((-1, 3))
    mask = (I == 0)
    I[mask] = 1
    V = np.log(255 / I)
    return V


def getW(V):
    W = spams.trainDL(V.T, K=STAIN_NUM, lambda1=LAMBDA, mode=2, modeD=0, posAlpha=True, posD=True).T
    W = W / np.linalg.norm(W, axis=1)[:, None]
    return W


def getH(V, W):
    return 0


def stain_separate(img, flag=True):
    V = getV(img)
    W = getW(V)
    print('W:\n', W)
    H = getH(V, W)
    return W,H