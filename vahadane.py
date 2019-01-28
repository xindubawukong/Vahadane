import spams
import numpy as np
import cv2


STAIN_NUM = 2
LAMBDA = 0.02


def getV(img):
    I0 = img.reshape((-1,3)).T
    mask = (I0 == 0)
    I0[mask] = 1
    V0 = np.log(255 / I0)
    
#     img_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#     mask = img_LAB[:, :, 0] / 255 < 0.9
#     I = img[mask].reshape((-1, 3)).T
#     mask = (I == 0)
#     I[mask] = 1
#     V = np.log(255 / I)
    return V0


def getW(V):
    W = spams.trainDL(V, K=STAIN_NUM, lambda1=LAMBDA, mode=2, modeD=0, posAlpha=True, posD=True)
    W = W / np.linalg.norm(W, axis=0)[None, :]
    return W


def getH(V, W):
    H = spams.lasso(V, W, mode=2, lambda1=LAMBDA, pos=True)
    return H.toarray()


def SNMF(img, flag=True):
    V = getV(img)
    W = getW(V)
    H = getH(V, W)
    return W,H


def SPCN(img, Ws, Hs, Wt, Ht):
    Hs_RM = np.percentile(Hs, 99, axis=1)
    Ht_RM = np.percentile(Ht, 99, axis=1)
    Hs_norm = Hs * Ht_RM[:, None] / Hs_RM[:, None]
    print(Ws)
    print(Wt)
    Vs_norm = np.dot(Wt, Hs)
    Is_norm = 255 * np.exp(-1 * Vs_norm)
    I = Is_norm.T.reshape(img.shape).astype(np.uint8)
    return I
