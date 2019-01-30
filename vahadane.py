import spams
import numpy as np
import cv2
import time


STAIN_NUM = 2
THRESH = 0.9
LAMBDA1 = 0.02
LAMBDA2 = 0.02
ITER = 100
getH_mode = 0 # 0: spams.lasso; 1: pinv;


def show_config():
    print('STAIN_NUM =', STAIN_NUM)
    print('THRESH =', THRESH)
    print('LAMBDA1 =', LAMBDA1)
    print('LAMBDA2 =', LAMBDA2)
    print('ITER =', ITER)
    print('getH_mode =', getH_mode)


def getV(img):
    
    I0 = img.reshape((-1,3)).T
    I0[I0==0] = 1
    V0 = np.log(255 / I0)
    
    img_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    mask = img_LAB[:, :, 0] / 255 < THRESH
    I = img[mask].reshape((-1, 3)).T
    I[I == 0] = 1
    V = np.log(255 / I)
    
    return V0, V


def getW(V):
    W = spams.trainDL(V, K=STAIN_NUM, lambda1=LAMBDA1, iter=ITER, mode=2, modeD=0, posAlpha=True, posD=True)
    W = W / np.linalg.norm(W, axis=0)[None, :]
    if (W[0,0] < W[0,1]):
        W = W[:, [1,0]]
    return W


def getH(V, W):
    if (getH_mode == 0):
        H = spams.lasso(V, W, mode=2, lambda1=LAMBDA2, pos=True).toarray()
    elif (getH_mode == 1):
        H = np.linalg.pinv(W).dot(V);
        H[H<0] = 0
    else:
        H = 0
    return H


def stain_separate(img):
    V0, V = getV(img)
    W = getW(V)
    H = getH(V0, W)
    return W, H


def SPCN(img, Ws, Hs, Wt, Ht):
    Hs_RM = np.percentile(Hs, 99)
    Ht_RM = np.percentile(Ht, 99)
    Hs_norm = Hs * Ht_RM / Hs_RM
    Vs_norm = np.dot(Wt, Hs_norm)
    Is_norm = 255 * np.exp(-1 * Vs_norm)
    I = Is_norm.T.reshape(img.shape).astype(np.uint8)
    return I
