import spams
import numpy as np
import cv2
import time


STAIN_NUM = 2
LAMBDA = 0.1


def getV(img):
    
    I0 = img.reshape((-1,3)).T
    I0[I0==0] = 1
    V0 = np.log(255 / I0)
    
    img_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    mask = img_LAB[:, :, 0] / 255 < 0.8
    I = img[mask].reshape((-1, 3)).T
    I[I == 0] = 1
    V = np.log(255 / I)
    
    return V0, V


def getW(V):
    W = spams.trainDL(V, K=STAIN_NUM, lambda1=0.1,  mode=2, modeD=0, posAlpha=True, posD=True)
    W = W / np.linalg.norm(W, axis=0)[None, :]
    if (W[0,0] < W[0,1]):
        W = W[:, [1,0]]
    return W


def getH(V, W):
    H = spams.lasso(V, W, mode=2, lambda1=0.01, pos=True).toarray()
#     H = np.linalg.pinv(W).dot(V); print(H.min(), H.max());H[H<0] = 0
    return H


def SNMF(img):
    start = time.time()
    V0, V = getV(img)
    print('getV: ' + str(time.time() - start) + ' s')
    start = time.time()
    W = getW(V)
    print(W)
    print('getW: ' + str(time.time() - start) + ' s')
    start = time.time()
    H = getH(V0, W)
    print('getH: ' + str(time.time() - start) + ' s')
    return W,H


def SPCN(img, Ws, Hs, Wt, Ht):
    Hs_RM = np.percentile(Hs, 99, axis=1)
    Ht_RM = np.percentile(Ht, 99, axis=1)
    print(Hs_RM)
    print(Ht_RM)
    Hs_norm = Hs #* (Ht_RM / Hs_RM)[:, None]
    Vs_norm = np.dot(Wt, Hs_norm)
    Is_norm = 255 * np.exp(-1 * Vs_norm)
    I = Is_norm.T.reshape(img.shape).astype(np.uint8)
    return I
