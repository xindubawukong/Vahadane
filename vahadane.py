import spams
import numpy as np
import cv2
import time


class vahadane(object)


STAIN_NUM = 2
THRESH = 0.9
LAMBDA1 = 0.02
LAMBDA2 = 0.02
ITER = 100
sep_mode = 0 # 0: normal; 1: fast
getH_mode = 0 # 0: spams.lasso; 1: pinv;


def show_config():
    print('STAIN_NUM =', STAIN_NUM)
    print('THRESH =', THRESH)
    print('LAMBDA1 =', LAMBDA1)
    print('LAMBDA2 =', LAMBDA2)
    print('ITER =', ITER)
    print('sep_mode =', sep_mode)
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
    W = spams.trainDL(np.asfortranarray(V), K=STAIN_NUM, lambda1=LAMBDA1, iter=ITER, mode=2, modeD=0, posAlpha=True, posD=True)
    W = W / np.linalg.norm(W, axis=0)[None, :]
    if (W[0,0] < W[0,1]):
        W = W[:, [1,0]]
    return W


def getH(V, W):
    print(V, W)
    if (getH_mode == 0):
        H = spams.lasso(np.asfortranarray(V), np.asfortranarray(W), mode=2, lambda1=LAMBDA2, pos=True).toarray()
    elif (getH_mode == 1):
        H = np.linalg.pinv(W).dot(V);
        H[H<0] = 0
    else:
        H = 0
    return H


def stain_separate(img):
    start = time.time()
    if (sep_mode == 0):
        V0, V = getV(img)
        W = getW(V)
        H = getH(V0, W)
    elif (sep_mode == 1):
        m = img.shape[0]
        n = img.shape[1]
        grid_size_m = int(m / 10)
        lenm = int(m / 30)
        grid_size_n = int(n / 10)
        lenn = int(n / 30)
        W = np.zeros((81, 3, STAIN_NUM)).astype(np.float64)
        for i in range(0, 9):
            for j in range(0, 9):
                px = (i + 1) * grid_size_m
                py = (j + 1) * grid_size_n
                patch = img[px - lenm : px + lenm, py - lenn: py + lenn, :]
                V0, V = getV(patch)
                W[i*9+j] = getW(V)
        W = np.mean(W, axis=0)
        V0, V = getV(img)
        H = getH(V0, W)
    print('SNMF time:', time.time()-start, 's')
    return W, H


def SPCN(img, Ws, Hs, Wt, Ht):
    Hs_RM = np.percentile(Hs, 99)
    Ht_RM = np.percentile(Ht, 99)
    Hs_norm = Hs * Ht_RM / Hs_RM
    Vs_norm = np.dot(Wt, Hs_norm)
    Is_norm = 255 * np.exp(-1 * Vs_norm)
    I = Is_norm.T.reshape(img.shape).astype(np.uint8)
    return I
