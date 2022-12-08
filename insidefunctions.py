import cv2 
import numpy as np
from scipy import ndimage

def OtsuBasedInside(mask,img, minsize = 3):
    img= ndimage.gaussian_filter(img, [minsize,minsize], mode='constant')
    h,w = mask.shape[:2]
    if h>= minsize and w >= minsize:
        for i in range(h):
            for j in range(w):
                if mask[i,j] == 0:
                    img[i,j]= 0
        otstrsh=cv2.threshold(img,0,255, cv2.THRESH_OTSU)[0]
        img = cv2.threshold(img,2*otstrsh,255,cv2.THRESH_BINARY)[1]
    else:
        img = np.zeros((h,w),dtype=np.uint8)
    return img

def DistanceBasedInside(mask,img,erode=3, adaptive = False):
    if erode != 0:
        borders = cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_CROSS, (erode,erode)), iterations = 1)
    distance = cv2.distanceTransform(mask,cv2.DIST_C,5).astype(np.uint8)
    surefg = cv2.threshold(distance,0 ,255, cv2.THRESH_OTSU)[1]
    return surefg

