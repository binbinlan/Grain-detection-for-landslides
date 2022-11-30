# -*- coding:utf-8 -*-
#本程序用于大津算法的实现
'''
author: binbinlan, marcolovati
data: 2022.05.10
email:787250087@qq.com, mlov@cdut.edu.cn
'''

import time

import cv2  #导入opencv模块
import numpy as np
import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker


root = tk.Tk()
#root.withdraw()


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def adaptiveOtsu(gray,div_pixel,fraction):
    thresholded = np.zeros((gray.shape[0],gray.shape[1]), dtype=np.uint8)
    hei = int(gray.shape[0]/div_pixel)
    wid = int(gray.shape[1]/div_pixel)
    otsumat = np.zeros((hei,wid),dtype=np.uint8)
    for y in range(hei):
        for x in range(wid):
            otsumat[y][x] = cv2.threshold(gray[y*div_pixel:y*div_pixel+div_pixel,x*div_pixel:x*div_pixel+div_pixel],0 ,255, cv2.THRESH_OTSU)[0]
    otsumat = ndimage.gaussian_filter(otsumat, [3,3], mode='constant')
    for y in range(hei):
        for x in range(wid):
            thresholded[y*div_pixel:y*div_pixel+div_pixel,x*div_pixel:x*div_pixel+div_pixel]=cv2.threshold(gray[y*div_pixel:y*div_pixel+div_pixel,x*div_pixel:x*div_pixel+div_pixel],otsumat[y][x]*fraction,255,cv2.THRESH_BINARY)[1]
        if gray.shape[1]%div_pixel!=0:
            thresholded[y*div_pixel:y*div_pixel+div_pixel,wid*div_pixel:gray.shape[1]]=cv2.threshold(gray[y*div_pixel:y*div_pixel+div_pixel,wid*div_pixel:gray.shape[1]],otsumat[y][wid-1]*fraction,255,cv2.THRESH_BINARY)[1]
    if gray.shape[0]%div_pixel!=0:
        for x in range(wid):
            thresholded[hei*div_pixel:gray.shape[0],x*div_pixel:x*div_pixel+div_pixel]=cv2.threshold(gray[hei*div_pixel:gray.shape[0],x*div_pixel:x*div_pixel+div_pixel],otsumat[hei-1][x]*fraction,255,cv2.THRESH_BINARY)[1]
    return thresholded

def confirmedThreshold(to_b_conf,confirmed):
    cleaningmask = cv2.addWeighted(to_b_conf, 0.8, confirmed, 0.2, 0)
    dst = cv2.addWeighted(to_b_conf, 0.5, confirmed, 0.5, 0)
    mask = np.zeros((cleaningmask.shape[0],cleaningmask.shape[1]), dtype=np.uint8)
    mask = cv2.copyMakeBorder(mask,top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=[0, 0, 0])
    for y in range(cleaningmask.shape[0]):
        for x in range(cleaningmask.shape[1]):
            if cleaningmask[y][x]== 0:
                cleaningmask = cv2.floodFill( cleaningmask, mask, (x,y), 0, 0, 100)[1]
    cleaningmask = cv2.threshold(cleaningmask,0,255,cv2.THRESH_BINARY)[1]
    for y in range(cleaningmask.shape[0]):
        for x in range(cleaningmask.shape[1]):
            if cleaningmask[y][x]== 255:
                dst[y][x] = 255
    return dst

def DoubleOtsu(input_img,big_t = 0.9,small_t = 0.5):
    dst1 = adaptiveOtsu(gray,3,big_t)
    dst2 = adaptiveOtsu(gray,3,small_t)
    dst = confirmedThreshold(dst1,dst2)
    return dst

def bottom_hat(input_img,doubleotsu_img,closings = 5):
    bottom_hat = np.zeros((input_img.shape[0],input_img.shape[1]), dtype=np.uint8)
    closed = cv2.dilate(cv2.erode(input_img,k,iterations=closings),k,iterations = closings)
    for i in range(bottom_hat.shape[0]):
        for j in range(bottom_hat.shape[1]):
            bottom_hat[i][j] = int(closed[i][j])-int(input_img[i][j])
    confirmer = cv2.threshold(doubleotsu_img,200,255,cv2.THRESH_BINARY)[1]
    bottom_hat = confirmedThreshold(bottom_hat,confirmer)
    return bottom_hat

def findfg(borders,img,erode=0, adaptive = False):
    if erode != 0:
        borders = cv2.erode(borders,cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5)), iterations = 1)
    distance = cleanmask(cv2.distanceTransform(borders,cv2.DIST_L2,5).astype(np.uint8),img)
    if adaptive :
        surefg = adaptiveOtsu(distance,3,0.9)
    else:
        surefg = cv2.threshold(distance,distance.min() ,distance.max(), cv2.THRESH_OTSU)[1]
    surefg = surefg.astype(np.uint8)
    return surefg

def cleanmask(toclean,img, color = [0,255,0]):
    h,w = toclean.shape[:2]
    for i in range(h):
        for j in range(w):
            if list(img[i,j])== color:
                toclean[i,j]= 0
    return toclean

def find_centroid(bwimage):
    h2,w2 = bwimage.shape
    arealen = 0
    xsum = 0
    ysum = 0
    for i in range(h2):
        for j in range(w2):
            if bwimage[i,j]==255:
                xsum+=j
                ysum+=i
                arealen +=1
    return [xsum/arealen,ysum/arealen]

def rerangemarkers(markers,h,w):
    for i in range(h):
        for j in range(w):
            currentpix = markers[i,j]
            if currentpix in [1,-1]:
                markers[i,j] = 0
            else:
                markers[i,j]=255
    markers = cleanmask(markers,img)
    return markers.astype(np.uint8)

def grain_size_dis(x):
    x_sort = sorted(x)
    y_label=[]
    for enu,i in enumerate(x_sort):
        y = (enu+1)/len(x_sort)
        y = '{:.0%}'.format(y)
        y_label.append(y)
    #print(y_label)
    plt.Figure()
    fig,ax = plt.subplots(1,1)
    plt.xscale('log')
    plt.plot(x_sort,y_label)
    plt.xlabel('grain size')
    plt.ylabel('ratio')
    plt.title('Particle size curve')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=9))
    plt.savefig('颗粒级配曲线.png')
    plt.show()

type = sys.getfilesystemencoding()
#sys.stdout = Logger("count.txt")
print("welcome to GRADELAND! (GRAin DEtection for LANDslides)\t")     #打印“hello！”，验证模块导入成功
#print('请输入比例尺')
ratio = float(1)

#img = cv2.imread("lishi2.png")  #导入图片，图片放在程序所在目录
img = filedialog.askopenfilename()
img = cv2.imread(img)
root.destroy()
#cv2.namedWindow("imagshow", 2)   #创建一个窗口
#cv2.imshow('imagshow', img)    #显示原始图片

#使用局部阈值的大津算法进行图像二值化
#dst = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,101, 1)

# 选择roi区域
roi = cv2.selectROI(windowName="original", img=img, showCrosshair=True, fromCenter=False)  # 选择ROI区域
x, y, w, h = roi  # 将选择的roi区域转换成坐标，x,y为ROI的坐标，w,h为ROI的大小
print(roi)  # 输出roi区域的坐标

# 显示ROI并保存图片
if roi != (0, 0, 0, 0):
    img = img[y:y + h, x:x + w]  # 切片获取roi区域
    cv2.imshow('crop', img)  # 显示roi区域
    # cv2.imwrite('crop.jpg', crop)  # 保存roi区域
    # print('Saved!')  # 输出保存成功

cv2.waitKey(0)  # 边框等待时长
cv2.destroyAllWindows()  # 关闭所有边框

print('图像比例尺是',ratio)
def on_tracebar_changed(args):
    pass
def on_buttom_changed(args):
    pass
print('选择好阈值后按q退出')

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
#cv2.createTrackbar('thres','Image',-10,10,on_tracebar_changed)
#cv2.createTrackbar('erosion','Image',0,5,on_tracebar_changed)
#cv2.createTrackbar('dilation','Image',0,5,on_tracebar_changed)
cv2.createTrackbar('inverse','Image',0,1,on_tracebar_changed)
#cv2.createTrackbar('min_rec','Image',0,50,on_tracebar_changed)
#cv2.createTrackbar('Alpha', 'Image', 0, 300, on_tracebar_changed)
k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

h, w = gray.shape[:2]
toshow = gray.copy()

while True:
    time.sleep(0.1)
    inverse = cv2.getTrackbarPos('inverse', 'Image')
    #min_rec = cv2.getTrackbarPos('min_rec', 'Image')
    if inverse == 1:
        toshow = cv2.bitwise_not(gray)
    cv2.imshow('Image',toshow)
    key = cv2.waitKey(0)
    if key == ord('q') :
        break
if inverse == 1:
    gray = cv2.bitwise_not(gray)

cv2.destroyAllWindows()

dst = DoubleOtsu(gray)
bottom_hat = bottom_hat(gray,dst)
canny = confirmedThreshold(cv2.bitwise_not(cv2.Canny(gray,50,150)),cv2.threshold(dst,200,255,cv2.THRESH_BINARY)[1])
canny = cv2.threshold(canny,200,255,cv2.THRESH_BINARY)[1]
surefg = findfg(canny,img, adaptive = False)
unknown = cv2.subtract(cv2.threshold(dst,200,255,cv2.THRESH_BINARY)[1],surefg)
markers = cv2.connectedComponents(surefg)[1]
markers = markers+1
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
marked = img.copy()
marked[markers == -1] = [0,0,255]

#cv2.imshow("bottom hat",bottom_hat)
#cv2.imshow("original",img)
#cv2.imshow("surefg",surefg)
#cv2.imshow("Canny",canny)
#cv2.imshow("dst",dst)


rectangles = False
colorize = True
centroids = True
rectcolor = [0,170,255]
centroidcolor = [197,97,255]

h,w = markers.shape


markers = rerangemarkers(markers,h,w)

rectangular = img.copy()
if colorize:
    colorful = np.zeros((h,w,3), dtype="int8")
    counter = 0
    colorpool = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[0,170,255],[0,255,200],[0,255,106],[0,140,255],[115,0,255],[255,0,166],[145,149,255],[153,255,145]]
    colorful[:,:] = (0,0,0)  # (B, G, R)
if centroids:
    centrimg = marked.copy()
    centroid_list = []

for i in range(h):
    for j in range(w):
        if markers[i,j] == 255:
            temporary = np.zeros((h,w), dtype=np.uint8)
            mask = cv2.bitwise_not(cv2.copyMakeBorder(markers.astype(np.uint8),top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=255))
            retval, temporary, mask2, rect = cv2.floodFill(temporary,mask,[j,i],255,10,10)
            markers = cv2.floodFill(markers,cv2.copyMakeBorder(np.zeros((h,w),dtype=np.uint8),top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=0),[j,i],0,10,10)[1]
            if rectangles:
                rectangular = cv2.rectangle(rectangular, [rect[0],rect[1]], [rect[0]+rect[2],rect[1]+rect[3]], rectcolor, 1)
            if colorize:
                colorful[temporary==255] = colorpool[counter]
                counter+=1
                if counter == len(colorpool):
                    counter = 0
            if centroids:
                subimage = temporary.copy()[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
                centroid_loc = find_centroid(subimage)
                centroid = [int(centroid_loc[0])+rect[0],int(centroid_loc[1])+rect[1]]
                centrimg = cv2.rectangle(centrimg, [centroid[0]-1,centroid[1]-1], [centroid[0]+1,centroid[1]+1], centroidcolor, 2)

if rectangles:
    cv2.imshow('rectangles',rectangular)
if colorize:
    cv2.imshow('colorful',colorful.astype(np.uint8))
if centroids:
    cv2.imshow('centroids',centrimg.astype(np.uint8))

input("waiting to show the fill operation")
cv2.destroyAllWindows()