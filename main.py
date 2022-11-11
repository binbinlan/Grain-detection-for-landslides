# -*- coding:utf-8 -*-
#本程序用于大津算法的实现
'''
author: binbinlan, marcolovati
data: 2022.05.10
email:787250087@qq.com
'''

import time

import cv2  #导入opencv模块
import numpy as np
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



type = sys.getfilesystemencoding()
#sys.stdout = Logger("count.txt")
print("Hello binbinlan!\t")     #打印“hello！”，验证模块导入成功
print('请输入比例尺')
ratio = 1#input('')
ratio = float(ratio)

#img = cv2.imread("lishi2.png")  #导入图片，图片放在程序所在目录
img = filedialog.askopenfilename()
img = cv2.imread(img)
print("the image is ", len(img),len(img[0]))
root.destroy()
#cv2.namedWindow("imagshow", 2)   #创建一个窗口
#cv2.imshow('imagshow', img)    #显示原始图片

#gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #转换为灰度图

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
cv2.createTrackbar('thres','Image',0,255,on_tracebar_changed)
cv2.createTrackbar('erosion','Image',0,5,on_tracebar_changed)
cv2.createTrackbar('dilation','Image',0,5,on_tracebar_changed)
#cv2.createTrackbar('inverse','Image',0,1,on_tracebar_changed)
#cv2.createTrackbar('min_rec','Image',0,500,on_tracebar_changed)
#cv2.createTrackbar('Alpha', 'Image', 0, 300, on_tracebar_changed)



while True:
    time.sleep(0.1)
    thresh = cv2.getTrackbarPos('thres','Image')
    erosion = cv2.getTrackbarPos('erosion','Image')
    dilation = cv2.getTrackbarPos('dilation', 'Image')
    inverse = 0#cv2.getTrackbarPos('inverse', 'Image')
    min_rec = 10#cv2.getTrackbarPos('min_rec', 'Image')
    #Alpha = cv2.getTrackbarPos('Alpha', 'Image')
    struct = 3
    k = np.zeros((struct,struct),np.uint8)
    k[0] = [0,1,0]
    k[1] = [1,1,1]
    k[2] = [0,1,0]
    grays = [gray]
    grays.append(cv2.erode(grays[-1],k,iterations=erosion))
    grays.append(cv2.dilate(grays[-1],k,iterations=dilation))
    dst = cv2.threshold(grays[-1],thresh,255,cv2.THRESH_BINARY)[1]
    del grays
    #Alpha = Alpha * 0.01
    #img2 = cv2.convertScaleAbs(img,alpha=Alpha,beta=0)
    if inverse == 0:
        pass
    else:
        h, w = dst.shape[:2]
        imgInv = np.empty((w, h), np.uint8)
        for i in range(h):
            for j in range(w):
                dst[i][j] = 255 - dst[i][j]

    cv2.imshow('Image',dst)
    cv2.imshow('Src', img)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q') :
        break

cv2.destroyAllWindows()





#全局大津算法，效果较差
#res ,dst = cv2.threshold(gray,0 ,255, cv2.THRESH_OTSU)

#暂时不用
#element = cv2.getStructuringElement(cv2.MORPH_CROSS,(1, 1))#形态学去噪
#dst=cv2.morphologyEx(dst,cv2.MORPH_OPEN,element)  #开运算去噪

contours, hierarchy = cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #轮廓检测函数
cv2.drawContours(img,contours,-1,(120,120,0),1)  #绘制轮廓

count=0 #砾石总数
ares_avrg=0  #砾石平均
count_list = []
area_list = []
x_list = []
y_list = []
#遍历找到的所有砾石
for cont in contours:

    ares = cv2.contourArea(cont)#计算包围性状的面积
    #ares = float(ares)
    if ares<min_rec:   #过滤面积小于min_value的形状
        continue

    count+=1    #总体计数加1
    count_list.append(count)
    ares_avrg+=ares

    print("{}-砾石面积:{}".format(count,ares/(ratio*ratio)),end="  ") #打印出每个砾石的面积
    area_list.append(ares/(ratio*ratio))

    rect = cv2.boundingRect(cont) #提取矩形坐标
    x_list.append(rect[0])
    y_list.append(rect[1])
    print('rect',rect)

    print("x:{} y:{}".format(rect[0],rect[1]))#打印坐标

    cv2.rectangle(img,rect,(0,0,0xff),1)#绘制矩形

    y=10 if rect[1]<10 else rect[1] #防止编号到图片之外

    cv2.putText(img,str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1) #在砾石左上角写上编号

print("砾石平均面积:{}".format(round(ares_avrg/count,2)/(ratio*ratio))) #打印出每个砾石的面积


cv2.namedWindow("imagshow", cv2.WINDOW_NORMAL)   #创建一个窗口
cv2.imshow('imagshow', img)    #显示原始图片

#cv2.namedWindow("dst", cv2.WINDOW_KEEPRATIO)   #创建一个窗口
#cv2.imshow("dst", dst)  #显示灰度图


plt.hist(gray.ravel(), 256, [0, 256]) #计算灰度直方图
plt.title('histogram')
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()


colums_name = ['order','squre','x','y']
data_list = [count,area_list,x_list,y_list]
test = pd.DataFrame({'count':count_list,'area':area_list,'x':x_list,'y':y_list})
test.to_csv('statics.csv',index=False)


#sys.stdout = Logger('/media/linux/harddisk1/lst/hanhan/log')


def grain_size_dis(x):
    x_sort = sorted(x)
    #print(x_sort)
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


grain_size_dis(area_list)

