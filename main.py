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
    dst1 = adaptiveOtsu(input_img,3,big_t)
    dst2 = adaptiveOtsu(input_img,3,small_t)
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

def cleanmask(toclean,img, color = [0,255,0]):
    h,w = toclean.shape[:2]
    for i in range(h):
        for j in range(w):
            if list(img[i,j])== color:
                toclean[i,j]= 0
    return toclean

def find_centroid(bwimage):
    def calclen(pointlist,centroid):
        if len(pointlist)==0:
            return 1, [[int(centroid[0]),int(centroid[1])],[int(centroid[0]),int(centroid[1])]]
        else:
            sortedp = sorted(pointlist, key=lambda point: point[0])
            if sortedp[0][0]==sortedp[-1][0]:
                sortedp = sorted(pointlist, key=lambda point: point[1])
            return ((sortedp[-1][0]-sortedp[0][0])**2+(sortedp[-1][1]-sortedp[0][1])**2)**0.5, [[sortedp[0][0],sortedp[0][1]],[sortedp[-1][0],sortedp[-1][1]]]
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
    centroid = [xsum/arealen,ysum/arealen]
    slopes = [[0.5,1,2],[-2,-1,-0.5]]
    lines = [[],[]]
    for i in range(len(slopes)):
        for j in range(len(slopes[i])+1):
            lines[i].append([])
    for i in range(h2):
        for j in range(w2):
            if bwimage[i,j]==255:
                dx = j-centroid[0]
                dy = i-centroid[1]
                relativept = [j,i]
                if abs(dx) <= 1:
                    lines[1][0].append(relativept)
                elif abs(dy) <= 1:
                    lines[0][0].append(relativept)
                else:
                    for h in range(len(slopes)):
                        for k in range(len(slopes[h])):
                            heightline = int(slopes[h][k]*dx)
                            if int(dy) == heightline:
                                lines[h][k+1].append(relativept)
    ratios = []
    axelens = {'Lmax': [], 'Lmin': []}
    for i in range(len(lines[0])):
        axe1,lines[0][i]= calclen(lines[0][i],centroid)
        axe2,lines[1][i] = calclen(lines[1][i],centroid)
        if min(axe1,axe2)!= 0:
            ratios.append(max(axe1,axe2)/min(axe1,axe2))
        else:
            ratios.append(0)
        axelens['Lmax'].append(max(axe1,axe2))
        axelens['Lmin'].append(min(axe1,axe2))
    realaxe = ratios.index(max(ratios))
    lines[0] = lines[0][realaxe]
    lines[1] = lines[1][realaxe]
    axelens['Lmax'] = axelens['Lmax'][realaxe]
    axelens['Lmin'] = axelens['Lmin'][realaxe]
    return centroid,lines, arealen, axelens

def exportGSD(markers,img, basename, centroidcolor = [197,97,255]):
    csvdata = {"area":[],"Lmax":[],"Lmin":[]}
    h,w = markers.shape
    centrimg = img.copy()
    centrimg[markers == -1] = centroidcolor
    markers = rerangemarkers(markers,h,w)
    for i in range(h):
        for j in range(w):
            if markers[i,j] == 255:
                temporary = np.zeros((h,w), dtype=np.uint8)
                mask = cv2.bitwise_not(cv2.copyMakeBorder(markers.astype(np.uint8),top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=255))
                retval, temporary, mask, rect = cv2.floodFill(temporary,mask,[j,i],255,10,10)
                subimage = temporary.copy()[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
                centroid_loc,centr_lines, arealen, axelens = find_centroid(subimage)
                csvdata["area"].append(arealen)
                csvdata["Lmax"].append(axelens["Lmax"])
                csvdata["Lmin"].append(axelens["Lmin"])
                centroid = [int(centroid_loc[0])+rect[0],int(centroid_loc[1])+rect[1]]
                markers = cv2.floodFill(markers,cv2.copyMakeBorder(np.zeros((h,w),dtype=np.uint8),top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=0),[j,i],0,10,10)[1]
                centrimg = cv2.rectangle(centrimg, [centroid[0]-1,centroid[1]-1], [centroid[0]+1,centroid[1]+1], centroidcolor, 2)
                for k in range(len(centr_lines)):
                    cv2.line(centrimg, np.add(np.array(centr_lines[k][0]),np.array(rect[:2])), np.add(np.array(centr_lines[k][1]),np.array(rect[:2])), centroidcolor, 2)
    df = pd.DataFrame.from_dict(csvdata)
    df.to_csv(basename+'.csv', index=False)
    cv2.imshow("detection",centrimg)
    input("check the output")
    return  

def rerangemarkers(markers,h,w):
    for i in range(h):
        for j in range(w):
            currentpix = markers[i,j]
            if currentpix in [1,-1]:
                markers[i,j] = 0
            else:
                markers[i,j]=255
    return markers.astype(np.uint8)

def separate_img(insidefunc, markers, img):
    outputlist = []
    #markers = rerangemarkers(markers,h,w)
    for i in range(h):
        for j in range(w):
            if markers[i,j] == 255:
                temporary = np.zeros((h,w), dtype=np.uint8)
                mask = cv2.bitwise_not(cv2.copyMakeBorder(markers.astype(np.uint8),top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=255))
                retval, temporary, mask2, rect = cv2.floodFill(temporary,mask,[j,i],255,10,10)
                outputlist.append([insidefunc(temporary[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]],img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]),rect])
                markers = cv2.floodFill(markers,cv2.copyMakeBorder(np.zeros((h,w),dtype=np.uint8),top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=0),[j,i],0,10,10)[1]
    return outputlist

def reunite_img(outputlist,h,w):
    toreunite = np.zeros((h,w),dtype=np.uint8)
    for i in outputlist:
        h2,w2 = i[0].shape[:2]
        for a in range(h2):
            for b in range(w2):
                    if i[0][a,b]!=0:
                        toreunite[a+i[1][1],b+i[1][0]]=255
    return toreunite

def DistanceBasedInside(mask,img,erode=3):
    if len(img.shape)>=3:
        mask = cleanmask(mask,img)
    if erode != 0:
        mask = cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_CROSS, (erode,erode)), iterations = 1)
    distance = cv2.distanceTransform(mask,cv2.DIST_L2,5).astype(np.uint8)
    surefg = cv2.threshold(distance,0 ,255, cv2.THRESH_OTSU)[1]
    return surefg

def FragAnalFg(func,dividing,img,gray,h,w):
    countmask = cv2.countNonZero(cv2.inRange(img, np.array([0, 250, 0]), np.array([10, 255, 10])))
    if countmask ==0:
        markers = gray
    else:
        markers = img
    outputlist = separate_img(func, dividing, markers)
    return reunite_img(outputlist,h,w), outputlist

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

def colordetect(markers,img):
    h,w = markers.shape
    colorful = np.zeros((h,w,3), dtype=np.uint8)
    counter = 0
    colorpool = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[0,170,255],[0,255,200],[0,255,106],[0,140,255],[115,0,255],[255,0,166],[145,149,255],[153,255,145]]
    markers = rerangemarkers(markers,h,w)
    for i in range(h):
        for j in range(w):
            if markers[i,j] == 255:
                temporary = np.zeros((h,w), dtype=np.uint8)
                mask = cv2.bitwise_not(cv2.copyMakeBorder(markers.astype(np.uint8),top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=255))
                retval, temporary, mask, rect = cv2.floodFill(temporary,mask,[j,i],255,10,10)
                markers = cv2.floodFill(markers,cv2.copyMakeBorder(np.zeros((h,w),dtype=np.uint8),top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=0),[j,i],0,10,10)[1]
                colorful[temporary==255] = colorpool[counter]
                counter+=1
                if counter == len(colorpool):
                    counter = 0
    return cv2.addWeighted(colorful, 0.25, img, 0.75, 0)

def merge_split(window_name,markers,surefg,surebg,img):
    loadpic = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)),"loading.png"))
    colored = colordetect(markers,img)
    cv2.namedWindow(window_name)

    def mouse_callback(event, x, y,flags,params):
        if event == cv2.EVENT_LBUTTONDOWN:
            toMerge[-1].append((x, y))
            if len(toMerge[-1]) == 2:
                toMerge.append([])
                return
        if event == cv2.EVENT_RBUTTONDOWN:
            toSplit[-1].append((x, y))
            if len(toSplit[-1]) == 2:
                toSplit.append([])
                return

    cv2.setMouseCallback(window_name, mouse_callback)
    toMerge = [[]]
    toSplit = [[]]
    while True:
        cv2.imshow(window_name, colored)
        cv2.imshow("original",img)
        key = cv2.waitKey(1)
        if key == 13:
            cv2.imshow('loading', loadpic)
            cv2.waitKey(1)
            if len(toMerge[0])!= 0:
                del toMerge[-1]
                print("uniting...",toMerge)
                for i in range(len(toMerge)):
                    cv2.line(surefg,toMerge[i][0],toMerge[i][1],255,2)
                unknown = cv2.subtract(surebg,surefg)
                markers = cv2.connectedComponents(surefg)[1]
                markers = markers+1
                markers[unknown==255] = 0
                markers = cv2.watershed(img,markers)
                colored = colordetect(markers,img)
                toMerge = [[]]
                print("done")
            if len(toSplit[0])!=0:
                del toSplit[-1]
                print("dividing...",toSplit)
                mask = cv2.bitwise_not(cv2.copyMakeBorder(markers.astype(np.uint8),top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=255))
                for i in range(len(toSplit)):
                    surefg = cv2.floodFill(surefg,mask,toSplit[i][0],0,0,255)[1]
                for i in range(len(toSplit)):
                    for j in range(len(toSplit[i])):
                        cv2.line(surefg, toSplit[i][j], toSplit[i][j], 255, 2)
                unknown = cv2.subtract(surebg,surefg)
                markers = cv2.connectedComponents(surefg)[1]
                markers = markers+1
                markers[unknown==255] = 0
                markers = cv2.watershed(img,markers)
                colored = colordetect(markers,img)
                toSplit = [[]]
                print("done")
            if len(toMerge[0])==0 and len(toSplit[0])==0:
                cv2.destroyWindow('loading')
        if key == 27:
            break
    unknown = cv2.subtract(surebg,surefg)
    markers = cv2.connectedComponents(surefg)[1]
    markers = markers+1
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    cv2.destroyAllWindows()
    return surefg,unknown,markers,colored

def show_popup(title,innertext,buttontext = "Close"):
    popup_window = tk.Tk()
    popup_window.wm_title(title)
    label = tk.Label(popup_window, text=innertext)
    label.pack(side="top", fill="x", pady=10)
    button = tk.Button(popup_window, text=buttontext, command= popup_window.destroy)
    button.pack()
    popup_window.mainloop()

type = sys.getfilesystemencoding()

ratio = float(1)
show_popup("你好！","welcome to GRADELAND! \n (GRAin DEtection for LANDslides)","开始！")
pathfile = filedialog.askopenfilename()
basename = os.path.splitext(pathfile)[0]
img = cv2.imread(pathfile)

# 选择roi区域
roi = cv2.selectROI(windowName="select ROI", img=img, showCrosshair=True, fromCenter=False)  # 选择ROI区域
x, y, w, h = roi  # 将选择的roi区域转换成坐标，x,y为ROI的坐标，w,h为ROI的大小
print('roi区域的坐标是 ',roi)  # 输出roi区域的坐标

# 显示ROI并保存图片
if roi != (0, 0, 0, 0):
    img = img[y:y + h, x:x + w]  # 切片获取roi区域
    cv2.imshow('crop', img)  # 显示roi区域
    # cv2.imwrite('crop.jpg', crop)  # 保存roi区域
    # print('Saved!')  # 输出保存成功

cv2.destroyAllWindows() 

#print('图像比例尺是',ratio)
def on_tracebar_changed(args):
    pass
def on_buttom_changed(args):
    pass
print('选择好阈值后按q退出')

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
cv2.createTrackbar('inverse','Image',0,1,on_tracebar_changed)
k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

h, w = gray.shape[:2]
toshow = gray.copy()

while True:
    time.sleep(0.1)
    inverse = cv2.getTrackbarPos('inverse', 'Image')
    #min_rec = cv2.getTrackbarPos('min_rec', 'Image')
    if inverse == 1:
        toshow = cv2.bitwise_not(gray)
    else:
        toshow = gray.copy()
    cv2.imshow('Image',toshow)
    key = cv2.waitKey(1)
    if key == ord('q') :
        break
if inverse == 1:
        gray = cv2.bitwise_not(gray)

cv2.destroyAllWindows()

start = time.time()
dst = DoubleOtsu(gray)
#bottom_hat = bottom_hat(gray,dst)
#canny = confirmedThreshold(cv2.bitwise_not(cv2.Canny(gray,50,150)),cv2.threshold(dst,200,255,cv2.THRESH_BINARY)[1])
#canny = cv2.threshold(canny,200,255,cv2.THRESH_BINARY)[1]
#dst = cv2.addWeighted(dst, 0.5, canny, 0.5, 0)
surefg = FragAnalFg(DistanceBasedInside,cv2.threshold(dst,200,255,cv2.THRESH_BINARY)[1],img,gray,h,w)[0]
surebg =  cleanmask(cv2.threshold(dst,0,255,cv2.THRESH_BINARY)[1],img)
unknown = cv2.subtract(surebg,surefg)
markers = cv2.connectedComponents(surefg)[1]
markers = markers+1
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
end = time.time()
print("entire process lasted", end-start)

surefg,unknown,markers,colored = merge_split("prova",markers,surefg,surebg,img)
exportGSD(markers,img, basename)