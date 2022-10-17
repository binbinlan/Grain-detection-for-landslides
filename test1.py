import cv2  # 调用opencv
import matplotlib.pyplot as plt
import numpy.random
import pandas as pd
import numpy as np


# imgpath = 'lishi2.png'  # 写入路径
# img = cv2.imread(imgpath)  # 读取照片
# cv2.imshow('original', img)  # 显示照片

# # 选择roi区域
# roi = cv2.selectROI(windowName="original", img=img, showCrosshair=True, fromCenter=False)  # 选择ROI区域
# x, y, w, h = roi  # 将选择的roi区域转换成坐标，x,y为ROI的坐标，w,h为ROI的大小
# print(roi)  # 输出roi区域的坐标
#
# # 显示ROI并保存图片
# if roi != (0, 0, 0, 0):
#     crop = img[y:y + h, x:x + w]  # 切片获取roi区域
#     cv2.imshow('crop', crop)  # 显示roi区域
#     cv2.imwrite('crop.jpg', crop)  # 保存roi区域
#     print('Saved!')  # 输出保存成功
#
# cv2.waitKey(0)  # 边框等待时长
# cv2.destroyAllWindows()  # 关闭所有边框


x = [0.1,0.2,0.3,0.1,0.5,30,10,11,2.8]
# def grain_size_dis(x):
#     sorted_index = 0
#     index = sorted(x)
#     for n in index:
#         for _,i in enumerate(x):
#             if x[_] == n :
#                 sorted_index = _
#             print(sorted_index)

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
    plt.xscale('log')
    plt.plot(x_sort,y_label)
    plt.show()

grain_size_dis(x)
