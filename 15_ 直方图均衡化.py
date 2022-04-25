import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import Common

# 均衡化
# img = cv.imread('D:/aiImg/cat.jpg',0)
# # plt.hist(img.ravel(),256)
# # plt.show()
#
# equ = cv.equalizeHist(img)# equalizeHist均衡化函数
# plt.hist(equ.ravel(),256)
# # plt.show()

# res = np.hstack((img,equ))
# Common.cv_show(res)

img = cv.imread('D:/aiImg/58.jpg',0)
plt.hist(img.ravel(),256)
equ = cv.equalizeHist(img)# equalizeHist均衡化函数
clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))# createCLAHE 自适应均衡化
res_clahe = clahe.apply(img)
res = np.hstack((img,equ,res_clahe))
Common.cv_show(res)