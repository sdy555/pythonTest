import cv2 as cv
import numpy as np
# 角点检测 cornerHarris
#.img:数据类型为float32的入图像
# .blockSize:角点检测中指定区域的大小
# .ksize: Sobel求导中使用的窗口大小
# .k:取值参数为[0,04,0.06]
img = cv.imread('D:/aiImg/chessboard.jpg')
print('img.shape',img.shape)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
dst=cv.cornerHarris(gray,2,3,0.04)
print('dst.shape',dst.shape)

img[dst>0.01*dst.max()]=[0,0,225]
cv.imshow('dst',img)
cv.waitKey(0)
