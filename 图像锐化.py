#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import cv2
import cv2 as cv
import numpy as np
# from skimage.color import rgb2yuv
#
#
# def custom_blur_demo(image):
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
#     dst = cv.filter2D(image, -1, kernel=kernel)
#     cv.imshow("custom_blur_demo", dst)
#
#
# src = cv.imread("D:/aiImg/588.jpg")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)
# custom_blur_demo(src)
# cv.waitKey(0)
# cv.destroyAllWindows()

import skimage.io
# from PIL import ImageEnhance
# from PIL import Image #Image类是PIL中的核心类
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = Image.open('D:/aiImg/588.jpg')
# #图像增强
# img_color = ImageEnhance.Color(img).enhance(0.1) #颜色增强,增强因子决定图像的颜色饱和度，为0.0将产生黑白图像；为1.0将给出原始图像
# im_bright = ImageEnhance.Brightness(img).enhance(0.2) #用于调整图像的亮度
# im_contrast = ImageEnhance.Contrast(img).enhance(0.5) #用于调整图像的对比度
# im_sharp =ImageEnhance.Sharpness (img).enhance(2.0) #增强因子为0.0将产生模糊图像；为1.0将保持原始图像，为2.0将产生锐化过的图像
# plt.subplot(121), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
# plt.subplot(122), plt.imshow(im_sharp, 'gray'), plt.title('hhh')
# plt.show()

#导入库
# import cv2
# import numpy as np
# #导入图片
# img=cv2.imread("D:/aiImg/588.jpg")
# #转换灰度
# gimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #拉普拉斯算子锐化
# # [0, -1, 0], [-1, 5, -1], [0, -1, 0]
# kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)#定义拉普拉斯算子
# dst=cv2.filter2D(img,-1,kernel=kernel)#调用opencv图像锐化函数
# #sobel算子锐化
# #对x方向梯度进行sobel边缘提取
# x=cv2.Sobel(gimg,cv2.CV_64F,1,0)
# #对y方向梯度进行sobel边缘提取
# y=cv2.Sobel(gimg,cv2.CV_64F,0,1)
# #对x方向转回uint8
# absX=cv2.convertScaleAbs(x)
# #对y方向转会uint8
# absY=cv2.convertScaleAbs(y)
# #x，y方向合成边缘检测结果
# dst1=cv2.addWeighted(absX,0.5,absY,0.5,0)
# #与原图像堆叠
# res=dst1+gimg
# #测试
# #print("dstshape:",dst1)
# #print("resshape:",res)
# #按要求左右显示原图与拉普拉斯处理结果
# imges1=np.hstack([img,dst])
# cv2.imshow('lapres',imges1)
# #按要求左右显示原图与sobel处理结果
# # image=np.hstack([gimg,res])
# # cv2.imshow('sobelres',image)
# #去缓存
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# from PIL import Image
# import numpy as np
#
# # 读入原图像
# img = Image.open('D:/aiImg/588.jpg')
# # img.show()
#
# # 为了减少计算的维度，因此将图像转为灰度图
# img_gray = img.convert('L')
# img_gray.show()
#
# # 得到转换后灰度图的像素矩阵
# img_arr = np.array(img_gray)
# h = img_arr.shape[0]  # 行
# w = img_arr.shape[1]  # 列
#
# # 拉普拉斯算子锐化图像，用二阶微分
# new_img_arr = np.zeros((h, w))  # 拉普拉斯锐化后的图像像素矩阵
# for i in range(2, h-1):
#     for j in range(2, w-1):
#         new_img_arr[i][j] = img_arr[i+1, j] + img_arr[i-1, j] + \
#                             img_arr[i, j+1] + img_arr[i, j-1] - \
#                             4*img_arr[i, j]
#
# # 拉普拉斯锐化后图像和原图像相加
# laplace_img_arr = np.zeros((h, w))  # 拉普拉斯锐化图像和原图像相加所得的像素矩阵
# for i in range(0, h):
#     for j in range(0, w):
#         laplace_img_arr[i][j] = new_img_arr[i][j] + img_arr[i][j]
#
# img_laplace = Image.fromarray(np.uint8(new_img_arr))
# img_laplace.show()
#
# img_laplace2 = Image.fromarray(np.uint8(laplace_img_arr))
# img_laplace2.show()




# 拉普拉斯算子 锐化图像
# from matplotlib import pyplot as plt
#
# lapimg = cv2.imread("D:/aiImg/588.jpg")
# h=cv2.imread("D:/aiImg/588.jpg")
# # 定义图片的参数
# height = lapimg.shape[0]
# width = lapimg.shape[1]
#
# # 定义拉普拉斯滤波器
# lapVector = np.array([[0, 1, 0],
#                       [1, -4, 1],
#                       [0, 1, 0]])
#
# # 定义输出图像矩阵
# outimg = np.zeros((3, height, width)).astype(np.float32)
#
# # 用拉普拉斯算子锐化图像
# for channel in range(3):
#     for i in range(height):
#         for j in range(width):
#             if (i >= 1 and j >= 1 and i < height - 1 and j < width - 1):
#                 # 找到中心坐标左上角的坐标
#                 initial_x = i - 1
#                 initial_y = j - 1
#
#                 # 用于记录每次滤波的结果
#                 curnum = 0.0
#
#                 # 对本窗口进行滤波
#                 for m in range(3):
#                     for n in range(3):
#                         # 二阶微分
#                         curnum += (float)(lapVector[m][n] * lapimg[initial_x + m][initial_y + n][channel] * 0.1)
#
#                 # 这里一定要对计算结果取绝对值，因为若为负值会使图片颜色溢出
#                 outimg[channel][i][j] = np.abs(curnum)
#
# resimg = cv2.merge([outimg[0], outimg[1],
#                     outimg[2]])
# resimg /= 255.0
# plt.subplot(221)
# plt.imshow(resimg)
# plt.subplot(222)
# plt.imshow(h)
# plt.subplot(223)
# plt.imshow(lapimg)
#
# plt.subplot(224)
# addimg = (resimg + lapimg)
# addimg = addimg.astype(np.int)
# plt.imshow(addimg)
#
# plt.show()


import cv2 as cv
import numpy as np
from cv2 import CV_8U, CV_16S

src = cv.imread('D:/aiImg/588.jpg')
# cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
# cv.imshow("input", src)

# sharpen_op = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
sharpen_op = np.array([
        [-1,-1,-1,-1,-1],
        [-1,2,2,2,-1],
        [-1,2,8,2,-1],
        [-1,2,2,2,-1],
        [-1,-1,-1,-1,-1]])/8.0
sharpen_image = cv.filter2D(src, ddepth=CV_16S, kernel=sharpen_op)
sharpen_image = cv.convertScaleAbs(sharpen_image)
# cv.imshow("sharpen_image", sharpen_image)

h, w = src.shape[:2]
result = np.zeros([h, w*2, 3], dtype=src.dtype)
result[0:h,0:w,:] = src
result[0:h,w:2*w,:] = sharpen_image
cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
cv.putText(result, "sharpen image", (w+10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
cv.imshow("sharpen_image", result)
# cv.imwrite("result.png", result)

cv.waitKey(0)
cv.destroyAllWindows()



# import cv2
# import numpy as np
# #加载图像
# image = cv2.imread('D:/aiImg/51.png')
# #自定义卷积核
# kernel_sharpen_1 = np.array([
#         [-1,-1,-1],
#         [-1,9,-1],
#         [-1,-1,-1]])
# kernel_sharpen_2 = np.array([
#         [1,1,1],
#         [1,-7,1],
#         [1,1,1]])
# kernel_sharpen_3 = np.array([
#         [-1,-1,-1,-1,-1],
#         [-1,2,2,2,-1],
#         [-1,2,8,2,-1],
#         [-1,2,2,2,-1],
#         [-1,-1,-1,-1,-1]])/8.0
# #卷积
# output_1 = cv2.filter2D(image,-1,kernel_sharpen_1)
# output_2 = cv2.filter2D(image,-1,kernel_sharpen_2)
# output_3 = cv2.filter2D(image,-1,kernel_sharpen_3)
# #显示锐化效果
# cv2.imshow('Original Image',image)
# cv2.imshow('sharpen_1 Image',output_1)
# cv2.imshow('sharpen_2 Image',output_2)
# cv2.imshow('sharpen_3 Image',output_3)
# #停顿
# if cv2.waitKey(0) & 0xFF == 27:
#     cv2.destroyAllWindows()


#导入cv模块
import cv2
import numpy as np
from matplotlib import pyplot as plt


#1.1、直方图处理
#导入本地图片
# inputPic1 = cv2.imread('D:/aiImg/588.jpg',-1)
# #直方图均衡化处理之后，灰度范围变大，对比度变大，清晰度变大，所以能有效增强图像。
# zhiFangTuPic = cv2.equalizeHist(inputPic1)
# #把两张图片展示在一起，使用numpy的矩阵堆叠方法
# result1 = np.hstack((inputPic1,zhiFangTuPic))
# # 输出
# cv2.imshow('inputPic',result1)



#1.2、彩色直方图均衡化
#直方图均衡化处理之后，灰度范围变大，对比度变大，清晰度变大，所以能有效增强图像。
# inputPic2 = cv2.imread('D:/aiImg/588.jpg',1)
# (b,g,r) = cv2.split(inputPic2) #色彩通道分解
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# temp = cv2.merge((bH,gH,rH),)#通道合成
# #将输入图片和输出图片进行合并，方便观察
# result2 = np.hstack((inputPic2,temp))
# #展示图片
# cv2.imshow('inputPic',result2)



# #2.1、均值滤波
# inputPic3 = cv2.imread('pic\\2.jpg',1)
# junzhilvboPic = cv2.blur(inputPic3, (5,5))
# #将输入图片和输出图片进行合并，方便观察
# result3 = np.hstack((inputPic2,junzhilvboPic))
#cv2.imshow('inputPic',result3)



#2.2、中值滤波
# inputPic4 = cv2.imread('pic\\2.jpg',1)
# zhongzhilvboPic = cv2.medianBlur(inputPic4, 5)
# #将输入图片和输出图片进行合并，方便观察
# result4 = np.hstack((inputPic4,zhongzhilvboPic))
#打印图片
#cv2.imshow('inputPic',result4)



#3.1、锐化滤波器-梯度算子
# inputPic6 = cv2.imread('D:/aiImg/588.jpg',1)
# # 水平梯度
# sobelX = cv2.Sobel(inputPic6, cv2.CV_64F, 1, 0)
# sobelX = np.uint8(np.absolute(sobelX))
# #cv2.imshow("Sobel X", sobelX)
# # 垂直梯度
# sobelY = cv2.Sobel(inputPic6, cv2.CV_64F, 0, 1)
# sobelY = np.uint8(np.absolute(sobelY))
# #cv2.imshow("Sobel Y", sobelY)
# # 结合x和y两个方向的梯度
# sobelCombined = cv2.bitwise_or(sobelX,sobelY)
# #cv2.imshow("Sobel Combined", sobelCombined)
# #合并打印
# result6 = np.hstack((inputPic6,sobelX,sobelY,sobelCombined))
# cv2.imshow("Sobel XY", result6)


#3.2、锐化滤波器-拉普拉斯算子
#加载图片
# inputPic5 = cv2.imread('D:/aiImg/588.jpg',0)
# #深度采用cv2.CV_16S
# gray_lap = cv2.Laplacian(inputPic5,cv2.CV_16S,ksize = 3)
# laplacian = cv2.convertScaleAbs(gray_lap)
# #合并图片
# result5 = np.hstack((inputPic5,laplacian))
# #打印
# cv2.imshow('inputPic',result5)
#
#
#
# dst = cv2.convertScaleAbs(gray_lap)
# laplacian=cv2.Laplacian(inputPic5,cv2.CV_64F,ksize = 3)#CV_16S为图像深度
# sobelx=cv2.Sobel(inputPic5,cv2.CV_64F,1,0,ksize=5)#1，0参数表示在x方向求一阶导数
# sobely=cv2.Sobel(inputPic5,cv2.CV_64F,0,1,ksize=5)#0,1参数表示在y方向求一阶导数
#将输入图片和输出图片进行合并，方便观察

# cv2.waitKey(0)
# cv2.destroyAllWindows()
