# import cv2
# import numpy as np
# # Load the image
# image = cv2.imread('D:/aiImg/huifu.png', -1)
# (hei,wid,_) = image.shape
# #Grayscale and blur the image
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (3,3), 0)
# #Threshold the image
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# #Retrieve contours
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# #Create box-list
# box = []
# # Get position (x,y), width and height for every contour
# for c in contours:
#     x, y, w, h = cv2.boundingRect(c)
#     box.append([x,y,w,h])
# # Create separate lists for all values
# heights = []
# widths = []
# xs = []
# ys = []
# # Store values in lists
# for b in box:
#     heights.append(b[3])
#     widths.append(b[2])
#     xs.append(b[0])
#     ys.append(b[1])
# # Retrieve minimum and maximum of lists
# min_height = np.min(heights)
# min_width = np.min(widths)
# min_x = np.min(xs)
# min_y = np.min(ys)
# max_y = np.max(ys)
# max_x = np.max(xs)
# #Retrieve height where y is maximum (edge at bottom, last row of table)
# for b in box:
#     if b[1] == max_y:
#         max_y_height = b[3]
# #Retrieve width where x is maximum (rightmost edge, last column of table)
# for b in box:
#     if b[0] == max_x:
#         max_x_width = b[2]
# # Obtain horizontal lines mask
# horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
# horizontal_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
# horizontal_mask = cv2.dilate(horizontal_mask, horizontal_kernel, iterations=9)
# # Obtain vertical lines mask
# vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,50))
# vertical_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
# vertical_mask= cv2.dilate(vertical_mask, vertical_kernel, iterations=9)
# # Bitwise-and masks together
# result = 255 - cv2.bitwise_or(vertical_mask, horizontal_mask)
# #Cropping the image to the table size
# crop_img = result[(min_y+5):(max_y+max_y_height), (min_x):(max_x+max_x_width+5)]
# #Creating a new image and filling it with white background
# img_white = np.zeros((hei, wid), np.uint8)
# img_white[:, 0:wid] = (255)
# #Retrieve the coordinates of the center of the image
# x_offset = int((wid - crop_img.shape[1])/2)
# y_offset = int((hei - crop_img.shape[0])/2)
# #Placing the cropped and repaired table into the white background
# img_white[ y_offset:y_offset+crop_img.shape[0], x_offset:x_offset+crop_img.shape[1]] = crop_img
# #Viewing the result
# cv2.imshow('Result', img_white)
# cv2.waitKey()


# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('D:/aiImg/huifu.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# mask = cv2.imread('D:/aiImg/huifu.png', 0)
#
# dst = cv2.inpaint(img, mask, 2, cv2.INPAINT_TELEA)
# dst1 = cv2.inpaint(img, mask, 2, cv2.INPAINT_NS)  # 3 是每个点染色的圆形邻域的半径
#
# plt.subplot(221)
# plt.imshow(img)
#
# plt.subplot(222)
# plt.imshow(mask)
#
# plt.subplot(223)
# plt.imshow(dst)
#
# plt.subplot(224)
# plt.imshow(dst1)
# plt.show()

# OpenCVdemo08.py
# Demo08 of OpenCV
# 8. 图像的频率域滤波
# Copyright 2021 Youcans, XUPT
# Crated：2021-12-15

# 8.22：GLPF 低通滤波案例：印刷文本字符修复 (Text character recognition)
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
#
# def gaussLowPassFilter(shape, radius=10):  # 高斯低通滤波器
#     # 高斯滤波器：# Gauss = 1/(2*pi*s2) * exp(-(x**2+y**2)/(2*s2))
#     u, v = np.mgrid[-1:1:2.0 / shape[0], -1:1:2.0 / shape[1]]
#     D = np.sqrt(u ** 2 + v ** 2)
#     D0 = radius / shape[0]
#     kernel = np.exp(- (D ** 2) / (2 * D0 ** 2))
#     return kernel
#
#
# def dft2Image(image):  # 最优扩充的快速傅立叶变换
#     # 中心化, centralized 2d array f(x,y) * (-1)^(x+y)
#     mask = np.ones(image.shape)
#     mask[1::2, ::2] = -1
#     mask[::2, 1::2] = -1
#     fImage = image * mask  # f(x,y) * (-1)^(x+y)
#
#     # 最优 DFT 扩充尺寸
#     rows, cols = image.shape[:2]  # 原始图片的高度和宽度
#     rPadded = cv2.getOptimalDFTSize(rows)  # 最优 DFT 扩充尺寸
#     cPadded = cv2.getOptimalDFTSize(cols)  # 用于快速傅里叶变换
#
#     # 边缘扩充(补0), 快速傅里叶变换
#     dftImage = np.zeros((rPadded, cPadded, 2), np.float32)  # 对原始图像进行边缘扩充
#     dftImage[:rows, :cols, 0] = fImage  # 边缘扩充，下侧和右侧补0
#     cv2.dft(dftImage, dftImage, cv2.DFT_COMPLEX_OUTPUT)  # 快速傅里叶变换
#     return dftImage
#
#
# # (1) 读取原始图像
# imgGray = cv2.imread("D:/aiImg/huifu.png", flags=0)  # flags=0 读取为灰度图像
# rows, cols = imgGray.shape[:2]  # 图片的高度和宽度
#
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 3, 1), plt.title("Original"), plt.axis('off'), plt.imshow(imgGray, cmap='gray')
#
# # (2) 快速傅里叶变换
# dftImage = dft2Image(imgGray)  # 快速傅里叶变换 (rPad, cPad, 2)
# rPadded, cPadded = dftImage.shape[:2]  # 快速傅里叶变换的尺寸, 原始图像尺寸优化
# print("dftImage.shape:{}".format(dftImage.shape))
#
# D0 = [10, 30, 60, 90, 120]  # radius
# for k in range(5):
#     # (3) 构建 高斯低通滤波 (Gauss low pass filter)
#     lpFilter = gaussLowPassFilter((rPadded, cPadded), radius=D0[k])
#
#     # (5) 在频率域修改傅里叶变换: 傅里叶变换 点乘 低通滤波器
#     dftLPfilter = np.zeros(dftImage.shape, dftImage.dtype)  # 快速傅里叶变换的尺寸(优化尺寸)
#     for j in range(2):
#         dftLPfilter[:rPadded, :cPadded, j] = dftImage[:rPadded, :cPadded, j] * lpFilter
#
#     # (6) 对低通傅里叶变换 执行傅里叶逆变换，并只取实部
#     idft = np.zeros(dftImage.shape[:2], np.float32)  # 快速傅里叶变换的尺寸(优化尺寸)
#     cv2.dft(dftLPfilter, idft, cv2.DFT_REAL_OUTPUT + cv2.DFT_INVERSE + cv2.DFT_SCALE)
#
#     # (7) 中心化, centralized 2d array g(x,y) * (-1)^(x+y)
#     mask2 = np.ones(dftImage.shape[:2])
#     mask2[1::2, ::2] = -1
#     mask2[::2, 1::2] = -1
#     idftCen = idft * mask2  # g(x,y) * (-1)^(x+y)
#
#     # (8) 截取左上角，大小和输入图像相等
#     result = np.clip(idftCen, 0, 255)  # 截断函数，将数值限制在 [0,255]
#     imgLPF = result.astype(np.uint8)
#     imgLPF = imgLPF[:rows, :cols]
#
#     plt.subplot(2, 3, k + 2), plt.title("GLPF rebuild(n={})".format(D0[k])), plt.axis('off')
#     plt.imshow(imgLPF, cmap='gray')
#
# print("image.shape:{}".format(imgGray.shape))
# print("lpFilter.shape:{}".format(lpFilter.shape))
# print("dftImage.shape:{}".format(dftImage.shape))
#
# plt.tight_layout()
# plt.show()

import cv2
from matplotlib import pyplot as plt

src_image = cv2.imread("D:/aiImg/huifu.png")
print(src_image.shape)
src_fig = plt.figure()
plt.title("src_image")
plt.imshow(src_image)

gray_image = cv2.cvtColor(src_image, cv2.COLOR_RGB2GRAY)
plt.figure()
plt.title("gray_image")
plt.imshow(gray_image)  # 二值化

ret, binary = cv2.threshold(gray_image, 245, 255, cv2.THRESH_BINARY)  # 对灰度图进行二值化，binary是返回图像
plt.figure()
plt.title("binary")
plt.imshow(binary)  # 进行开操作，去除噪音小点。

kernel_3X3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 构造卷积核
binary_after_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_3X3)  # 形态学
plt.figure()
plt.title("binary_after_open")
plt.imshow(binary_after_open)

restored = cv2.inpaint(src_image, binary_after_open, 9, cv2.INPAINT_NS)
restored_fig = plt.figure()
plt.title("restored")
plt.imshow(restored)
plt.show()

