import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage import io,data, img_as_float
from skimage import exposure
img = io.imread('D:/aiImg/588.jpg')
# Logarithmic 对数
logarithmic_corrected = exposure.adjust_log(img, 1)
# Gamma
gamma_corrected = exposure.adjust_gamma(img, 2)
# plt.subplot(121)
# plt.imshow(logarithmic_corrected,plt.cm.gray)
#
# plt.subplot(122)
# plt.imshow(gamma_corrected,plt.cm.gray)
# plt.show()
cv2.imshow('Logarithmic',logarithmic_corrected)
cv2.imshow('Gamma',gamma_corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()


# import cv2
# import skimage
# img = cv2.imread('D:/aiImg/588.jpg')


#直方图
# img_histeq = skimage.exposure.equalize_adapthist (img,20)
# skimage.io.imshow(img_histeq)
# skimage.io.show()

# gamma
# img_gamma = skimage.exposure.adjust_gamma(img, gamma=0.5, gain=1)
# skimage.io.imshow(img_gamma)
# skimage.io.show()

# # 彩色图像直方图均衡化
# img = cv2.imread('D:/aiImg/16.jpg', 1)
# cv2.imshow("src", img)
#
# # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
# (b, g, r) = cv2.split(img)
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# # 合并每一个通道
# result = cv2.merge((bH, gH, rH))
# cv2.imshow("dst_rgb", result)
#
# cv2.waitKey(0)


import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Fig0338.tif')  # 测试图片
H = img.shape[0]
W = img.shape[1]

pixa = np.zeros((H, W), np.int32)
mImgae = np.zeros((H, W, 3), np.uint8)  # 标定(scale)前的滤波图像
smImga = np.zeros((H, W, 3), np.uint8)  # 标定(scale)后的滤波图像
pixb = np.zeros((H, W), np.int32)
mImgbe = np.zeros((H, W, 3), np.uint8)  # 标定前的滤波图像
smImgb = np.zeros((H, W, 3), np.uint8)  # 标定后的滤波图像
imga = np.zeros((H, W, 3), np.uint8)  # xy方向模板滤波后图像
imgb = np.zeros((H, W, 3), np.uint8)  # 加上对角方向模板滤波后图像

# a用到的算子是        b用到的算子是
# 0  1  0            1  1  1
# 1 -4  1            1 -8  1
# 0  1  0            1  1  1
# 先绘制标定滤波图像
# 标定指的是最小值设置为0，最大值设置为255的进行归一化的结果
for i in range(1, H - 1):
    for j in range(1, W - 1):
        pixa[i, j] = int(img[i - 1, j, 0]) + img[i + 1, j, 0] + img[i, j - 1, 0] + img[i, j + 1, 0] - 4 * int(
            img[i, j, 0])
        pixb[i, j] = int(img[i - 1, j - 1, 0]) + img[i - 1, j, 0] + img[i - 1, j + 1, 0] + img[i, j - 1, 0] + img[
            i, j + 1, 0] + img[i + 1, j - 1, 0] + img[i + 1, j, 0] + img[i + 1, j + 1, 0] - 8 * int(img[i, j, 0])

maxa = 0
maxb = 0
mina = 255
minb = 255

for i in range(H):
    for j in range(W):
        # 求出像素最大值和最小值，以利于scale
        if pixa[i, j] > maxa:
            maxa = pixa[i, j]
        if pixa[i, j] < mina:
            mina = pixa[i, j]
        if pixb[i, j] > maxb:
            maxb = pixb[i, j]
        if pixb[i, j] < minb:
            minb = pixb[i, j]
        if pixa[i, j] < 0:
            mImgae[i, j] = [0, 0, 0]
        else:
            mImgae[i, j, 0] = pixa[i, j]
            mImgae[i, j, 1] = pixa[i, j]
            mImgae[i, j, 2] = pixa[i, j]
        if pixb[i, j] < 0:
            mImgbe[i, j] = [0, 0, 0]
        else:
            mImgbe[i, j, 0] = pixb[i, j]
            mImgbe[i, j, 1] = pixb[i, j]
            mImgbe[i, j, 2] = pixb[i, j]

ka = 0
kb = 0
if maxa > mina:
    ka = 255 / (maxa - mina)
if maxb > minb:
    kb = 255 / (maxb - minb)

# scale处理
for i in range(H):
    for j in range(W):
        smImga[i, j, 0] = (pixa[i, j] - mina) * ka
        smImga[i, j, 1] = smImga[i, j, 0]
        smImga[i, j, 2] = smImga[i, j, 0]
        smImgb[i, j, 0] = (pixb[i, j] - minb) * kb
        smImgb[i, j, 1] = smImgb[i, j, 0]
        smImgb[i, j, 2] = smImgb[i, j, 0]

# 加上拉普拉斯算子
# pixa和pixb里面就是两个算子的结果
# lapa和lapb是原图加算子的结果，用来裁剪或者scale的原始数据
lapa = np.zeros((H, W), np.int32)
lapb = np.zeros((H, W), np.int32)

# 缩放处理
# maxa = 0
# maxb = 0
# mina = 255
# minb = 255

for i in range(H):
    for j in range(W):
        lapa[i, j] = img[i, j, 0] - pixa[i, j]
        lapb[i, j] = img[i, j, 0] - pixb[i, j]
        # 裁剪处理
        if lapa[i, j] > 255:
            lapa[i, j] = 255
        if lapa[i, j] < 0:
            lapa[i, j] = 0
        if lapb[i, j] > 255:
            lapb[i, j] = 255
        if lapb[i, j] < 0:
            lapb[i, j] = 0
        # 缩放处理
        # if lapa[i, j] > maxa:
        #     maxa = lapa[i, j]
        # if lapa[i, j] < mina:
        #     mina = lapa[i, j]
        # if lapb[i, j] > maxb:
        #     maxb = lapb[i, j]
        # if lapb[i, j] < minb:
        #     minb = lapb[i, j]

# 缩放处理
# ka = 0
# kb = 0
# if maxa > mina:
#     ka = 255 / maxa
# if maxb > minb:
#     kb = 255 / maxb

# scale处理
for i in range(H):
    for j in range(W):
        # 裁剪处理
        imga[i, j, 0] = lapa[i, j]
        imga[i, j, 1] = lapa[i, j]
        imga[i, j, 2] = lapa[i, j]
        imgb[i, j, 0] = lapb[i, j]
        imgb[i, j, 1] = lapb[i, j]
        imgb[i, j, 2] = lapb[i, j]
        # 缩放处理
        # if lapa[i, j] > 0:
        #     imga[i, j, 0] = lapa[i, j] * ka
        # else:
        #     imga[i, j, 0] = 0
        # imga[i, j, 1] = imga[i, j, 0]
        # imga[i, j, 2] = imga[i, j, 0]
        # if lapb[i, j] > 0:
        #     imgb[i, j, 0] = lapb[i, j] * kb
        # else:
        #     imgb[i, j, 0] = 0
        # imgb[i, j, 1] = imgb[i, j, 0]
        # imgb[i, j, 2] = imgb[i, j, 0]

# 原图
plt.subplot(1, 4, 1)
plt.axis('off')
plt.title('Original image')
plt.imshow(img)

# 图3.37a的模板
plt.subplot(2, 4, 2)
plt.axis('off')
plt.title('Before sale a')
plt.imshow(mImgae)

# scale后图3.37a的模板
plt.subplot(2, 4, 3)
plt.axis('off')
plt.title('After sale a')
plt.imshow(smImga)

# 图3.37a的模板锐化后的图像
plt.subplot(2, 4, 4)
plt.axis('off')
plt.title('Sharpened Image a')
plt.imshow(imga)

# 图3.37b的模板
plt.subplot(2, 4, 6)
plt.axis('off')
plt.title('Before sale b')
plt.imshow(mImgbe)

# scale后图3.37b的模板
plt.subplot(2, 4, 7)
plt.axis('off')
plt.title('After sale b')
plt.imshow(smImgb)

# 图3.37b的模板锐化后的图像
plt.subplot(2, 4, 8)
plt.axis('off')
plt.title('Sharpened Image b')
plt.imshow(imgb)

plt.show()

