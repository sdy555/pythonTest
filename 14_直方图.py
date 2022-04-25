import cv2 as cv
import Common as s
# 直方图
# cv2.calcHist(images,channels,mask,histSize,ranges)
# • images: 原图像图像格式为uint8 或 float32。当传入函数时应用中括号口括来例如[img]
# • channels:同样用中括号括来它会告函数我们统幅图像的直方图。如果入图像是灰度图它的值就是[0]如果是彩色图像的传入的参数可以是[0][1][2]它们分别对应着BGR。
# • mask:掩模图像。统整幅图像的直方图就把它为None。但是如 果你想统图像某一分的直方图的你就制作一个掩模图像并使用它。
# .histSize:BIN 的数目。也应用中括号括来
# • ranges:像素值范围常为[0256]
import matplotlib.pyplot as plt

# img = cv.imread('D:/aiImg/cat.jpg',0)# 0表示灰度图
# hist = cv.calcHist([img],[0],None,[256],[0,256])
# plt.hist(img.ravel(),256)
# plt.show()
import numpy as np


# color = ('b','g','r')
# for i, col in enumerate(color):
#     histr = cv.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
#     # plt.show()


img = cv.imread('D:/aiImg/58.jpg',0)
# 均衡化原理
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400]=255

masked_img=cv.bitwise_and(img,img,mask=mask) # 与操作
hist_full = cv.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv.calcHist([img],[0],mask,[256],[0,256])
plt.subplot(221),plt.imshow(img, 'gray')
plt.subplot(222),plt.imshow(mask, 'gray')
plt.subplot(223),plt.imshow(masked_img, 'gray')
plt.subplot(224),plt.plot(hist_full),plt.plot(hist_mask)
plt.xlim([0,256])
plt.show()

# cv.imshow('name', masked_img)
# cv.waitKey(0)
# cv.destroyAllWindows()









