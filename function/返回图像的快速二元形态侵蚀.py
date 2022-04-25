import cv2
import matplotlib.pyplot as plt
import skimage
#
from skimage.color import rgb2gray
from skimage.morphology import disk

img = rgb2gray(cv2.imread('D:/aiImg/zhang.png'))
img[img <= 0.5] = 0
img[img > 0.5] = 1
plt.gray()
plt.subplot(231)
plt.imshow(img)
plt.axis('off')
# # 生成平坦的圆盘形足迹
# kernel = skimage.morphology.disk(5)
#返回图像的快速二元形态膨胀。
img_dialtion = skimage.morphology.dilation(img, disk(12))
plt.subplot(232)
plt.imshow(img_dialtion)
plt.title('dilation')
plt.axis('off')
# 返回图像的灰度形态侵蚀。
img_erosion = skimage.morphology.erosion(img, disk(6))
plt.subplot(233)
plt.imshow(img_erosion)
plt.title('erosion')
plt.axis('off')
# 返回图像的灰度形态开度
img_open =skimage.morphology.opening(img, disk(6))
plt.subplot(234)
plt.imshow(img_open)
plt.title('opening')
plt.axis('off')
# 返回图像的灰度形态闭合。
img_close =skimage.morphology.closing(img, disk(6))
plt.subplot(235)
plt.imshow(img_close)
plt.title('closing')
plt.axis('off')
#返回图像的白色礼帽
img_white =skimage.morphology.white_tophat(img, disk(6))
plt.subplot(236)
plt.imshow(img_white)
plt.title('white_tophat')
plt.axis('off')
plt.show()
#
# import skimage
# img = skimage.data.binary_blobs(100)
# skimage.io.imshow(img)
# skimage.io.show()
#
# kernel = skimage.morphology.disk(5)
# img_dialtion = skimage.morphology.dilation(img, kernel)
#
# skimage.io.imshow(img_dialtion)
# skimage.io.show()
#
# img_erosion = skimage.morphology.erosion(img, kernel)
# skimage.io.imshow(img_erosion)
# skimage.io.show()
#
# img_open =skimage.morphology.opening(img, kernel)
# skimage.io.imshow(img_open)
# skimage.io.show()
#
# img_close =skimage.morphology. closing(img, kernel)
# skimage.io.imshow(img_close)
# skimage.io.show()
#
# img_white =skimage.morphology.white_tophat(img, kernel)
# skimage.io.imshow(img_white)
# skimage.io.show()


# from skimage import io
# from skimage.morphology import binary_opening, binary_closing, disk, binary_erosion, binary_dilation
# from skimage.util import invert
# from skimage.color import rgb2gray
# from skimage.io import imread
# import numpy as np
# import matplotlib.pyplot as plt
#
# im = rgb2gray(imread('D:/aiImg/dog.jpg'))
# print(np.max(im))
# im[im <= 0.5] = 0
# im[im > 0.5] = 1
# plt.gray()
# plt.figure(figsize=(20,10))
# plt.subplot(231)
# plt.imshow(im)
# plt.title('original', size=20)
# plt.axis('off')
# plt.subplot(2,3,2)
# im1 = binary_opening(im, disk(12))
# plt.imshow(im1)
# plt.title('opening with disk size ' + str(12), size=20)
# plt.axis('off')
# plt.subplot(2,3,3)
# # im1 = invert(binary_closing(invert(im), disk(6)))
# im1 = binary_closing(im, disk(6))
# plt.imshow(im1)
# plt.title('closing with disk size ' + str(6), size=20)
# plt.axis('off')
# plt.subplot(2,3,5)
# im1 = binary_erosion(im, disk(12))
# plt.imshow(im1)
# plt.title('erosion with disk size ' + str(12), size=20)
# plt.axis('off')
# plt.subplot(2,3,6)
# im1 = binary_dilation(im, disk(6))
# plt.imshow(im1)
# plt.title('dilation with disk size ' + str(6), size=20)
# plt.axis('off')
# plt.show()
