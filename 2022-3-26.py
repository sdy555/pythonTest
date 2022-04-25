import cv2
import skimage
# img = cv2.imread('D:/aiImg/yibiao.jpg')


# 直方图
# img_histeq = skimage.exposure.equalize_adapthist (img,20)
# skimage.io.imshow(img_histeq)
# skimage.io.show()
#
# # gamma
# img_gamma = skimage.exposure.adjust_gamma(img, gamma=0.5, gain=1)
# skimage.io.imshow(img_gamma)
# skimage.io.show()

# 彩色图像直方图均衡化
# img = cv2.imread('D:/aiImg/yibiao.jpg', 1)
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


import matplotlib.pyplot as plt

from skimage import data
from skimage import io
from skimage import exposure
from skimage.exposure import match_histograms


reference = io.imread('D:/aiImg/yibiao.jpg')# 目标图像
image = io.imread('D:/aiImg/922.jpg')# 源图像

    # 参数1：源图像；参数2：目标图像；参数3：多通道匹配
    # skimage.exposure.match_histograms（图像、参考、*、通道轴=无、多通道=假）
matched = match_histograms(image, reference, channel_axis=-1)# 调整图像，使其累积直方图与另一张相匹配
    # nrows：行数 ncols：列数
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
#                                         sharex=True, sharey=True)
# for aa in (ax1, ax2, ax3):
#     aa.set_axis_off() #轴。set_axis_off ( ) 关闭 x 轴和 y 轴。这会影响轴线、刻度、刻度标签、网格和轴标签。


# 显示图像
plt.subplot(131)
plt.imshow(image)
plt.title('Source')
plt.subplot(132)
plt.imshow(reference)
plt.title('Reference')
plt.subplot(133)
plt.imshow(matched)
plt.title('Matched')


plt.tight_layout()
plt.show()

# fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
#
#
# for i, img in enumerate((image, reference, matched)):
#     for c, c_color in enumerate(('red', 'green', 'blue')):
#         img_hist, bins = exposure.histogram(img[..., c], source_range='dtype')
#         axes[c, i].plot(bins, img_hist / img_hist.max())
#         img_cdf, bins = exposure.cumulative_distribution(img[..., c])
#         axes[c, i].plot(bins, img_cdf)
#         axes[c, 0].set_ylabel(c_color)
#
# axes[0, 0].set_title('Source')
# axes[0, 1].set_title('Reference')
# axes[0, 2].set_title('Matched')
#
# plt.tight_layout()
# plt.show()

