from skimage import io,data,exposure
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms


def aug(number):
    """图像亮度增强"""

    reference = io.imread('D:/aiImg/588.jpg')  # 目标图像
    image = io.imread('D:/aiImg/51.png')  # 源图像
    if get_lightness(image) > number:
        print('true')
        print("图片亮度足够，不做增强,并返回到指定位置")
    else:
        # 参数1：源图像；参数2：目标图像；参数3：多通道匹配
        matched = match_histograms(image, reference, channel_axis=-1)  # 使源图像的累积直方图和目标图像一致
        # nrows：行数 ncols：列数
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                                sharex=True, sharey=True)
        for aa in (ax1, ax2, ax3):
            aa.set_axis_off()  # 轴。set_axis_off ( ) 关闭 x 轴和 y 轴。这会影响轴线、刻度、刻度标签、网格和轴标签。

            ax1.imshow(image)
            ax1.set_title('Source')
            ax2.imshow(reference)
            ax2.set_title('Reference')
            ax3.imshow(matched)
            ax3.set_title('Matched')

            plt.tight_layout()
            plt.show()
            # return matched
def get_lightness(src):
    # 计算亮度
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()

    return lightness

img = cv2.imread('D:/aiImg/588.jpg')
img1 = cv2.imread('D:/aiImg/51.png')
aug(50)
# cv2.imwrite('D:/aiImg/output/101.jpg', img3) # 输出到指定文件