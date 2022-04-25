import cv2
from matplotlib import pyplot as plt
from skimage import io

import numpy as np
from skimage.exposure import is_low_contrast, match_histograms

"""
is_low_contrast（图像， fraction_threshold = 0.05， lower_percentile = 1， upper_percentile = 99， method = 'linear'）
当图像被确定为低对比度时为真
"""
def is_low_Contrast(img_url,img_url2):
    image = io.imread(img_url)# 源图像
    image1 = io.imread(img_url2)# 参照目标图像
    # # image = np.linspace(0, 0.04, 100)
    # image[-1] = 1
    if get_lightness(image) > 500:
        print('true')
        print("图片亮度足够，不做增强,并返回到指定位置")
        cv2.imwrite('D:/aiImg/output/5555.jpg', image)  # 均衡后图片的位置
    else:
        print('flase')
        print("图片亮度不够，做均衡处理,并返回到指定位置")
        reference = image1 # 目标图像
        image = image # 源图像

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

        cv2.imwrite('D:/aiImg/output/5555.jpg',matched) # 均衡后图片的位置

def get_lightness(src):
    # 计算亮度
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()

    return lightness

is_low_Contrast('D:/aiImg/588.jpg','D:/aiImg/cat.jpg')