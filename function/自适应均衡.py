import imageio
import matplotlib.pyplot as plt
import matplotlib

import numpy as np

from skimage import data, io , img_as_float
from skimage import exposure
def equalize_Adapthist(img_url):
    """

    :param img_url: 图片地址
    :return:
    """
    img = io.imread(img_url)
    # skimage.exposure.equalize_adapthist（图像，kernel_size =无，clip_limit = 0.01，nbins = 256)
    img_adapteq = exposure.equalize_adapthist(img,clip_limit=0.03) #对比度受限自适应直方图均衡
    plt.subplot(121)
    plt.title('yuantu')
    plt.imshow(img)

    plt.subplot(122)
    plt.title('Adaptive equalization')
    plt.imshow(img_adapteq)
    plt.show()

equalize_Adapthist('D:/aiImg/112.png')