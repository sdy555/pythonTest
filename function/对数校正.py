# 对数校正
from matplotlib import pyplot as plt
from skimage import data, io, exposure, img_as_float


def log(img_url):
    """"
    img_url:图片地址
    """
    img = io.imread(img_url)
    image = img_as_float(img)
    """
对输入图像执行对数校正。

此函数在将每个像素缩放到 0 到 1 的范围后 ，根据方程对输入图像进行逐像素变换。对于逆对数校正，方程
    """
    gamma_corrected = exposure.adjust_log(image)
    # Output is darker for gamma > 1
    # io.imshow(image)
    plt.subplot(121)
    plt.imshow(img, plt.cm.gray)  # 原始图像

    plt.subplot(122)
    plt.imshow(gamma_corrected, plt.cm.gray)  # Sigmoid 校正图像
    plt.show()



log('D:/aiImg/588.jpg')