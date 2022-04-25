import matplotlib.pyplot as plt
from skimage import data, io , img_as_float
from skimage import exposure
def equalize_Hist(img_url):
    """

    :param img_url: 图片地址
    :return:
    """
    img = io.imread(img_url)
    # skimage.exposure.equalize_hist（图像，nbins = 256，掩码=无）
    img_eq = exposure.equalize_hist(img)# 返回直方图均衡后的图像
    plt.subplot(121)
    plt.title('yuantu')
    plt.imshow(img)

    plt.subplot(122)
    plt.title('Histogram equalization')
    plt.imshow(img_eq)
    plt.show()

equalize_Hist('D:/aiImg/112.png')