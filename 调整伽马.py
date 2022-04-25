# 调整伽玛
from matplotlib import pyplot as plt
from skimage import data,io, exposure, img_as_float
def gamma(img_url):
    """"
    img_url:图片地址
    """
    img = io.imread(img_url)
    image = img_as_float(img)
    """
    对于大于 1 的 gamma，直方图将向左移动，输出图像将比输入图像更暗。
    
    对于小于 1 的 gamma，直方图将向右移动，输出图像将比输入图像更亮。
    """
    gamma_corrected = exposure.adjust_gamma(image, 2)
        #Output is darker for gamma > 1
    # io.imshow(image)
    plt.subplot(121)
    plt.imshow(img,plt.cm.gray) #原始图像

    plt.subplot(122)
    plt.imshow(gamma_corrected,plt.cm.gray) #均衡化图像
    plt.show()
    print(image.mean() > gamma_corrected.mean())

gamma('D:/aiImg/588.jpg')