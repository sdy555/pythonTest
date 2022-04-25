import cv2 as cv
import cv2
import plistlib
import matplotlib.pyplot as plt
import numpy as np


# 边界处理
def cv_bianjiezhenghe(img):
    """
        img:图片地址
    """
    img = cv.imread(img)
    top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
    replicate = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv.BORDER_REPLICATE)
    reflect = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_REFLECT)
    reflect101 = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_REFLECT_101)
    wrap = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_WRAP)
    constant = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_CONSTANT, value=0)

    plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
    plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
    plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
    plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
    plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
    plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')
    plt.show()


# 输出图片
def cv_show(img):

    cv.imshow('res', img)
    cv.waitKey(0)
    cv.destroyAllWindows()





# 输出视屏
def cv_vcshow(mp4Url):
    """
    mp4Url：'视频绝对地址'
    :return: 输出视频
    """
    vc = cv.VideoCapture(mp4Url)
    # 检查是否能打开
    if vc.isOpened():
        oepn, frame = vc.read()
    else:
        oepn = False

    while open:
        ret, frame = vc.read()
        if frame is None:
            break
        if ret == True:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            cv.imshow('result', gray)
            if cv.waitKey(10) & 0xFF == 27:
                break
    vc.release()


def yuzhi(url):
    """
    :param url:图片绝对地址
    :return: 返回阈值处理图片
    """
    img = cv.imread(url)
    # 设置；两个变量。第二个变量为阈值，调用threshold。输入src，基本阈值，最大阈值，type类型
    ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    # THRESH_BINARY 超过阈值部分去maxval，否则取0
    ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    # THRESH_BINARY_INV THRESH_BINARY的反转
    ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
    # THRESH_TRUNC 大于阈值部分设为阈值，否者不变
    ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
    # 大于阈值部分不改变
    ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
    # 取反
    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


# 轮廓检测
def cv_lunkuojianche(img):
    """
    :param img:图片地址
    """
    img = cv.imread(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    draw_img = img.copy()
    res = cv.drawContours(draw_img, contours, -1, (0, 0, 255), 2)

    cv.imshow('res', res)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 方框滤波
def fangkuanglvbo(img):
    """
    :param img:图片变量
    :return: 返回平滑处理后的图片
    """
    # 方框滤波
    # 基本和均值一样，可以选择归一化
    box = cv.boxFilter(img, -1, (3, 3), normalize=True)
    cv.imshow('box', box)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 方框滤波2
def fangkuanglvbo2(img):
    """
    :param img:图片变量
    :return: 返回平滑处理后的图片
    """
    # 方框滤波
    # 基本和均值一样，可以选择归一化，容易越界,越界全区255，变白
    box = cv.boxFilter(img, -1, (3, 3), normalize=True)
    cv.imshow('box', box)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 均值滤波
def junzhilvbo(img):
    """
    :param img:图片变量
    :return: 返回均值滤波处理的图片
        """
    # 均值滤波
    # 简单的平均卷积操作
    blur = cv.blur(img, (4, 4))
    cv.imshow('blur', blur)
    cv.waitKey(0)
    cv.destroyAllWindows()

# 平滑处理
def pinghuachuli(img_url):
    """
    :param img_url:图片地址&变量
    :return:返回所有平滑处理后的图片
    """
    img_url = cv.imread(img_url)
    edian = cv.medianBlur(img_url, 5)  # 中值滤波
    aussian = cv.GaussianBlur(img_url, (5, 5), 1)  # 高斯滤波
    box = cv.boxFilter(img_url, -1, (3, 3), normalize=True)  # 方框滤波
    box1 = cv.boxFilter(img_url, -1, (3, 3), normalize=False)  # 方框滤波
    blur = cv.blur(img_url, (4, 4))  # 均值滤波

    res = np.hstack((edian, aussian, box, box1, blur))
    print(res)
    cv.imshow('median vs averager', res)
    cv.waitKey(0)
    cv.destroyAllWindows()

# 轮廓近似
def cv_lunkuojinsi(Img_url):
    """
    Img_url:图片地址
    """
    img = cv.imread(Img_url)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnt = contours[0]
    epsilon = 0.1 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)
    draw_img = img.copy()
    res = cv.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)
    cv.imshow('res',res)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 边缘检测 Canny
def Canny(img_url):
    img = cv.imread(img_url,cv.IMREAD_GRAYSCALE)
    # Canny边缘检测 使用高斯滤波器，以平滑图像，滤波噪声
    # 计算图像中每个像素点的梯度强度和方向
    # 应用非极大值（Non Maximum Suppression)抑制，以消除边缘检测带来的杂散响应
    # 应用双阈值（Double Threshold）边缘来确定真实的和潜在的边缘
    # 通过抑制鼓励的弱边缘最终完成边缘检测
    Canny1 = cv.Canny(img,0,0)
    Canny2 = cv.Canny(img,150,80)
    res = np.hstack((Canny1,Canny2))
    cv.imshow('res',res)
    cv.waitKey(0)

#膨胀操作
def pengzhangcaozuo(img_url):
    """

    :param img_url:图片绝对地址
    :return:返回输出值
    """
    img =  cv.imread(img_url)
    # cv_imshow('1',img)

    kernel = np.ones((10,10),np.uint8) #调整膨胀的大小（10，10）值越大就越膨胀
    dilate3 = cv.dilate(img,kernel,iterations=3)
    dilate1 = cv.dilate(img,kernel,iterations=1)
    dilate2 = cv.dilate(img,kernel,iterations=2)
    res = np.hstack((dilate2,dilate1,dilate3))
    cv.imshow ('res',res)
    cv.waitKey(0)
    cv.destroyAllWindows()

# 边界矩形 boundingRect  rectangle
def cv_bianjuejuxing(img_url):
    """
    img_url:图片地址
    """
    img = cv.imread(img_url)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnt = contours[2]  # 0是变量
    x, y, w, h = cv.boundingRect(cnt)
    img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 开运算：先腐蚀，再膨胀
def opening(img_url):
    """

    :param img_url:
    :return:
    """
    img = cv.imread(img_url)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    cv.imshow('opening', opening)
    cv.waitKey(0)
    cv.destroyAllWindows()

# 闭运算：先膨胀，再腐蚀
def closeing(img_url):
    """

    :param img_url: 图片绝对地址
    :return: 返回闭运算
    """
    img = cv.imread(img_url)
    kernel = np.ones((5,5),np.uint8)
    closing = cv.morphologyEx(img,cv.MORPH_CLOSE,kernel)

    cv.imshow('closing',closing)
    cv.waitKey(0)
    cv.destroyAllWindows()

# 模板匹配多个对象 rectangle
def cv_mobanpipei(img_url,img_url2):
    """
    img_url:原图图片地址
    img_url2：需要匹配的图片地址
    """
    img_rgb = cv.imread(img_url)
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    template = cv.imread(img_url2, 0)
    h, w = template.shape[:2]
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    threshold = 0.8
    # 取匹配程度大于%80的坐标
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]): # *号表示可选参数
        bottom_right = (pt[0] + w, pt[1] + h)
        cv.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)

    cv.imshow('img_rab', img_rgb)
    cv.waitKey(0)

# 直方图均衡化
def cv_junhenghua(img_url):
    """"
    img_url：图片地址
    equalizeHist 均衡化
    createCLAHE 自适应均衡化
    img:原图
    equ：均衡化后的图片
    res_clahe：自适应均衡化的图片
    """
    img = cv.imread(img_url, 0)
    plt.hist(img.ravel(), 256)
    equ = cv.equalizeHist(img)  # equalizeHist均衡化函数
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # createCLAHE 自适应均衡化
    res_clahe = clahe.apply(img)
    res = np.hstack((img, equ, res_clahe))
    cv.imshow('res',res)
    cv.waitKey(0)

def sobelx_xy(img_url):
    """

    :param img_url:图片地址
    :return:返回处理图片
    """
    img = cv.imread(img_url,cv.IMREAD_GRAYSCALE)
    # 直接计算sobelx因子
    sobelxy = cv.Sobel(img,cv.CV_64F,1,2,ksize=3)
    sobelxy = cv.convertScaleAbs(sobelxy)
    # 调用cv_imshow函数展示图片
    cv.imshow('res',sobelxy)
    cv.waitKey(0)

# 高通滤波
def gaotonglvbo(img_url):
    """"
    img_url：图片地址

    """
    img = cv2.imread(img_url, 0)
    img_float32 = np.float32(img)

    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置

    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    # IDFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_back, cmap='gray')
    plt.title('Result'), plt.xticks([]), plt.yticks([])

    plt.show()


# 低通滤波
def ditonglvbo(img_url):
    """"
    img_url：图片地址

    """
    img = cv2.imread(img_url, 0)
    img_float32 = np.float32(img)

    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置

    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    # IDFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_back, cmap='gray')
    plt.title('Result'), plt.xticks([]), plt.yticks([])

    plt.show()


# 边角检测
def jiaodianjianche(img_url):
    """"
    img_url:图片地址
    """
    img = cv.imread(img_url)
    print('img.shape', img.shape)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    print('dst.shape', dst.shape)

    img[dst > 0.01 * dst.max()] = [0, 0, 225]
    cv.imshow('dst', img)
    cv.waitKey(0)



# cv_mobanpipei('D:/aiImg/mario.jpg','D:/aiImg/mario_coin.jpg')
# cv_bianjuejuxing('D:/aiImg/contours.png')
# Canny('D:/aiImg/cat.jpg')

# cv_show('D:/aiImg/cat.jpg')




