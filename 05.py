# 边界处理
import cv2 as cv
import matplotlib.pyplot as plt
def cv_bianjiezhenghe(img):
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

cv_bianjiezhenghe('D:/aiImg/car.png')
# def cv_bianjiezhenghe(img):
#     """
#     :param img: 图片变量
#     :return: 整合图片
#     """
#     top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
#     # 复制
#     replicate = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv.BORDER_REPLICATE)
#     # 反射
#     reflect = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_REFLECT)
#     # 反射
#     reflect101 = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_REFLECT_101)
#     # 外包法
#     wrap = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_WRAP)
#     # 常量法
#     constant = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_CONSTANT, value=0)
#
#     plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
#     plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
#     plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
#     plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
#     plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
#     plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')
#     plt.show()