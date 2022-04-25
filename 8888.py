import cv2 as cv
import cv2 #导入openCV包
import numpy as np

# def contrast_brightness_demo(image, c, b):  # C 是对比度，b 是亮度
#     h, w, ch = image.shape
#     blank = np.zeros([h, w, ch], image.dtype)
#     dst = cv.addWeighted(image, c, blank, 1-c, b)   #改变像素的API
#     cv.imshow("con-bri-demo", dst)
#
# print("--------hello python------------")
#
#
#
# src=cv.imread("F:/shiyan/1.png")  #读取F:/shiyan/1.png路径下的名为1格式为.png的图片
# cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)  #给图片显示的窗口命名为input image
# cv.imshow("input image",src)  #显示图片
# contrast_brightness_demo(src, 1.2, 100)
# cv.waitKey(0)  #等待下一步指令
# cv.destroyAllWindows()  #为了能正常关闭所有的绘图窗口。

def get_lightness(src):
    # 计算亮度
    src = cv.imread(src)
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()
    print(lightness)

get_lightness('D:/aiImg/922.jpg')

get_lightness('D:/aiImg/output/666.jpg')

get_lightness('D:/aiImg/output/922.jpg')

