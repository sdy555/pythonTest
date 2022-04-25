# import cv2
# import numpy as np
#
# img = cv2.imread('D:/aiImg/huifu.png')
# def cv_show(img):
#     cv2.imshow('ss', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# kernel = np.ones((3, 3), dtype=np.uint8)
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, 1)
# ss = np.hstack((img, opening))
# cv_show(ss)
# # cv_show(img)
# # kernel = np.ones((3, 3), dtype=np.uint8)
# # closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  ## 有缺陷，填补缺陷
# # ss = np.hstack((img, closing))
# # cv_show(ss)
# # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, 1)
# # ss = np.hstack((img, opening))
# # cv_show(ss)
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 1 读取图像
img = cv.imread("D:/aiImg/img/3.png")

# 2 创建核结构
kernel = np.ones((2, 2), np.uint8)

# 3 图像腐蚀和膨胀
erosion = cv.erode(img, kernel)  # 腐蚀


# 4 图像展示
plt.imshow(erosion)

plt.show()