import matplotlib.pyplot as plt
import numpy as np

import cv2 as cv

# """"
# 傅里叶变换的作用
# .高频:变化剧烈的灰度分量,例如边界
# .低频:变化缓慢的灰度分量,例如一片大海
# 滤波
# .低通滤波器:只保留低频,会使得图像模糊
# .高通滤波器:只保留高频,会使得图像细节增强
#
# .opencv中主要就是cv2.dft()和cv2.idft(),输入图像需要先转换成np.float32格式。
# .得到的结果中频率为0的部分会在左上角,通常要转换到中心位置,可以通过shift变换来实现。
# .cv2.dft()返回的结果是双通道的(实部,虚部),通常还需要转换成图像格式才能展示(0,255)
# """

img = cv.imread('D:/aiImg/lena.jpg',0)

img_float32 = np.float32(img)

dft = cv.dft(img_float32,flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
# 得到灰度图能表示的形式
magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.subplot(121),plt.imshow(img,cmap='gray')
plt.title('Input Image'),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum,cmap='gray')
plt.title('Magnitude Spectrum'),plt.xticks([]),plt.yticks([])
plt.show()



