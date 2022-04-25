import cv2
import numpy as np

img = cv2.imread("D:/aiImg/23.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create() # 根据表述我们可以使用estimateAffine2D和estimateAffinePartial2D两个方法代替使用，但是到底应该选择哪一个方法进行替代，还需要看estimateRigidTransform方法的第三个参数fullAffine的取值。
kp = sift.detect(gray,None)
img = cv2.drawKeypoints(gray,kp,img)
# cv2.imshow('drawKeypoints',img)
# cv2.waitKey(0)

kp, des = sift.compute(gray,kp) # 得到关键点
print(np.array(kp).shape)
print(des.shape)

