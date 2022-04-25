import cv2
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
img1 = cv2.imread('D:/AiImg/2.jpg')
img2 = cv2.imread('D:/aiImg/1.jpg')

sift = cv2.xfeatures2d.SIFT_create()
kp1,des1= sift.detectAndCompute(img1,None) #检测并计算
kp2,des2= sift.detectAndCompute(img2,None)

# bf = cv2.BFMatcher(crossCheck=True)
# # 一对一匹配
# matches = bf.match(des1,des2)
# matches = sorted(matches,key=lambda x: x.distance)

# k对最佳匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
good = []
for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)


img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=2)
cv2.imshow('img3',img3)
cv2.waitKey(0)
