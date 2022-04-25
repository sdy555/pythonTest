import cv2 as cv

img = cv.imread("D:/aiImg/cat.jpg")

def cv_show(name,img):
    cv.imshow(name,img)
    cv.waitKey(0)
    cv.destroyAllWindows()


cat = img[0:150,0:200]

# cv_show('cat',cat)
#颜色通道提取
b,g,r = cv.split(img)
# print(r.shape)

# img = cv.merge((b,g,r))
# print(img.shape)

# #只保留R
# cur_img = img.copy()
# cur_img[:,:,0]=0
# cur_img[:,:,1]=0
# cv_show('R',cur_img)

# #只保留G
# cur_img = img.copy()
# cur_img[:,:,0]=0
# cur_img[:,:,2]=0
# cv_show('G',cur_img)

#只保留B
cur_img = img.copy()
cur_img[:,:,1]=0
cur_img[:,:,2]=0
cv_show('B',cur_img)


