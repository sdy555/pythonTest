import cv2 as cv

# vc = cv.VideoCapture("D:/aiImg/cs.mp4")
#
# if vc.isOpened():
#     open, frame = vc.read()
# else:
#     open=False
#
# while open:
#     ret,frame = vc.read()
#     if frame is None:
#         break
#     if ret == True:
#         gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
#         cv.imshow('result',gray)
#         if cv.waitKey(10) & 0xFF == 27:
#             break
# vc.release()
# cv.destroyAllWindows()


#定义一个输出图片函数
# def cv_imshow(name,img):
#     """
#     :param name:输出图片函数
#     :param img: （name,img）（指定输出图框名称，输出图片变量）
#     :return: 返回图片
#     """
#     cv.imshow(name,img)
#     cv.waitKey(0)
#     # cv.destroyALLWindows(0)
def cv_vcshow(dizhi):
    """
    输入：'视频绝对地址'
    :return: 输出视频
    """
    vc = cv.VideoCapture(dizhi)
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



cv_vcshow('D:/aiImg/cs.mp4')