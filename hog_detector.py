import cv2
import numpy as np

hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
hog.load('/home/wu/桌面/SVM.xml')
Video_Path = '/home/wu/下载/test19.avi'
cap = cv2.VideoCapture(Video_Path)
fourcc = cv2.VideoWriter_fourcc('M','J','P','G') #opencv3.0
videoWriter = cv2.VideoWriter('/home/wu/桌面/1.avi', fourcc, 30, (288, 352))
while(1):
    ret, frame = cap.read()
    # framea = cv2.imread('/home/wu/桌面/Frame/53.jpg', cv2.COLOR_BGR2GRAY)
    if ret == 0:
        break
    print('is running')
    print(frame.shape)
    frame1 = frame[60:250]
    # cv2.imshow('ss', frame1)
    frame_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame_Filter = cv2.GaussianBlur(frame_gray, (3, 3), 0)
    Sharpen_Kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    frame_Sharpen = cv2.filter2D(frame_Filter, -1, Sharpen_Kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 椭圆结构
    frame_Morph = cv2.morphologyEx(frame_Sharpen, cv2.MORPH_GRADIENT, kernel)  # 开运算
    rects, wei = hog.detectMultiScale(frame_Morph, winStride=(4, 4), padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y + 60), (x + w, y + h + 60), (0, 0, 255), 2)
    #创建一个矩形，来让我们在图片上写文字，参数依次定义了文字类型，高，宽，字体厚度等。
    font=cv.InitFont(cv.CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 3, 8)
    # 将文字框加入到图片中，(5, 20)定义了文字框左顶点在窗口中的位置，最后参数定义文字颜色
    cv.PutText(image, "Hello World", (30, 30), font, (0, 255, 0))
    cv2.imshow('b', frame)
    videoWriter.write(frame)
    cv2.waitKey(1)

videoWriter.release()
cv2.destroyAllWindows()
