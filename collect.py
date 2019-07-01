import cv2
import numpy as np


global Frame
global point1, point2
global Sample_NUM


def on_mouse(event, x, y, flags, param):
    global Frame, point1, point2, Sample_NUM
    img2 = Frame.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 2)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):               #按住左键拖曳
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 2)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0,0,255), 2)
        cv2.imshow('image', img2)
        min_x = min(point1[0], point2[0])
        min_y = min(point1[1], point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] - point2[1])
        cut_img = Frame[min_y:min_y+height, min_x:min_x+width]
        Save_File = Save_Path + str(Sample_NUM) + ".jpg"
        Sample_NUM += 1
        cv2.imwrite(Save_File, cut_img)
        print(Save_File)


def cut_frame():
    global Sample_NUM, Frame, Save_Path
    Sample_NUM = 849
    Current_Frame = 1
    Video_Path = '/home/wu/下载/test4.avi'
    Save_Path = '/home/wu/桌面/PosSample/'
    Interval_Frame = 50
    cap = cv2.VideoCapture(Video_Path)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', (720, 560))
    while(1):
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_Filter = cv2.GaussianBlur(frame_gray, (3, 3), 0)
        Sharpen_Kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        frame_Sharpen = cv2.filter2D(frame_Filter, -1, Sharpen_Kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 椭圆结构
        Frame = cv2.morphologyEx(frame_Sharpen, cv2.MORPH_GRADIENT, kernel)  # 开运算
        Current_Frame += 1
        #
        # if ret == 0:
        #     break
        #
        if Current_Frame % Interval_Frame == 0:
        #     Current_Frame += 1
        #     img_path = Save_Path + str(Sample_NUM) + '.jpg'
        #     cv2.imwrite(img_path, frame)
        #     Sample_NUM += 1
        #     print(img_path)

            cv2.setMouseCallback('image', on_mouse)
            cv2.imshow('image', Frame)
            cv2.imshow('rgb', frame)
            if cv2.waitKey(0) & 0xFF == ord('z'):
                continue
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


cut_frame()
