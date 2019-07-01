import cv2
import numpy as np
import random
from PIL import Image


def load_images(dirname):
    img_list = []
    file = open(dirname)
    img_name = file.readline()
    while img_name != '':  # 文件尾
        img_name = img_name.strip('\n')
        image = cv2.imread(img_name)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        img_list.append(image)
        img_name = file.readline()
    return img_list


# wsize: 处理图片大小，通常64*128; 输入图片尺寸>= wsize
def computeHOGs(img_lst, gradient_lst):
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    for i in range(len(img_lst)):
        isa = cv2.resize(img_lst[i], (64, 64))
        gradient_lst.append(hog.compute(isa))
        # return gradient_lst


def get_svm_detector(svm):
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    print(sv.shape)
    print(rho)
    a = np.append(sv, [[-rho]], 0)
    return a


# 主程序
# 第一步：计算HOG特征
neg_list = []
pos_list = []
gradient_lst = []
labels = []
hard_neg_list = []
pos_list = load_images(r'/home/wu/桌面/PosSample/pos.lst')
neg_list = load_images('/home/wu/桌面/NegSample/neg.lst')
computeHOGs(pos_list, gradient_lst)
[labels.append(+1) for _ in range(len(pos_list))]
computeHOGs(neg_list, gradient_lst)
[labels.append(-1) for _ in range(len(neg_list))]

print("after 1")
print(len(gradient_lst), len(labels))
# 第二步：训练SVM
svm = cv2.ml.SVM_create()
svm.setCoef0(0)
svm.setDegree(3)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10000, 1e-6)
svm.setTermCriteria(criteria)
svm.setGamma(0.1)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setNu(0.5)
svm.setP(0.2) # for EPSILON_SVR, epsilon in loss function?
svm.setC(5) # From paper, soft classifier
svm.setType(cv2.ml.SVM_ONE_CLASS) # C_SVC # EPSILON_SVR # may be also NU_SVR # do regression task
svm.train(np.array(gradient_lst), cv2.ml.ROW_SAMPLE, np.array(labels))

print(len(gradient_lst))
print('after 2')
# 第三步：加入识别错误的样本，进行第二轮训练
# 参考 http://masikkk.com/article/SVM-HOG-HardExample/
hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
# hard_neg_list.clear()
# hog.setSVMDetector(get_svm_detector(svm))
# for i in range(len(full_neg_lst)):
#     rects, wei = hog.detectMultiScale(full_neg_lst[i], winStride=(4, 4),padding=(8, 8), scale=1.05)
# for (x,y,w,h) in rects:
#     hardExample = full_neg_lst[i][y:y+h, x:x+w]
# hard_neg_list.append(cv2.resize(hardExample,(128,128)))
# computeHOGs(hard_neg_list, gradient_lst)
# [labels.append(-1) for _ in range(len(hard_neg_list))]
# svm.train(np.array(gradient_lst), cv2.ml.ROW_SAMPLE, np.array(labels))

print('after 3')
# 第四步：保存训练结果
hog.setSVMDetector(get_svm_detector(svm))
hog.save('/home/wu/桌面/SVM.xml')
