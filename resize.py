import numpy as np
import cv2
import os
from PIL import Image
import random


Pic_Path = '/home/wu/桌面/PosOrigin/'
Path = os.path.join(Pic_Path)
Pic_List = os.listdir('/home/wu/桌面/PosOrigin/')
Sample_Num = 1
Sample_path = '/home/wu/桌面/PosSample/'

for i in Pic_List:
    Pic_Array = Image.open(os.path.join(Pic_Path, i))
    New_Pic = Pic_Array.resize((64, 64))
    Save_Path = Sample_path + "sample" + '{}.jpg'.format(Sample_Num)
    Sample_Num += 1
    New_Pic.save(Save_Path)
    for j in range(0, 1):
        Rand_Int = random.randint(0, 360)
        Pic_Rotate = New_Pic.rotate(Rand_Int)
        Save_Path = Sample_path + "sample" + '{}.jpg'.format(Sample_Num)
        Sample_Num += 1
        Pic_Rotate.save(Save_Path)

# for i in Pic_List:
#     Pic_Array = Image.open(os.path.join('/home/wu/桌面/NegOrigin/', i))
#     Pic_Array.resize((640, 640))
#     for j in range(0, 100):
#         Randw = random.randint(0, 592)
#         Randh = random.randint(0, 592)
#         Box = (Randw, Randh, Randw + 48, Randh + 48)
#         New_Pic = Pic_Array.crop(Box)
#         Save_Path = Pic_Path + '{}.jpg'.format(Sample_Num)
#         Sample_Num += 1
#         New_Pic.save(Save_Path)
#
