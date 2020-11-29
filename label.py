import os
import shutil
import numpy as np
import pydicom
from PIL import Image
import cv2
import shutil

def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    if not contours:
        return mask
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out

def convert_from_dicom_to_jpg(img, low_window, high_window):
    """

    :param img: dicom图像的像素值信息
    :param low_window: dicom图像像素值的最低值
    :param high_window: dicom图像像素值的最高值
    :param save_path: 新生成的jpg图片的保存路径
    :return:
    """
    lungwin = np.array([low_window * 1., high_window * 1.])  # 将pydicom解析的像素值转换为array
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  # 将像素值归一化0-1
    newimg = (newimg * 255).astype('uint8')  # 再转换至0-255，且将编码方式由原来的unit16转换为unit8
    # print(newimg.shape)
    return newimg

def generate_gray(img_path,Fixed_RESHAPE_SIZE,mode='png'):


    if mode == 'dcm':
        ds = pydicom.dcmread(img_path)
        # ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        img = np.uint(ds.pixel_array)
        high = np.max(img)  # 找到最大的
        low = np.min(img)  # 找到最小的
        # 调用函数，开始转换
        img = convert_from_dicom_to_jpg(img, low, high)

        img = np.array(Image.fromarray(np.uint8(img)).resize((Fixed_RESHAPE_SIZE, Fixed_RESHAPE_SIZE), Image.ANTIALIAS))
    else:
        img = np.asarray(np.uint8(Image.open(img_path).convert("L").resize((Fixed_RESHAPE_SIZE, Fixed_RESHAPE_SIZE))))


    return img

def splittrainimage(Fixed_RESHAPE_SIZE, img_path):

    img=generate_gray(img_path,Fixed_RESHAPE_SIZE, mode='dcm')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # os.remove(img_path)
    img_path=os.path.join('image',os.path.split(img_path)[-1][:-3]+'jpg')
    cv2.imwrite(img_path,img)




def splittraindatasets(Fixed_RESHAPE_SIZE,img_Filelist,subfolder_trainA):

    for idx in range(len(img_Filelist)):
        img = img_Filelist[idx]
        img_path=os.path.join(subfolder_trainA,img)
        splittrainimage(Fixed_RESHAPE_SIZE,img_path)
    return

def txt_generate():
    train = open("train.txt", 'w', encoding='utf-8')
    train_val = open("trainval.txt", 'w', encoding='utf-8')
    test = open("test.txt", 'w', encoding='utf-8')
    val = open("val.txt", 'w', encoding='utf-8')

    imgs_path = os.path.join("image")

    imgs_name = os.listdir(imgs_path)

    length=len(imgs_name)
    trainval_len=0.8*length
    train_len=0.9*trainval_len
    for i in range(length):
        img_name = str(imgs_name[i]).strip().split('.')[0]
        if i <trainval_len:
            train_val.writelines(img_name+ '\n')
            if i <train_len:
                train.writelines(img_name + '\n')
            else:
                val.writelines(img_name + '\n')
        else:
            test.writelines(img_name + '\n')



if __name__=='__main__':
    hole='True'#否则只生成牙齿边缘的gt
    label='False'#否则gt为灰度图
    image='False'#否则尺寸和位置不对
    Fixed_RESHAPE_SIZE=1024
    if label=='True':
        llist= os.listdir('label')
        for li in llist:
            lifile=os.listdir(os.path.join('label',li))
            for file in lifile:
                file_path=os.path.join('label',li,file)
                img_grey = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                img_grey = cv2.resize(img_grey, (Fixed_RESHAPE_SIZE, Fixed_RESHAPE_SIZE))
                if hole:
                    mask = FillHole(img_grey)
                else:
                    th1 = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)  # 换行符号 \
                    mask = 255 - th1  
                mask[mask!=0]=1 #if voc dataset, it is better to use mask[mask!=0]==1 not 255
                file_path = os.path.join('label',os.path.split(file_path)[-1])
                cv2.imwrite(file_path, mask)

    if image == 'True':
        subfolder_trainA = os.listdir('image')
        for folder in subfolder_trainA:
            folder=os.path.join('image',folder)
            img_Filelist=os.listdir(folder)
            splittraindatasets(Fixed_RESHAPE_SIZE,img_Filelist,folder)

    txt_generate()



