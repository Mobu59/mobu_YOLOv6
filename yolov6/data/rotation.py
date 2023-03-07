import cv2
import json
import numpy as np
import math
import random


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
        #return value
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
                          or single float values. Got {}".format(
                value
            )
        )


def rotate_bound(img, angle, borderValue=(114,114,114)):
    """旋转后依然保留图像全部细节"""
    h, w = img.shape[:2]
    cX, cY = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += nW / 2 - cX
    M[1, 2] += nH / 2 - cY
    new_img = cv2.warpAffine(img, M, (nW, nH), borderValue=borderValue)

    return new_img


def rotate(img, angle, borderValue=(114,114,114)):
    """旋转后图片细节会消失一部分"""
    h, w = img.shape[:2]
    cX, cY = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_img = cv2.warpAffine(img, M, (w, h), borderValue=borderValue)

    return new_img


def Srotate(angle, valuex, valuey, pointx, pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    sRotatex = (valuex - pointx) * math.cos(angle) + (valuey - pointy) * math.sin(angle) + pointx
    sRotatey = (valuey - pointy) * math.cos(angle) - (valuex - pointx) * math.sin(angle) + pointy
    return sRotatex, sRotatey


def ellipserotate(angle, b_H, b_W, x_c, y_c, pointx, pointy):
    y_1 = []
    x_1 = np.arange(x_c - b_W / 2, x_c + b_W / 2, 0.1)
    for x in x_1:
        y = math.sqrt(math.fabs((1 -(x - x_c) ** 2 / (b_W ** 2 / 4)) * (b_H
            ** 2) / 4)) + y_c
        y_1.append(y)
    for x in x_1:
        y = -math.sqrt(math.fabs((1 - (x - x_c) ** 2 / (b_W ** 2 / 4)) * (b_H
            ** 2) / 4)) + y_c
        y_1.append(y)
    x_1 = np.append(x_1, x_1)
    y_1 = np.array(y_1)
    x_2, y_2 = Srotate(math.radians(angle), x_1, y_1, pointx, pointy)

    #assert x_2.size != 0
    #assert y_2.size != 0
    if x_2.size == 0:
        x_2 = np.zeros((490,))
    if y_2.size == 0:
        y_2 = np.zeros((490,))

    x_min = np.min(x_2)
    y_min = np.min(y_2)
    x_max = np.max(x_2)
    y_max = np.max(y_2)

    return x_min, y_min, x_max, y_max


def getRotatedImg(img, label, degrees):
    angle = get_aug_params(degrees)
    #rotation_image = imutils.rotate_bound(img, angle)
    #fill_value = random.randint(0, 255)
    fill_value = 114
    rotation_image = rotate_bound(img, angle, borderValue=(fill_value,fill_value,fill_value))
    #rotation_image = rotate(img, angle, borderValue=(fill_value,fill_value,fill_value))
    height, width = img.shape[:2]
    height_new, width_new = rotation_image.shape[:2]
    box = label.copy()
    for i in range(box.shape[0]):
        x_c = box[i, 1] 
        y_c = box[i, 2]
        b_W = box[i, 3]
        b_H = box[i, 4]
                
        pointx = width / 2
        pointy = height / 2

        x_min, y_min, x_max, y_max = ellipserotate(-angle, 
                b_H,
                b_W,
                x_c,
                y_c,
                pointx,
                pointy
        )
        x_min = x_min + (width_new - width) / 2
        x_max = x_max + (width_new - width) / 2
        y_min = y_min + (height_new - height) / 2
        y_max = y_max + (height_new - height) / 2

        label[i, 1] = (x_min + x_max) / 2
        label[i, 2] = (y_min + y_max) / 2 
        label[i, 3] = x_max - x_min  
        label[i, 4] = y_max - y_min 

    return rotation_image, label


if __name__ == '__main__':
    angle = 65.
    img_path = '/world/data-gpu-94/ped_detection_data/biped_ry_k_data/K项目_data/无锡恒隆-K项目/16769523552060.9032638657715084.jpg'
    label = np.array([[425, 400, 494, 487], [170, 662, 239, 721]])
    img = cv2.imread(img_path)
    rotation_image, new_label = getRotatedImg(img, label, angle)
    for i in new_label:
        cv2.rectangle(rotation_image, (i[0], i[1]), (i[2], i[3]), 255)
    cv2.imwrite("/home/liyang/cfg_yolox/hahahahahahahaa.jpg", rotation_image)
