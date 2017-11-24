import numpy as np
import ast
import scipy   
import matplotlib.pyplot as plt
import cv2
import os
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image    
from keras.models import Model   
import sys
import time

def pretrained_path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    x = np.expand_dims(x, axis=0)
    # convert RGB -> BGR, subtract mean ImageNet pixel, and return 4D tensor
    return preprocess_input(x)

def get_ResNet():
    # define ResNet50 model
    model = ResNet50(weights='imagenet')
    # get AMP layer weights
    all_amp_layer_weights = model.layers[-1].get_weights()[0]
    # extract wanted output
    ResNet_model = Model(inputs=model.input, 
        outputs=(model.layers[-4].output, model.layers[-1].output)) 
    return ResNet_model, all_amp_layer_weights
    
def ResNet_CAM(img_path, model, all_amp_layer_weights):
    # get filtered images from convolutional output + model prediction vector

    last_conv_output, pred_vec = model.predict(pretrained_path_to_tensor(img_path))
    # change dimensions of last convolutional outpu tto 7 x 7 x 2048
    last_conv_output = np.squeeze(last_conv_output)

    # get model's prediction (number between 0 and 999, inclusive)
    pred = np.argmax(pred_vec)

    # bilinear upsampling to resize each filtered image to size of original image 
    mat_for_mult = scipy.ndimage.zoom(last_conv_output, (32, 32, 1), order=0) # dim: 224 x 224 x 2048
    #mat_for_mult = cv2.resize(last_conv_output, (224, 224))

    # get AMP layer weights
    amp_layer_weights = all_amp_layer_weights[:, pred] # dim: (2048,) 
    # get class activation map for object class that is predicted to be in the image
    final_output = np.dot(mat_for_mult.reshape((224*224, 2048)), amp_layer_weights).reshape(224,224) # dim: 224 x 224
    # return class activation map

    return final_output, pred
    
def binary_ResNet_CAM(img_path, file_name, model, all_amp_layer_weights):
    t1 = time.time()
    # load image, convert BGR --> RGB, resize image to 224 x 224,
    img = cv2.imread(img_path)
    # 计算缩放比例
    w = img.shape[0]
    h = img.shape[1]
    ratioW = w/224
    ratioH = h/224
    # 获取激活图
    CAM, pred = ResNet_CAM(img_path, model, all_amp_layer_weights)

    # 阈值化 寻找最大连通域并加入bounding-box
    maxVal = np.max(CAM)
    threshVal = 0.2*maxVal
    ret, CAMbinary = cv2.threshold(CAM, threshVal, 255, cv2.THRESH_BINARY)
    CAMbinary = CAMbinary.astype(np.uint8)
    image, contours, hierarchy = cv2.findContours(CAMbinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxArea = 0
    maxCnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > maxArea:
            maxArea = area
            maxCnt = cnt
    x, y, w, h = cv2.boundingRect(maxCnt)
    # 按比例还原
    #cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
    x = int(x*ratioH)
    w = int(w*ratioH)
    y = int(y*ratioW)
    h = int(h*ratioW)
    #cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi = img[y:y+h, x:x+w]

    SAVE_SUFFIX = "../data/test_crop/"
    cv2.imwrite(SAVE_SUFFIX+file_name, roi)
    t2 = time.time()
    print(t2 - t1)


if __name__ == '__main__':
    ResNet_model, all_amp_layer_weights = get_ResNet()
    trainPath = "../data/raw_data/train/train/"
    valPath = "../data/raw_data/val/test1/"
    testPath = "../data/test/test-01/image/"
    for parent, dirnames, filenames in os.walk(testPath):
        # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        for index, filename in enumerate(filenames):
            print(index, filename)
            img_path = testPath + filename
            CAM = binary_ResNet_CAM(img_path, filename, ResNet_model, all_amp_layer_weights)

    # for parent, dirnames, filenames in os.walk(valPath):
    #     # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
    #     for index, filename in enumerate(filenames):
    #         print(index, filename)
    #         img_path = valPath + filename
    #         CAM = binary_ResNet_CAM(img_path, filename, ResNet_model, all_amp_layer_weights)
    #

    # filename = "587309241,1722340063.jpg"
    # img_path = trainPath + filename
    # a = binary_ResNet_CAM(img_path, filename, ResNet_model, all_amp_layer_weights)
