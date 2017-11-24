from code.model1 import VGG16CNNModel, ResNet50Model, InceptionV3Model, XInceptionModel, BigModel
from code.model_2 import ResNet101Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
import os
import numpy as np
import cv2
import random


def random_crop(img):
    h = 224
    w = 224

    step = 180

    starth = np.random.randint(0, 224-step)
    startw = np.random.randint(0, 224-step)

    if starth+step > h:
        endh = 224
    else:
        endh = starth+step
    if startw+step > w:
        endw = 224
    else:
        endw = startw + step

    crop_img = img[starth:endh, startw:endw]
    crop_img = cv2.resize(crop_img, (224, 224))

    return crop_img

def PCA_Jittering(img):

    img = img/255.0
    #img = cv2.resize(img, (224,224))
    img_size = img.size/3.0
    img1 = img.reshape(img_size, 3)
    img1 = np.transpose(img1)
    imgcov = np.cov([img1[0], img1[1], img1[2]])
    lamda, p = np.linalg.eig(imgcov)
    p = np.transpose(p)
    a1 = random.normalvariate(0, 0.3)
    a2 = random.normalvariate(0, 0.3)
    a3 = random.normalvariate(0, 0.3)
    v = np.transpose((a1*lamda[0], a2*lamda[1], a3*lamda[2]))
    add_sum = np.dot(p, v)
    img2 = np.array([img[:, :, 0]+add_sum[0], img[:, :, 1]+add_sum[1], img[:, :, 2]+add_sum[2]])
    img2 = np.swapaxes(img2, 0, 2)
    img2 = np.swapaxes(img2, 0, 1)
    return img2

def train(modelPath):

    batchSize = 32
    imgSize = 224
    classLabel = os.listdir("../data/train_data_crop/")
    classLabel = list(map(int, classLabel))
    classLabel.sort()
    classLabel = list(map(str, classLabel))
    print("class labels are:", classLabel)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        # featurewise_center=True,
        shear_range=0.2,
        zoom_range= 0.4,
        # samplewise_std_normalization=False,
        # zca_whitening=False,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        channel_shift_range=0.0,
        fill_mode='nearest',
        # cval=0.,
        # preprocessing_function=preprocessing_img,
        #preprocessing_function=PCA_Jittering,
        vertical_flip=True,
        horizontal_flip=True
        )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        #'../data/total_train_crop/',
        #'../data/train_data/',
        #'../data/all-train-pic/',
        #'../data/train_data_crop/',
        #'../data/valid_data',
        '../data/test_with_label',
        target_size=(imgSize, imgSize),
        batch_size=batchSize, shuffle=True,
        classes=classLabel, class_mode="categorical") # categorical返回one-hot的类别，binary返回单值

    print(train_generator.class_indices)

    val_generator = test_datagen.flow_from_directory(
        '../data/valid_data_crop/',
        target_size=(imgSize, imgSize),
        batch_size=batchSize,
        classes=classLabel, class_mode="categorical") # 此处标签可能没对上

    #CNN = VGG16CNNModel(imgSize=imgSize, load=False)
    #CNN = VGG16CNNModel(imgSize=imgSize, load=True, loadW=False, loadModelPath=modelPath)
    #CNN = ResNet50Model(imgSize=imgSize, load=False)
    #CNN = ResNet50Model(imgSize=imgSize, load=True, loadModelPath=modelPath)
    #CNN = ResNet101Model(imgSize=imgSize, load=False)
    #CNN = ResNet101Model(imgSize=imgSize, load=True, loadW=True, loadModelPath=modelPath)
    #CNN = InceptionV3Model(imgSize=imgSize, load=False)
    #CNN = InceptionV3Model(imgSize=imgSize, load=True, loadModelPath=modelPath)
    #CNN = XInceptionModel(imgSize=imgSize, load=False)
    CNN = XInceptionModel(imgSize=imgSize, load=True, loadModelPath=modelPath)

    #CNN = BigModel(imgSize=imgSize, loadfc=False)
    #CNN = BigModel(imgSize=imgSize, loadfc=True, loadModelPath=modelPath)
    CNN.trainModel(generator=train_generator, validation_generator=val_generator, batchSize=batchSize)
    CNN.save(modelPath)

    return CNN



if __name__ == "__main__":

    # for i in range(10):
    #     modelPath = "../model/vgg16_"+str(i)
    #     CNN = train(modelPath=modelPath)
    #     print("----------------------")
    #CNN = None
    #predict(CNN, load=False)

    #CNN = train(modelPath="../model/inceptionv3_crop.h5") # train 0.82
    #CNN = train(modelPath="../model/resnet50_crop.h5") #train 0.8151
    CNN = train(modelPath="../model/Xinception_crop_2.h5") # 0.21
    #CNN = train(modelPath="../model/bigmodel_crop_2.h5") # train 0.83




