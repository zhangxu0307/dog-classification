from code3.bicnn1 import BiCNNVGG16
from code3.bincnn2 import BiCNNResnet50
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
import os

def train(modelPath):

    batchSize = 32
    imgSize = 224
    classLabel = os.listdir("../data/train/")
    classLabel = list(map(int, classLabel))
    classLabel.sort()
    classLabel = list(map(str, classLabel))
    print("class labels are:", classLabel)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        #featurewise_center=True,
        shear_range=0.2,
        zoom_range=0.2,
        # samplewise_std_normalization=False,
        # zca_whitening=False,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        # channel_shift_range=0.,
        fill_mode='nearest',
        # cval=0.,
        vertical_flip=True,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        #'../data/train/',
        '../data/train_data_crop/',
        #'../data/train_data/',
        #'../data/all-train-pic/',
        #'../data/valid_data',
        target_size=(imgSize, imgSize),
        batch_size=batchSize, shuffle=True,
        classes=classLabel, class_mode="categorical") # categorical返回one-hot的类别，binary返回单值

    print(train_generator.class_indices)

    val_generator = test_datagen.flow_from_directory(
        '../data/valid_data_crop/',
        target_size=(imgSize, imgSize),
        batch_size=batchSize,
        classes=classLabel, class_mode="categorical") # 此处标签可能没对上

    #CNN = BiCNNVGG16(imgSize=imgSize, load=False)
    CNN = BiCNNVGG16(imgSize = imgSize,load=True, loadW=True,loadModelPath=modelPath)
    #CNN = BiCNNResnet50(imgSize=imgSize, load=False)
    #CNN = BiCNNResnet50(imgSize = imgSize,load=True, loadW=True,loadModelPath=modelPath)
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

    CNN = train(modelPath="../model/bicnn_1.h5")



