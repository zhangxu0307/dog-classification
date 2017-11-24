from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input, Flatten, Dense, Activation, Dropout
from keras.models import Model
from keras.models import load_model
from keras import regularizers
import numpy as np
import os
from skimage import io
import PIL
from PIL import Image
from keras import optimizers
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.applications.resnet50 import ResNet50
import keras.backend as K
from code3.extra_layer import MyLayer

class BaseModel():

    def __init__(self, imgSize, load=False, loadW = False, loadModelPath=None): # 选择是否加载已有模型文件或构建模型训练

        self.imgSize = imgSize
        lr = 1e-5
        if load:
            if loadW: # 加载权重
                self.model = self.buildModel()
                self.model.load_weights(loadModelPath)
                self.model.compile(loss='categorical_crossentropy',
                                   optimizer=optimizers.SGD(lr=lr,
                                                            momentum=0.9,
                                                            decay=0.0001,
                                                            nesterov=True,
                                                            ),
                                   # optimizer=optimizers.Adam(lr=1e-3),
                                   metrics=['categorical_accuracy'])
            else: # 加载模型和权重
                self.model = load_model(loadModelPath)
        else:
            self.model = self.buildModel()
            #self.model = make_parallel(self.model, 2)
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=optimizers.SGD(lr=lr,
                                                        momentum=0.9,
                                                        decay=0.0001,
                                                        nesterov=True,
                                                        ),
                               #optimizer=optimizers.Adam(lr=1e-3),
                               metrics=['categorical_accuracy'])

        self.model.summary()


    def buildModel(self): # 子类实现具体模型结构
        pass

    def trainModel(self, generator, validation_generator, batchSize): # 训练模型

        pass

    def inference(self, submissionFile): # 预测

        ansList = []
        path = "../data/test/test-01/image/"
        #submit = pd.read_csv("../data/submission.csv")
        submit = open(submissionFile, "w+")

        class_indices = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12,
         '13': 13, '14': 14, '16': 15, '17': 16, '18': 17, '19': 18, '20': 19, '21': 20, '22': 21, '23': 22, '24': 23,
         '25': 24, '26': 25, '27': 26, '28': 27, '29': 28, '30': 29, '31': 30, '32': 31, '33': 32, '34': 33, '35': 34,
         '36': 35, '37': 36, '38': 37, '39': 38, '40': 39, '41': 40, '42': 41, '43': 42, '45': 43, '46': 44, '47': 45,
         '48': 46, '49': 47, '50': 48, '51': 49, '52': 50, '53': 51, '54': 52, '57': 53, '59': 54, '60': 55, '61': 56,
         '62': 57, '63': 58, '64': 59, '65': 60, '66': 61, '67': 62, '68': 63, '69': 64, '70': 65, '71': 66, '72': 67,
         '73': 68, '74': 69, '75': 70, '76': 71, '77': 72, '78': 73, '79': 74, '80': 75, '81': 76, '82': 77, '83': 78,
         '84': 79, '85': 80, '86': 81, '87': 82, '88': 83, '94': 84, '95': 85, '97': 86, '101': 87, '109': 88,
         '111': 89, '114': 90, '115': 91, '120': 92, '123': 93, '126': 94, '127': 95, '128': 96, '129': 97, '132': 98,
         '133': 99}

        class_indices = {v: k for k, v in class_indices.items()}
        print("class labels are:", class_indices)

        for parent, dirnames, filenames in os.walk(path):
            # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
            for index, filename in enumerate(filenames):
                img = np.array(Image.open(path + filename).resize((self.imgSize, self.imgSize)))
                img = img / 255.0  # 归一化

                img = img.reshape(1, self.imgSize, self.imgSize, 3)  # 规范维度
                ans = self.model.predict(img)
                #print(ans)

                ansLabel = np.argmax(ans)
                ansLabel = class_indices[ansLabel]

                print(index, filename[:-4], ansLabel)

                record = str(ansLabel)+'\t'+str(filename[:-4])+"\n"
                submit.writelines(record)

        submit.close()


    def save(self, modelPath):
        pass

class BiCNNVGG16(BaseModel):

    def buildModel(self):

        # 输入层
        input = Input(shape=(self.imgSize, self.imgSize, 3), name='image_input')

        # 获取预训练的卷基层
        baseModel = VGG16(weights='imagenet', include_top=False)
        baseModel.summary()
        net1 = Model(inputs=baseModel.input, outputs=baseModel.get_layer('block5_conv3').output)
        net1.summary()

        #net2 = VGG16(weights='imagenet', include_top=False)

        # for layer in net2.layers:
        #     layer.name = layer.name + str("_two")
        # net2.summary()



        net1.name = "vgg1"
        #net2.name = "vgg2"
        # 预训练的卷基层
        output_vgg16_conv1 = net1(inputs=input)
        #output_vgg16_conv2 = net1(inputs=input)

        # for layer in net1.layers:
        #     layer.trainable = False
        # for layer in net2.layers:
        #     layer.trainable = False

        z_l2 = MyLayer()([output_vgg16_conv1, output_vgg16_conv1])

        fc = Dense(100, activation="softmax", trainable=True)(z_l2)

        # 建模
        my_model = Model(input=input, output=fc)

        return my_model

    def trainModel(self, generator, validation_generator, batchSize):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        self.model.fit_generator(generator, steps_per_epoch=10000 // batchSize, epochs=5,
                                 validation_data=validation_generator,
                                 validation_steps=4000 // batchSize,
                                 #                        verbose=1, callbacks=[early_stopping]
                                 )

    def save(self, modelPath):
        self.model.save_weights(modelPath)