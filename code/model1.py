from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input, Flatten, Dense, Activation, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, concatenate
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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from code.custom_layers import Scale
from code.multi_GPU import make_parallel
import keras.backend as K
import keras
from keras.utils import plot_model
from code.extra_layer import BilinearLayer

def categorical_hinge(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)
    return K.maximum(0., neg - pos + 1.)


# 模型基类
class BaseModel():

    def __init__(self, imgSize, load=False, loadW = False, loadModelPath=None): # 选择是否加载已有模型文件或构建模型训练

        self.imgSize = imgSize
        lr = 1e-5
        LOSS = 'categorical_crossentropy'
        #LOSS = categorical_hinge
        if load:
            if loadW: # 加载权重
                self.model = self.buildModel()
                self.model.load_weights(loadModelPath)
                self.model.compile(
                                   loss=LOSS,
                                    #loss = triplet_loss,
                                   optimizer=optimizers.SGD(lr=lr,
                                                            momentum=0.9,
                                                            decay=0.0001,
                                                            nesterov=True,
                                                            ),
                                   #optimizer=optimizers.Adam(lr=1e-3),
                                   metrics=['categorical_accuracy'])
            else: # 加载模型和权重
                self.model = load_model(loadModelPath)
        else:
            self.model = self.buildModel()
            #self.model = make_parallel(self.model, 2)
            self.model.compile(loss=LOSS,
                               optimizer=optimizers.SGD(lr=lr,
                                                        momentum=0.9,
                                                        decay=0.0001,
                                                        nesterov=True,
                                                        ),
                               #optimizer=optimizers.Adam(lr=lr),
                               metrics=['categorical_accuracy'])

        self.model.summary()


    def buildModel(self): # 子类实现具体模型结构
        pass

    def trainModel(self, generator, validation_generator, batchSize): # 训练模型

        pass

    def inference(self, submissionFile): # 预测

        ansList = []
        #path = "../data/test/test-01/image/"
        path="../data/test_crop/"
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


class VGG16CNNModel(BaseModel):

    def buildModel(self):

        # 获取预训练的卷基层
        model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
        model_vgg16_conv.summary()

        # 输入层，尺寸大小为128*128*3
        input = Input(shape=(self.imgSize, self.imgSize, 3),name = 'image_input')

        # 预训练的卷基层
        output_vgg16_conv = model_vgg16_conv(input)

        for layer in model_vgg16_conv.layers:
            layer.trainabale = True

        # 加入的全连接层
        #x = Flatten(name='flatten')(output_vgg16_conv)
        # x = Dense(4096, activation="relu", name="fc1")(x)
        # x = Dropout(0.5)(x)
        # x = Dense(1024, activation="relu",  name="fc2")(x)
        # x = Dropout(0.5)(x)
        x = keras.layers.pooling.GlobalAveragePooling2D(dim_ordering='default')(output_vgg16_conv)
        x = Dense(100, activation='softmax')(x)

        # 建模
        my_model = Model(input=input, output=x)

        return my_model

    def trainModel(self, generator, validation_generator, batchSize): # 训练模型

        # 冻结前20层

            #layer.trainable = False
        #early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        self.model.fit_generator(generator, steps_per_epoch=9000 // batchSize, epochs=10,
                                 validation_data=validation_generator,
                                 validation_steps=4000 // batchSize,
                                 verbose=1,
                                 #callbacks=[early_stopping]
                                 )

    def save(self, modelPath):
        self.model.save(modelPath)

class ResNet50Model(BaseModel):

    def buildModel(self):

        # 获取预训练的卷基层
        model_resnet50 = ResNet50(weights='imagenet', include_top=False, pooling="avg")
        model_resnet50.summary()

        # 输入层
        input = Input(shape=(self.imgSize, self.imgSize, 3),name = 'image_input')

        # 预训练的卷基层
        output_resnet50 = model_resnet50(input)

        # 加入的全连接层
        x = output_resnet50
        #x = Flatten(name='flatten')(output_resnet50)
        #x = Dense(1024, activation="relu", name="fc1")(x)
        #x = Dropout(0.5)(x)
        #x = Dense(512, activation="relu", name="fc1")(x) # 此处改成avgpool，只用一层，去掉dropout
        #x = Dropout(0.7)(x)
        x = Dense(100, activation='softmax')(x)

        #model_resnet50.trainabel = False # 冻结全部resnet的卷积权重

        # for layer in model_resnet50.layers[0:-33]: # 冻结前面的卷积层
        #     layer.trainable = False

        # 建模
        my_model = Model(input=input, output=x)

        return my_model

    def trainModel(self, generator, validation_generator, batchSize): # 训练模型

        adjustLR = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=0, mode='auto',
                                     epsilon=0.001, cooldown=0, min_lr=5e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        self.model.fit_generator(generator, steps_per_epoch=20000 // batchSize, epochs=15,
                                # validation_data=validation_generator,
                                # validation_steps=10000 // batchSize,
                                verbose=1,
                                callbacks=[adjustLR]
                                 )

    def save(self, modelPath):
        self.model.save(modelPath)


class InceptionV3Model(BaseModel):

    def buildModel(self):

        # 获取预训练的卷基层
        inceptionv3 = InceptionV3(weights='imagenet', include_top=False, pooling="avg")
        inceptionv3.summary()

        # 输入层
        input = Input(shape=(self.imgSize, self.imgSize, 3),name = 'image_input')

        # 预训练的卷基层
        output_inceptionv3  = inceptionv3(input)

        # 加入的全连接层
        x = output_inceptionv3
        x = Dense(100, activation='softmax')(x)

        #model_resnet50.trainabel = False # 冻结全部resnet的卷积权重

        # for layer in model_resnet50.layers[0:-93]: # 冻结前面的卷积层
        #     layer.trainable = False

        # 建模
        my_model = Model(input=input, output=x)

        return my_model

    def trainModel(self, generator, validation_generator, batchSize): # 训练模型

        adjustLR = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=0, mode='auto',
                                     epsilon=0.001, cooldown=0, min_lr=5e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        self.model.fit_generator(generator, steps_per_epoch=20000 // batchSize, epochs=10,
                                # validation_data=validation_generator,
                                # validation_steps=4000 // batchSize,
                                verbose=1,
                                 callbacks=[adjustLR ]
                                 )

    def save(self, modelPath):
        self.model.save(modelPath)

class XInceptionModel(BaseModel):

    def buildModel(self):

        # 获取预训练的卷基层
        Xinceptionv = Xception(weights='imagenet', include_top=False, pooling="avg")
        Xinceptionv.summary()

        #plot_model(Xinceptionv, to_file='../data/model.png', show_shapes=True)

        # 输入层
        input = Input(shape=(self.imgSize, self.imgSize, 3), name = 'image_input')

        # 预训练的卷基层
        output_Xinceptionv = Xinceptionv(input)


        # 加入的全连接层
        x = output_Xinceptionv
        # x1 = GlobalAveragePooling2D()(x)
        # x2 = GlobalMaxPooling2D()(x)
        # x = concatenate([x1, x2])

        #x = Flatten(name='flatten')(output_resnet50)
        #x = Dense(1024, activation="relu", name="fc1")(x)
        #x = Dropout(0.5)(x)
        #x = Dense(512, activation="relu", name="fc1")(x) # 此处改成avgpool，只用一层，去掉dropout
        #x = Dropout(0.5)(x)

        x = Dense(100, kernel_regularizer=regularizers.l2(0.01), activation='softmax')(x)

        #model_resnet50.trainabel = False # 冻结全部resnet的卷积权重

        # for layer in Xinceptionv.layers: # 冻结前面的卷积层
        #     layer.trainable = False

        # 建模
        my_model = Model(input=input, output=x)

        return my_model

    def trainModel(self, generator, validation_generator, batchSize): # 训练模型

        adjustLR = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=0, mode='auto',
                                     epsilon=0.001, cooldown=0, min_lr=5e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        self.model.fit_generator(generator, steps_per_epoch=20000 // batchSize, epochs=5,
                                # validation_data=validation_generator,
                                # validation_steps=4000 // batchSize,
                                verbose=1,
                                 callbacks=[adjustLR]
                                 )

    def save(self, modelPath):
        self.model.save(modelPath)


class BigModel():


    def __init__(self, imgSize, loadfc = False, loadModelPath=None): # 选择是否加载已有模型文件或构建模型训练

        self.imgSize = imgSize
        lr = 1e-4
        LOSS = 'categorical_crossentropy'
        if loadfc:
            # self.model = (loadModelPath)
            # self.model.summary()

            self.model = self.buildModel()
            self.model.load_weights(loadModelPath)
        else:
            self.model = self.buildModel()
        self.model.compile(loss=LOSS,
                           optimizer=optimizers.SGD(lr=lr,
                                                    momentum=0.9,
                                                    decay=0.0001,
                                                    nesterov=True,
                                                    ),
                           #optimizer=optimizers.Adam(lr=lr),
                           metrics=['categorical_accuracy'])

        self.model.summary()


    def buildModel(self):

        # 获取预训练的卷基层
        # xception = Xception(weights='imagenet', include_top=False, pooling="avg")
        # inceptionv3 = InceptionV3(weights='imagenet', include_top=False, pooling="avg")
        # resnet50 = ResNet50(weights='imagenet', include_top=False, pooling="avg")

        # 加载fine-tune的网络

        base_resnet50 = load_model("../model/resnet50_crop.h5")
        base_resnet50.summary()
        print("----------------------")
        base_inceptionv3 = load_model("../model/inceptionv3_crop.h5")
        base_inceptionv3.summary()
        print("----------------------")
        base_xception = load_model("../model/Xinception_crop_2.h5")
        base_xception.summary()
        print("----------------------")


        # 输入层
        input = Input(shape=(self.imgSize, self.imgSize, 3), name='image_input')

        # 获取gap层的输出
        xception = Model(inputs=base_xception.input, outputs=base_xception.get_layer("xception").get_output_at(-1))
        inceptionv3 = Model(inputs=base_inceptionv3.input, outputs=base_inceptionv3.get_layer('inception_v3').get_output_at(-1))
        resnet50 = Model(inputs=base_resnet50.input, outputs=base_resnet50.get_layer("resnet50").get_output_at(-1))

        x1 = xception(input)
        x2 = inceptionv3(input)
        x3 = resnet50(input)

        # 连接gap层
        gap = concatenate([x1, x2, x3])

        x = Dense(100, activation='softmax')(gap)

        # 冻结卷积层参数
        xception.trainable = False
        inceptionv3.trainable = False
        resnet50.trainabel = False # 冻结全部卷积权重

        # 建模
        fc_model = Model(input=input, output=x)

        return fc_model

    def trainModel(self, generator, validation_generator, batchSize): # 训练模型

        adjustLR = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=0, mode='auto',
                                     epsilon=0.001, cooldown=0, min_lr=5e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        self.model.fit_generator(generator, steps_per_epoch=20000 // batchSize, epochs=10,
                                # validation_data=validation_generator,
                                # validation_steps=4000 // batchSize,
                                verbose=1,
                                 callbacks=[adjustLR]
                                 )

    def save(self, modelPath):
        self.model.save_weights(modelPath)

    def inference(self, submissionFile): # 预测

        ansList = []
        #path = "../data/test/test-01/image/"
        path="../data/test_crop/"
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


