# -*- coding: utf-8 -*-

from keras.models import Sequential
from code.model1 import BaseModel
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, LSTM, TimeDistributed,\
    AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation, Merge, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import numpy as np
from PIL import Image

from sklearn.metrics import log_loss
from keras import regularizers

from code.custom_layers import Scale
import keras
from code.multi_GPU import make_parallel


def global_average_pooling(x):
    return K.mean(x, axis=(2, 3))


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


import sys
sys.setrecursionlimit(3000)

def identity_block(input_tensor, kernel_size, filters, stage, block, reg = 0.0, trainable = True):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1,kernel_regularizer=regularizers.l2(reg), trainable = trainable,
                      name=conv_name_base + '2a', bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Convolution2D(nb_filter2, kernel_size, kernel_size, kernel_regularizer=regularizers.l2(reg),trainable =trainable,
                      name=conv_name_base + '2b', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Convolution2D(nb_filter3, 1, 1,kernel_regularizer=regularizers.l2(reg),trainable =trainable,
                      name=conv_name_base + '2c', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum', name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), reg = 0.0, trainable = True):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides, kernel_regularizer=regularizers.l2(reg),trainable = trainable,
                      name=conv_name_base + '2a', bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Convolution2D(nb_filter2, kernel_size, kernel_size,kernel_regularizer=regularizers.l2(reg),trainable = trainable,
                      name=conv_name_base + '2b', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c',  kernel_regularizer=regularizers.l2(reg),trainable = trainable,
                      bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides, kernel_regularizer=regularizers.l2(reg),trainable = trainable,
                             name=conv_name_base + '1', bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum', name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

class ResNet101Model(BaseModel):

    def buildModel(self):
        """
        Resnet 101 Model for Keras
        Model Schema and layer naming follow that of the original Caffe implementation
        https://github.com/KaimingHe/deep-residual-networks
        ImageNet Pretrained Weights
        Theano: https://drive.google.com/file/d/0Byy2AcGyEVxfdUV1MHJhelpnSG8/view?usp=sharing
        TensorFlow: https://drive.google.com/file/d/0Byy2AcGyEVxfTmRRVmpGWDczaXM/view?usp=sharing
        Parameters:
          img_rows, img_cols - resolution of inputs
          channel - 1 for grayscale, 3 for color
          num_classes - number of class labels for our classification task
        """
        eps = 1.1e-5
        self.reg = 0.0001

        # Handle Dimension Ordering for different backends
        global bn_axis
        if K.image_dim_ordering() == 'tf':
          bn_axis = 3
          img_input = Input(shape=(self.imgSize, self.imgSize, 3), name='data')
        else:
          bn_axis = 1
          img_input = Input(shape=(3, self.imgSize, self.imgSize), name='data')

        x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
        x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1', bias=False,trainable=True,
                          kernel_regularizer=regularizers.l2(self.reg))(x)
        x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
        x = Scale(axis=bn_axis, name='scale_conv1')(x)
        x = Activation('relu', name='conv1_relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), reg=self.reg,trainable=True)
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', reg=self.reg,trainable=True)
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', reg=self.reg,trainable=True)

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', reg=self.reg,trainable=True)
        for i in range(1,3):
          x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i), reg=self.reg,trainable=True)

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', reg=self.reg,trainable=True)
        for i in range(1,23):
          x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i), reg=self.reg, trainable=True)

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', reg=self.reg, trainable=True)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', reg=self.reg, trainable=True)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', reg=self.reg, trainable=True)

        x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
        x_fc = Flatten()(x_fc)
        x_fc = Dense(1000, activation='softmax', name='fc1000', trainable=True)(x_fc)

        model = Model(img_input, x_fc)

        if K.image_dim_ordering() == 'th':
          # Use pre-trained weights for Theano backend
          weights_path = '../model/resnet101_weights_th.h5'
        else:
          # Use pre-trained weights for Tensorflow backend
          weights_path = '../model/resnet101_weights_tf.h5'

        model.load_weights(weights_path, by_name=True)

        # Truncate and replace softmax layer for transfer learning
        # Cannot use model.layers.pop() since model is not of Sequential() type
        # The method below works since pre-trained weights are stored in layers but not in the model
        x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
        x_newfc = Flatten()(x_newfc)

        #x_newfc = Lambda(global_average_pooling, output_shape=global_average_pooling_shape)(x)
        #x_newfc = keras.layers.pooling.GlobalAveragePooling2D(dim_ordering='default')(x)
        x_newfc = Dense(100, kernel_regularizer=regularizers.l2(self.reg), activation='softmax', name='fc8')(x_newfc)

        model = Model(img_input, x_newfc)

        return model

    def trainModel(self, generator, validation_generator, batchSize): # 训练模型

        adjustLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0, mode='auto',
                                          epsilon=0.001, cooldown=0, min_lr=5e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        self.model.fit_generator(generator, steps_per_epoch=10000 // batchSize, epochs=50,
                                validation_data=validation_generator,
                                validation_steps=4000 // batchSize,
                                verbose=1,
                                callbacks=[adjustLR]
                                 )

    def save(self, modelPath):
        self.model.save_weights(modelPath)

    def inference(self, submissionFile):

        #self.model = self.buildModel()
        self.model.load_weights("../model/resnet101_1.h5")

        ansList = []
        path = "../data/test/test-01/image/"
        # submit = pd.read_csv("../data/submission.csv")
        submit = open(submissionFile, "w+")

        class_indices = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
                         '11': 11, '12': 12,
                         '13': 13, '14': 14, '16': 15, '17': 16, '18': 17, '19': 18, '20': 19, '21': 20, '22': 21,
                         '23': 22, '24': 23,
                         '25': 24, '26': 25, '27': 26, '28': 27, '29': 28, '30': 29, '31': 30, '32': 31, '33': 32,
                         '34': 33, '35': 34,
                         '36': 35, '37': 36, '38': 37, '39': 38, '40': 39, '41': 40, '42': 41, '43': 42, '45': 43,
                         '46': 44, '47': 45,
                         '48': 46, '49': 47, '50': 48, '51': 49, '52': 50, '53': 51, '54': 52, '57': 53, '59': 54,
                         '60': 55, '61': 56,
                         '62': 57, '63': 58, '64': 59, '65': 60, '66': 61, '67': 62, '68': 63, '69': 64, '70': 65,
                         '71': 66, '72': 67,
                         '73': 68, '74': 69, '75': 70, '76': 71, '77': 72, '78': 73, '79': 74, '80': 75, '81': 76,
                         '82': 77, '83': 78,
                         '84': 79, '85': 80, '86': 81, '87': 82, '88': 83, '94': 84, '95': 85, '97': 86, '101': 87,
                         '109': 88,
                         '111': 89, '114': 90, '115': 91, '120': 92, '123': 93, '126': 94, '127': 95, '128': 96,
                         '129': 97, '132': 98,
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
                # print(ans)

                ansLabel = np.argmax(ans)
                ansLabel = class_indices[ansLabel]

                print(index, filename[:-4], ansLabel)

                record = str(ansLabel) + '\t' + str(filename[:-4]) + "\n"
                submit.writelines(record)

        submit.close()

if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10

    # img_rows, img_cols = 224, 224 # Resolution of inputs
    # channel = 3
    # num_classes = 10
    # batch_size = 16
    # nb_epoch = 10
    #
    # # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    # X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)
    #
    # # Load our model
    # model = resnet101_model(img_rows, img_cols, channel, num_classes)
    #
    # # Start Fine-tuning
    # model.fit(X_train, Y_train,
    #           batch_size=batch_size,
    #           nb_epoch=nb_epoch,
    #           shuffle=True,
    #           verbose=1,
    #           validation_data=(X_valid, Y_valid),
    #           )
    #
    # # Make predictions
    # predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
    #
    # # Cross-entropy loss score
    # score = log_loss(Y_valid, predictions_valid)
    pass

