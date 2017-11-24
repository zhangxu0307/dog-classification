import pandas as pd
from code.model1 import VGG16CNNModel, ResNet50Model, XInceptionModel, BigModel
from code.model_2 import ResNet101Model

def predict(modelPath, submissionFile):

    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    # test_generator = test_datagen.flow_from_directory(
    #     '../data/test/test/',
    #     target_size=(128, 128),
    #     batch_size=48, shuffle=False)


    #model = VGG16CNNModel(load=True, loadModelPath=modelPath)
    #model = ResNet50Model(imgSize=224, load=True, loadW=True, loadModelPath=modelPath)
    #model = ResNet101Model(imgSize=224, load=True, loadW=True, loadModelPath=modelPath)
    model = XInceptionModel(imgSize=224, load=True, loadModelPath=modelPath)
    #model = BigModel(imgSize=224, loadfc=True, loadModelPath=modelPath)
    ansList = model.inference(submissionFile)
    print(ansList)

if __name__ == "__main__":

    modelPath = "../model/Xinception_crop_2.h5" # 加载的模型路径
    submissionFile = "../data/submission7.txt" # 结果文件
    predict(modelPath,submissionFile)