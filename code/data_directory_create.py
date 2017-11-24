import pandas as pd
import os
from skimage import io

# 产生类别文件夹
def create_directory(rootPath):
    labelSet = pd.read_csv("../data/raw_data/label_name.txt", delimiter=" ", header=-1)
    labelNum = len(labelSet)
    for i in range(labelNum):
        os.mkdir(rootPath+str(i))

# train集数据整理
def createTrainSet(save_suffix):

    train = pd.read_csv("../data/raw_data/data_train_image.txt", delimiter=" ", header=-1)
    trainFileNum = len(train)
    path_suffix = "../data/raw_data/train_crop/"
    for i in range(trainFileNum):
        label = train.ix[i, 1]  # 0是图片文件名，1是label，2是URL
        imgName = str(train.ix[i, 0])+".jpg"
        print(i, imgName, label)
        img = io.imread(path_suffix+imgName)
        io.imsave(save_suffix+str(label)+"/"+imgName, img)

# val集数据整理
def createValidSet(save_suffix1):

    valid = pd.read_csv("../data/raw_data/val.txt", delimiter=" ", header=-1)
    validFileNum = len(valid)
    print(validFileNum)
    path_suffix = "../data/raw_data/val_crop/"
    #save_suffix = "../data/train/"

    # for i in range(0, validFileNum-validNum):
    #     label = valid.ix[i, 1]  # 0是图片文件名，1是label，2是URL
    #     imgName = str(valid.ix[i, 0])+".jpg"
    #     print(i, imgName, label)
    #     img = io.imread(path_suffix + imgName)
    #     io.imsave(save_suffix1 + str(label) + "/" + imgName, img)

    for i in range(validFileNum):
        label = valid.ix[i, 1]  # 0是图片文件名，1是label，2是URL
        imgName = str(valid.ix[i, 0]) + ".jpg"
        print(i, imgName, label)
        img = io.imread(path_suffix + imgName)
        io.imsave(save_suffix1 + str(label) + "/" + imgName, img)

def createTestSet(save_suffix):

    test = pd.read_csv("../data/submission6.txt", delimiter="\t", header=-1)
    testFileNum = len(test)
    print(testFileNum)
    path_suffix = "../data/test_crop/"

    for i in range(testFileNum):
        label = test.ix[i, 0]  # 0是label，1是文件名
        imgName = str(test.ix[i, 1]) + ".jpg"
        print(i, imgName, label)
        img = io.imread(path_suffix + imgName)
        io.imsave(save_suffix + str(label) + "/" + imgName, img)


def cleanDirectory(path):

    deleteNum = 0
    files = os.listdir(path)  # 获取路径下的子文件(夹)列
    for file in files:
        if os.path.isdir(path+file):  # 如果是文件夹
            if not os.listdir(path+file):  # 如果子文件为空
                os.rmdir(path+file)  # 删除这个空文件夹
                deleteNum += 1
                print("empty classes:", file)
    print(deleteNum)

if __name__ == "__main__":

    # create_directory("../data/train_data_crop/")
    # create_directory("../data/valid_data_crop/")
    #create_directory("../data/total_train_crop/")
    create_directory("../data/test_with_label/")

    # print("start create train...")
    #createTrainSet("../data/train_data_crop/")
    #createValidSet("../data/valid_data_crop/")
    createTestSet("../data/test_with_label/")

    # print("start create train and valid...")
    # createTrainSet("../data/train_data/")
    # createValidSet("../data/valid_data/")

    #createTrainSet("../data/total_train_crop/")
    #createValidSet("../data/total_train_crop/")

    # cleanDirectory("../data/train_data_crop/")
    # cleanDirectory("../data/valid_data_crop/")
    #cleanDirectory("../data/total_train_crop/")
    cleanDirectory("../data/test_with_label/")




