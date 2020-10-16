import tflearn
import os
import joblib
import tensorflow as tf

import util
from util import makeDirs
import config
from AlexNet import create_alexnet
from trainAlexNet import preTrainCNN_By17flowers, fineTuneCNN_By2flowers
from trainSVM import train_svms
from DataPreprocessing import preprocessingData, image_proposal

def RCNN_Train():
    # 1. 创建文件夹、预处理数据
    makeDirs()
    preprocessingData()

    # 2. 用17类大数据集 预训练 AlexNet
    with tf.Graph().as_default():
        preTrainCNN_By17flowers()

    # 3. 小数据集微调Alexnet
    with tf.Graph().as_default():
        fineTuneCNN_By2flowers()

    # 4. 训练SVM
    with tf.Graph().as_default():
        train_svms()

# 预测
def RCNN_Predict():
    train_file_folder = config.TRAIN_SVM
    img_path = config.DATA_PATH + '17flowers/jpg/7/image_0614.jpg'  # or './17flowers/jpg/16/****.jpg'
    # img_path = config.DATA_PATH + '17flowers/jpg/16/image_1356.jpg'
    imgs, verts = image_proposal(img_path)
    util.show_rect(img_path, verts)

    net = create_alexnet(istrain=False)
    model = tflearn.DNN(net)
    model.load(config.FINE_TUNE_MODEL_PATH)

    svms = []
    for file in os.listdir(train_file_folder):
        if file.split('_')[-1] == 'svm.model':
            svms.append(joblib.load(os.path.join(train_file_folder, file)))

    features = model.predict(imgs)
    results = []
    results_label = []
    count = 0
    for f in features:
        for svm in svms:
            pred = svm.predict([f.tolist()])
            # not background
            if pred[0] != 0:
                results.append(verts[count])
                results_label.append(pred[0])
        count += 1
    print("result:")
    print(results)
    print("result label:")
    print(results_label)
    util.show_rect(img_path, results)
    for i in range(len(results)):
        util.show_rect(img_path, [results[i]])


if __name__ == '__main__':
    RCNN_Train()
    RCNN_Predict()







