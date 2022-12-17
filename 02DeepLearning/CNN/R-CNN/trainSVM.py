import os
import tflearn
from sklearn import svm
import joblib

from AlexNet import create_alexnet
import config
from DataPreprocessing import loadSVMdata
import util


def train_svms():
    train_file_folder = config.TRAIN_SVM
    net = create_alexnet(istrain=False)
    model = tflearn.DNN(net)
    model.load(config.FINE_TUNE_MODEL_PATH)
    svms = []
    for fold in ["1","2"]:
        data_path = os.path.join(train_file_folder, fold)
        X, Y = loadSVMdata(data_path + "_svmData.npy")
        train_features = []
        for ind, i in enumerate(X):
            # extract features
            feats = model.predict([i])
            train_features.append(feats[0])
            util.view_bar("extract features of %s" % data_path, ind + 1, len(X))
        print("  ")
        # SVM training
        clf = svm.LinearSVC(max_iter=10000)
        print("fit svm")
        clf.fit(train_features, Y)
        svms.append(clf)
        joblib.dump(clf, data_path + '_svm.model')
    return svms




if __name__ == '__main__':
    train_svms()
