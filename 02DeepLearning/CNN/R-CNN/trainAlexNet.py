import os
import tflearn
import codecs
import cv2
import numpy as np
import pickle


from AlexNet import create_alexnet
import config
import util
from SelectiveSearch import SelectiveSearch
from DataPreprocessing import loadPreTrainData, loadFineTuneData



################ 用17类大数据集 预训练 AlexNet #################
def preTrainCNN_By17flowers(save=False):
    X, Y = loadPreTrainData()
    network = create_alexnet(config.TRAIN_CLASS, restore=False)
    # Training
    save_model_path = config.SAVE_MODEL_PATH   # './pre_train_model/model_save.model'
    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir=config.TENSORBOARD_DIR)
    if os.path.isfile(save_model_path + '.index'):
        model.load(save_model_path)
        print('load model...')
    for _ in range(1000):
        model.fit(X, Y, n_epoch=1, validation_set=0.1, shuffle=True,
                  show_metric=True, batch_size=200, snapshot_step=200,
                  snapshot_epoch=False, run_id='alexnet_oxflowers17')  # epoch = 1000
        # Save the model
        model.save(save_model_path)
        print('save model...')

######################### 小数据集微调Alexnet ######################
def fineTuneCNN_By2flowers():
    save_model_path = config.SAVE_MODEL_PATH
    fine_tune_model_path = config.FINE_TUNE_MODEL_PATH
    print("Loading Data")
    X, Y = loadFineTuneData()
    restore = False
    if os.path.isfile(config.FINE_TUNE_MODEL_PATH + '.index'):
        restore = True
        print("Continue fine-tune")
    # three classes include background
    network = create_alexnet(config.FINE_TUNE_CLASS, restore=restore)

    # Training
    model = tflearn.DNN(network, checkpoint_path='rcnn_model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir=config.TENSORBOARD_DIR)
    if os.path.isfile(fine_tune_model_path + '.index'):
        print("Loading the fine tuned model")
        model.load(fine_tune_model_path)
    elif os.path.isfile(save_model_path + '.index'):
        print("Loading the alexnet")
        model.load(save_model_path)
    else:
        print("No file to load, error")
        return False
    for _ in range(100):
        model.fit(X, Y, n_epoch=1, validation_set=0.1, shuffle=True,
                  show_metric=True, batch_size=200, snapshot_step=200,
                  snapshot_epoch=False, run_id='alexnet_rcnnflowers2')
        # Save the model
        model.save(fine_tune_model_path)

if __name__ == '__main__':
    # preTrainCNN_By17flowers(save=True)
    fineTuneCNN_By2flowers()



















