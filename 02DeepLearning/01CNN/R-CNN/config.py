# config path and files
IMAGE_SIZE = 224
DATA_PATH = "D:/ML_datas/17flowersRCNN/"
SAVE_MODEL_PATH = DATA_PATH + '/pre_train_model/model_save.model'
FINE_TUNE_MODEL_PATH = DATA_PATH + '/fine_tune_model/fine_tune_model_save.model'
TENSORBOARD_DIR = DATA_PATH + "logs"

TRAIN_LIST = DATA_PATH + 'train_list.txt'
FINE_TUNE_LIST = DATA_PATH + '/fine_tune_list.txt'

TRAIN_SVM = DATA_PATH + 'svm_train'
TRAIN_CLASS = 17
FINE_TUNE_CLASS = 3