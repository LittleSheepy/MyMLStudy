import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import config


# Building 'AlexNet'
def create_alexnet(num_classes=None, restore=True, istrain=True):
    network = input_data(shape=[None, config.IMAGE_SIZE, config.IMAGE_SIZE, 3])     # (?,224,224, 3)
    network = conv_2d(network, 96, 11, strides=4, activation='relu')                # (?, 56, 56,96)
    network = max_pool_2d(network, 3, strides=2)                                    # (?, 28, 28,96)
    network = local_response_normalization(network)                                 # (?, 28, 28,96)

    network = conv_2d(network, 256, 5, activation='relu')                           # (?, 28, 28,256)
    network = max_pool_2d(network, 3, strides=2)                                    # (?, 14, 14,256)
    network = local_response_normalization(network)                                 # (?, 14, 14,256)

    network = conv_2d(network, 384, 3, activation='relu')                           # (?, 14, 14,384)
    network = conv_2d(network, 384, 3, activation='relu')                           # (?, 14, 14,384)
    network = conv_2d(network, 256, 3, activation='relu')                           # (?, 14, 14,256)
    network = max_pool_2d(network, 3, strides=2)                                    # (?,  7,  7,256)
    network = local_response_normalization(network)                                 # (?,  7,  7,256)

    network = fully_connected(network, 4096, activation='tanh')                     # (?, 4096)
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')                     # (?, 4096)
    if istrain :
        network = dropout(network, 0.5)
        network = fully_connected(network, num_classes, activation='softmax', restore=restore)
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network
