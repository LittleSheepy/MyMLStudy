import tensorflow as tf
print(tf.__version__)


from yolo.YOLOv2 import YOLOv2
from yolo.YOLOv3 import YOLOv3
from ssd.SSD300 import SSD300



if __name__ == '__main__':
    mode = YOLOv3()
    mode.train()

    #mode.test_image()
