from ultralytics import YOLO
import numpy as np
#import multiprocessing
def train():
    model = YOLO(r"D:\08weight\08yolov8\8.2.0/yolov8s.pt")  # build a new model from scratch
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    model.train(cfg="ultralytics/cfg/default.yaml", data=r"ultralytics/cfg/datasets/coco128.yaml", epochs=2, batch=1)

def train_cls():
    model = YOLO(r"D:\08weight\08yolov8\8.2.0\yolov8s-cls.pt")  # build a new model from scratch
    # model = YOLO(r"D:\00myGitHub\aFolder_YOLO\03yolov8s\ultralytics\runs\detect\train11\weights/best.pt")  # build a new model from scratch
    # model.train(cfg="ultralytics/cfg/default.yaml", epochs=2)
    # model.train(data=r"D:\02dataset\imagenette2-160", epochs=2)
    # model.export(format="onnx")
    img_path = r"D:\02dataset\imgtest/224.jpg"         # 0607 1_1
    model.predict(img_path, save=True, save_txt=True, imgsz=640, conf=0.5, line_width=1, save_conf=False)


def predict():
    model = YOLO(r"D:\08weight\08yolov8\8.2.0/yolov8s6.pt")  # build a new model from scratch
    # model = YOLO(r"D:\00myGitHub\aFolder_YOLO\03yolov8s\ultralytics\runs\detect\train11\weights/best.pt")  # build a new model from scratch
    # model = YOLO(r"D:\03GitHub\aFolder_YOLO\03yolov8s\ultralytics\runs\detect\little_no_mosaic8\weights/last.pt")  # build a new model from scratch
    img_path = r"D:\02dataset\imgtest/000000000036.jpg"         # 0607 1_1
    model.predict(img_path, save=True, save_txt=True, imgsz=640, conf=0.5, line_width=1, save_conf=False)

if __name__ == '__main__':
    #multiprocessing.freeze_support()
    #model = YOLO("yolov8n.yaml")  # build a new model from scratch
    # model = YOLO("yolov8s.pt")  # build a new model from scratch
    # model = YOLO(r"D:\00myGitHub\aFolder_YOLO\03yolov8s\ultralytics\runs\detect\train7\weights/best.pt")  # build a new model from scratch
    #model = YOLO("ultralytics/models/v8/yolov8.yaml")
    #model.train(cfg="ultralytics/yolo/cfg/default.yaml", epochs=3)
    #model.train(cfg="ultralytics/yolo/cfg/LG.yaml", epochs=30)
    # Validate the model
    #metrics = model.val(cfg="ultralytics/yolo/cfg/LG.yaml")  # no arguments needed, dataset and settings remembered
    # metrics.box.map  # map50-95
    # metrics.box.map50  # map50
    # metrics.box.map75  # map75
    # metrics.box.maps  # a list contains map50-95 of each category
    #img_path = r"D:\02dataset\01work\00dataTrain\01NanJingLG\01coco128\images\valAIDI0519/"
    # img_path = r"F:\15project\02kd\03LG\01imgAIDI\01AIDI_SZ\img2/"
    # model.predict(img_path, save=True, save_txt=True, imgsz=1024, conf=0.2)
    train()
    pass