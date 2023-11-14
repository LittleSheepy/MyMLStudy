import json
import os
import glob
import os.path as osp
import cv2


def labelme2yolov2Seg(jsonfilePath="", resultDirPath="", classList=["JQJPS"]):
    """
    此函数用来将labelme软件标注好的数据集转换为yolov5_7.0sege中使用的数据集
    :param jsonfilePath: labelme标注好的*.json文件所在文件夹
    :param resultDirPath: 转换好后的*.txt保存文件夹
    :param classList: 数据集中的类别标签
    :return:
    """
    # 0.创建保存转换结果的文件夹
    if(not os.path.exists(resultDirPath)):
        os.mkdir(resultDirPath)

    # 1.获取目录下所有的labelme标注好的Json文件，存入列表中
    # jsonfileList = glob.glob(osp.join(jsonfilePath, "*.json"))
    # print(jsonfileList)  # 打印文件夹下的文件名称
    json_name_list = os.listdir(jsonfilePath)
    # 2.遍历json文件，进行转换
    for jsonname in json_name_list:
        print(jsonname)
        jsonfile = os.path.join(jsonfilePath, jsonname)
        img_path = os.path.join(img_dir, jsonname.replace(".json", ".jpg"))
        txt_path = os.path.join(resultDirPath, jsonname.replace(".json", ".txt"))
        # 3. 打开json文件
        with open(jsonfile, "r") as f:
            file_in = json.load(f)

            h, w = file_in["imageHeight"], file_in["imageWidth"]
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
            # 4. 读取文件中记录的所有标注目标
            shapes = file_in["shapes"]

            # 5. 使用图像名称创建一个txt文件，用来保存数据
            with open(txt_path, "w") as file_handle:
                # 6. 遍历shapes中的每个目标的轮廓
                for shape in shapes:
                    # 7.根据json中目标的类别标签，从classList中寻找类别的ID，然后写入txt文件中
                    #file_handle.writelines(str(classList.index(shape["label"])) + " ")
                    file_handle.writelines(str(0) + " ")

                    # 8. 遍历shape轮廓中的每个点，每个点要进行图像尺寸的缩放，即x/width, y/height
                    for point in shape["points"]:
                        x = round(point[0]/w, 6)  # mask轮廓中一点的X坐标
                        y = round(point[1]/h, 6)  # mask轮廓中一点的Y坐标
                        file_handle.writelines(str(x) + " " + str(y) + " ")  # 写入mask轮廓点

                    # 9.每个物体一行数据，一个物体遍历完成后需要换行
                    file_handle.writelines("\n")
            # 10.所有物体都遍历完，需要关闭文件
            file_handle.close()
        # 10.所有物体都遍历完，需要关闭文件
        f.close()

if __name__ == "__main__":
    root_dir = r"E:\0ProjectData\AI_PK_pictreue\5_Front_YW\/"
    right_str = ""
    jsonfilePath = root_dir + r"\json" + right_str + "/"    # 要转换的json文件所在目录
    resultDirPath = root_dir + r"\txt" + right_str + "/"    # 要生成的txt文件夹

    img_dir = root_dir + r"\img" + right_str + "/"
    if not os.path.exists(resultDirPath):
        os.mkdir(resultDirPath)
    labelme2yolov2Seg(jsonfilePath=jsonfilePath, resultDirPath=resultDirPath, classList=["ZSKPS","JY"])
