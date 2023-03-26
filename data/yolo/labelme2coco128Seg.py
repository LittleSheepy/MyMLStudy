import json
import os
import cv2
json_filepath="D:\Shanmh\DeskTop/vase"#包含json文件的目录
image_filepath="D:\Shanmh\DeskTop/vase"#包含原图文件的目录
txt_savapath="D:\Shanmh\DeskTop/vase"#txt保存目录
json_files=[i for i in os.listdir(json_filepath) if i.endswith("json")]
for i in json_files:
    #新建一个数据list
    info_list=[]
    json_path=os.path.join(json_filepath,i)
    with open(json_path,"r") as r:
        json_info=json.load(r)
    r.close()
    imagePath=json_info["imagePath"]
    #得到图像宽高
    h,w=cv2.imread(os.path.join(image_filepath,imagePath)).shape[:2]
    shapes=json_info["shapes"]
    for shape in shapes:#每一个区域
        row_str=""#初始化一行
        label=shape["label"]
        row_str+=label
        points=shape["points"]
        for point in points:
            x=round(float(point[0])/w,6)#小数保留6位
            y = round(float(point[1]) / h, 6)  # 小数保留6位
            row_str+=" "+str(x)+" "+str(y)
        row_str+="\n"
        info_list.append(row_str)
    with open(os.path.join(txt_savapath,i.replace(".json",".txt")),"w") as w:
        w.writelines(info_list)
    w.close()
    print(f"已保存文件{i.replace('.json','.txt')}")