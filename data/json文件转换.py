import os
import json





def json2txt(jsonFile,txtFile):
    with open(jsonFile, 'r') as load_f:
        load_dict = json.load(load_f)
    with open(txtFile, 'a') as f_txt:
        f_txt.truncate(0)
        for item in load_dict["shapes"]:
            txt_role = ""
            for point in item["points"]:
                txt_role = txt_role + str(round(point[0], 1))+","+str(round(point[1], 1))+","
            txt_role = txt_role + item["label"] + "\n"
            f_txt.write(txt_role)

def main():
    root_dir = r"D:\01sheepy\01work\08fuchuang\01imageTrain/"
    json_dir = root_dir + "json/"
    txt_dir = root_dir + "txt/"
    for jsonname in os.listdir(json_dir):
        json2txt(json_dir + jsonname, txt_dir + jsonname[:-5]+".txt")


if __name__ == '__main__':
    main()
