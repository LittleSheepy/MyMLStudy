from openxlab.dataset import get
import openxlab

openxlab.login(ak="djm90wzzbkqxnraw7mvz", sk ="wnaey41b2kxroll9obwydxp25q8v7zm6gymdpv3j")
# get(dataset_repo='OpenDataLab/CHASE_DB1', target_path='./data/')  # target_path：下载文件指定的本地路径

# get(dataset_repo='OpenDataLab/ADE20K_2016', target_path='./data/')  # target_path：下载文件指定的本地路径
get(dataset_repo='OpenDataLab/CityScapes', target_path='./data/')  # target_path：下载文件指定的本地路径