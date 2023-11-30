from algseg.dataset.CCityScapes import CCityScapes
from algseg.dataset.CChase_db1 import CChase_db1



def get_dataset(dataset_name, root):
    if "cityscapes" == dataset_name:
        return CCityScapes(root)
    elif "chase_db1" == dataset_name:
        return CChase_db1(root)


if __name__ == '__main__':
    print(get_dataset("chase_db1", "D:/02dataset/CHASEDB1/"))
