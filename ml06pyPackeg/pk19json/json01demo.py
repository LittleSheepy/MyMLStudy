import json

# 无格式化
def save_json1():
    data_dict = {}
    data_dict["key1"] = 1
    data_dict["key2"] = {"key2key":2}
    json_str = json.dumps(data_dict)
    with open('data.json', 'w') as f:
        f.write(json_str)

# 有格式化
def save_json():
    data_dict = {}
    data_dict["classesNum"] = 3
    data_dict["classes"] = ["00", "01", "02"]
    with open('last.json', 'w') as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    save_json()





















