import random


# 进行10000次模拟实验
num_trials = 10000
win_count_switch = 0
win_count_stay = 0
false_count = 0
doors = [0] * 3  # 0代表山羊，1代表汽车
doors[0] = 1
for _ in range(num_trials):
    random.shuffle(doors)

    # 主持人知道汽车
    if doors[2] == 1:
        doors[1] = 1
        doors[2] = 0

    if doors[0] == 1:
        win_count_stay += 1
    elif doors[1] == 1:
        win_count_switch += 1
    else:
        false_count += 1



win_probability_switch = win_count_switch / num_trials
win_probability_stay = win_count_stay / num_trials

print(f"模拟实验中换门赢得汽车的概率: {win_probability_switch}")
print(f"模拟实验中不换门赢得汽车的概率: {win_probability_stay}")