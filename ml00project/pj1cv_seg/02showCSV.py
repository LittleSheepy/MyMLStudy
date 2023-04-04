import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def readCSV(file_path_csv):
    data = pd.read_csv(file_path_csv, index_col=False, na_values=[float('inf')])
    return data

def show2D():
    x_name = "scoreThereMaxRight_max"
    y_name = "scoreRight_max"

    scoreDefectList = resultmatZero[x_name].values
    scoreRight_meanList = resultmatZero[y_name].values
    plt.plot(scoreDefectList, scoreRight_meanList, "x")
    scoreDefectList = resultmatOne[x_name].values
    scoreRight_meanList = resultmatOne[y_name].values
    plt.plot(scoreDefectList, scoreRight_meanList, "o")
    plt.show()

def show3D():
    xyz_name = ["scoreRight_max", "scoreThereMaxRight_max", "scoreDefect"]
    markers = ["o", "x", "+"]
    resultmatZero1 = resultmatZero[resultmatZero["scoreRight_max"] < 0.8]
    resultmat = [resultmatZero1, resultmatOne, resultmatTwo]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for i, resultmatX in enumerate(resultmat):
        item1 = resultmatX[xyz_name[0]].values
        item2 = resultmatX[xyz_name[1]].values
        item3 = resultmatX[xyz_name[2]].values
        ax.scatter(item1, item2, item3, s=50, alpha=0.5, marker=markers[i])
    plt.show()
if __name__ == '__main__':
    dir_root = r"D:/04DataSets/02/"
    file_path_csv = dir_root + "result.csv"
    dir_resultCSV = dir_root + "resultCSV/"
    resultmat = readCSV(file_path_csv)

    resultmatZero = resultmat[resultmat["class"] == 0]
    resultmatOne = resultmat[resultmat["class"] == 1]
    resultmatTwo = resultmat[resultmat["class"] == 2]
    a = 1
    #  scoreDefect
    # scoreDefectOne = resultmatOne["std/std"].values
    # scoreDefectZero = resultmatZero["std/std"].values
    # resultmatZero["std/std"][np.isinf(resultmatZero["std/std"])] = 100
    # resultmatZero["std/std"][np.isnan(resultmatZero["std/std"])] = 0
    # plt.hist(scoreDefectOne, bins=10, fill=False, density=True, edgecolor="green", label="One")
    # plt.hist(scoreDefectZero, bins=10, fill=False, density=True, edgecolor="red", label="Zero")
    # plt.title("scoreDefect")
    # plt.legend()
    # plt.show()

    # for column in resultmat.columns[2:-1]:
    #     print(column)
    #     itemOne = resultmatOne[column].values
    #     itemZero = resultmatZero[column].values
    #     plt.hist(itemOne, bins=10, fill=False, density=True, edgecolor="green", label="One")
    #     plt.hist(itemZero, bins=10, fill=False, density=True, edgecolor="red", label="Zero")
    #     plt.title(column)
    #     plt.legend()
    #     column = str(column).replace("/", "_")
    #     plt.savefig(dir_resultCSV + column + ".jpg")
    #     plt.show()

    num = resultmat.shape[0]
    #scoreDefect = resultmat["scoreDefect"]
    x_name = "scoreThereMaxRight_max"
    y_name = "scoreRight_max"

    #show2D()
    # for i in range(num):
    #     scoreDefect = resultmat["scoreDefect"][i]
    #     scoreRight_mean = resultmat["scoreRight_mean"][i]
    #     class_ = resultmat["class"][i]

    show3D()
    print(a)
    pass

