#!/usr/bin/env python
# -*- coding: UTF-8 -*-

dict1 = {}
dict1["a"] = []
dict1["a"].append(1)
dict1["a"].append(2)
dict1["n"]=2
list1 = dict1["a"]
n = dict1["n"]
dict1["a"].pop(0)
dict1["n"] -= 1
print(dict1.keys())
print(dict1["a"],list1,dict1["n"],n)
list1[0] = 10

print(dict1["a"],list1)

def getdict():
    dict1 = {}
    dict1["a"] = 1
    return dict1

dict11 = getdict()
dict12 = getdict()
dict11["a"] = 2
dict13 = getdict()
print(dict11, dict12, dict13)

aa = 3
for i in range(aa):
    if aa < 5:
        aa += 1
    print(i)
