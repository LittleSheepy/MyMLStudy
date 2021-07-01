#!/usr/bin/env python
# -*- coding: UTF-8 -*-

class cls:
    ss = {}
    ss["aa"] = [0,1,2]
    def func(self):
        self.ss["aa"].append(3)

cls1 = cls()
ss1 = cls1.ss
cls1.func()
print(ss1)

