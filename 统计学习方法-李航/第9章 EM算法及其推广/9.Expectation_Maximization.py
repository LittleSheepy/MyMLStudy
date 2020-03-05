#!/usr/bin/env python
# coding: utf-8
# # 第9章 EM算法及其推广

import numpy as np
import math

pro_A, pro_B, por_C = 0.5, 0.5, 0.5


def pmf(i, pro_A, pro_B, por_C):
    pro_1 = pro_A * math.pow(pro_B, data[i]) * math.pow(
        (1 - pro_B), 1 - data[i])
    pro_2 = pro_A * math.pow(pro_C, data[i]) * math.pow(
        (1 - pro_C), 1 - data[i])
    return pro_1 / (pro_1 + pro_2)

class EM:
    def __init__(self, prob):
        self.pro_A, self.pro_B, self.pro_C = prob

    # e_step
    def pmf(self, i):
        pro_1 = self.pro_A * math.pow(self.pro_B, data[i]) * math.pow(
            (1 - self.pro_B), 1 - data[i])
        pro_2 = (1 - self.pro_A) * math.pow(self.pro_C, data[i]) * math.pow(
            (1 - self.pro_C), 1 - data[i])
        result = pro_1 / (pro_1 + pro_2)
        return result

    # m_step
    def fit(self, data):
        count = len(data)
        print('init prob:{}, {}, {}'.format(self.pro_A, self.pro_B,
                                            self.pro_C))
        for d in range(count):
            #_ = yield
            _pmf = [self.pmf(k) for k in range(count)]
            pro_A = 1 / count * sum(_pmf)
            pro_B = sum([_pmf[k] * data[k] for k in range(count)]) / sum(_pmf)
            pro_C = sum([(1 - _pmf[k]) * data[k]
                         for k in range(count)]) / sum([(1 - _pmf[k])
                                                        for k in range(count)])
            print('{}/{}  pro_a:{:.16f}, pro_b:{:.16f}, pro_c:{:.16f}'.format(
                d + 1, count, pro_A, pro_B, pro_C))
            self.pro_A = pro_A
            self.pro_B = pro_B
            self.pro_C = pro_C

data = [1,1,0,1,0,0,1,0,1,1,1,1,0,1,0,0,1,0,1,1,1,1,0,1,0,0,1,0,1,1,1,1,0,1,0,0,1,0,1,1,1,1,0,1,0,0,1,0,1,1,1,1,0,1,0,0,1,0,1,1]

#em = EM(prob=[0.5, 0.5, 0.5])
#em.fit(data)


em = EM(prob=[0.4, 0.6, 0.7])
em.fit(data)

