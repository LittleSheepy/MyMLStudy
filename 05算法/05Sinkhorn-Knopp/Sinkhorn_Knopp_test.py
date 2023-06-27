"""
    https://zhuanlan.zhihu.com/p/542379144
"""
import pandas as pd
import matplotlib.pyplot as plt

from Sinkhorn_Knopp import compute_optimal_transport

# ## The dessert problem
# Take a look at the preferences for desserts in the [KERMIT](kermit.ugent.be) research unit.
# preferences 喜好
ren_list = ['ren1', 'ren2', 'ren3', 'ren4', 'ren5', 'ren6', 'ren7', 'ren8']
chi_list = ['chi1', 'chi2', 'chi3', 'chi4', 'chi5']
preferences = pd.DataFrame([
    [2, 2, 1, 0, 0],
    [0, -2, -2, -2, 2],
    [1, 2, 2, 2, -1],
    [2, 1, 0, 1, -1],
    [0.5, 2, 2, 1, 0],
    [0, 1, 1, 1, -1],
    [-2, 2, 2, 1, 1],
    [2, 1, 2, 1, -1]
], index=ren_list)
preferences.columns = chi_list

# M_df = preferences.copy()
M_df = - preferences
M = - preferences.values  # cost is negative preferences 成本是负面偏好

# portions per person  每人的分量
portions_per_person = pd.DataFrame([[3],
                                    [3],
                                    [3],
                                    [4],
                                    [2],
                                    [2],
                                    [2],
                                    [1]],
                                   index=ren_list)
# quantities  点心数量
quantities_of_dessert = pd.DataFrame([[4], [2], [6], [4], [4]],
                                     index=chi_list)

print(preferences.to_markdown())
print("")
print(M_df.to_markdown())

# How many portions per person. 画出每人多少份
ax = portions_per_person.plot(kind='bar')
ax.set_ylabel('Portions')
ax.set_title('Number of Dessert Portions per Person')
r = portions_per_person.values.ravel()  # store as vector

# How much of every dessert. 每种甜点要多少。
ax = quantities_of_dessert.plot(kind='bar')
ax.set_ylabel('Portions')
ax.set_title('How much of every dessert.')
c = quantities_of_dessert.values.ravel()  # store as vector


# plt.show()

# lam 越小越趋向于平均分配
def SK_Test(lam):
    P, d = compute_optimal_transport(M,
                                     r,
                                     c, lam=lam)

    partition = pd.DataFrame(P, index=preferences.index, columns=preferences.columns)
    ax = partition.plot(kind='bar', stacked=True)
    print('Sinkhorn distance: {}'.format(d))
    ax.set_ylabel('portions')
    ax.set_title(f'Optimal distribution ($\lambda={lam}$)')
    plt.show()


if __name__ == '__main__':
    SK_Test(1)
    SK_Test(10)
    SK_Test(50)
