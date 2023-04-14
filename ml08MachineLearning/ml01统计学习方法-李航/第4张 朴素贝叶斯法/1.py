import numpy as np
Num = 100000
W = 6
N = 8
p = np.random.rand(Num)
np.random.shuffle(p)
rolls = np.random.rand((W*2 - 1)*len(p))
np.random.shuffle(rolls)
rolls = rolls.reshape([W*2 - 1,Num])
Alice_wins = rolls < p
Bob_wins = rolls >= p
total_wins = Alice_wins + Bob_wins
assert np.all(total_wins == 1)
print('sanity check passed')
sum = np.sum(Alice_wins[0:, :], 0)
#ss = sum.sum()
good_games = np.sum(Alice_wins[0:N, :], 0) == (W - 1)
gg = good_games.sum()
pgg = p[good_games]
print('the respective probs:', p[good_games].sum())
print('# of suitable games: {0}'.format(good_games.sum()))

import pandas as pd
result = pd.value_counts(list(sum))

from collections import Counter
result1 = Counter(sum)
# truncate our results to consider only these good games
# 表达一种条件概率的概念
given_Alice_wins = Alice_wins[:, good_games]
given_Bob_wins = Bob_wins[:, good_games]
h = np.sum(given_Alice_wins[:, :], 0).sum()
h1 = np.sum(given_Bob_wins[:, :], 0).sum()
hh = h + h1
hhc = h/h1
q = np.sum(given_Alice_wins[:N, :], 0).sum()
q1 = np.sum(given_Bob_wins[:N, :], 0).sum()
qc = q/q1
ss = np.sum(given_Alice_wins[N:, :], 0).sum()
ss1 = np.sum(given_Bob_wins[N:, :], 0).sum()
c = ss/ss1
z = np.sum(Alice_wins[:, :], 0).sum()
z1 = np.sum(Bob_wins[:, :], 0).sum()
zc = z/z1
# Monte Carlo Prob
bwn = (np.sum(given_Bob_wins, 0) == W).sum()
P_B_mc = (np.sum(given_Bob_wins, 0) == W).sum() / good_games.sum()
print('Monte Carlo probability of Bob wining: {:.4f}'.format(P_B_mc))
print('MC odds against Bob wining :{:.4f}'.format((1-P_B_mc)/P_B_mc))

# Monte Carlo Prob
P_A_mc = (np.sum(given_Alice_wins, 0) >= W).sum() / good_games.sum()
print('Monte Carlo probability of Bob wining: {:.4f}'.format(P_A_mc))
print('MC odds against Bob wining :{:.4f}'.format(P_A_mc/(1-P_A_mc)))