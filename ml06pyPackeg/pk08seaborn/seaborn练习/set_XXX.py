import matplotlib.pyplot as plt
import seaborn as sns

""" set_style """
plt.plot()
plt.show()
sns.set_style("white")
plt.plot()
plt.show()
sns.set_style("dark")
plt.plot()
plt.show()

sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
sns.set_context("paper", font_scale=1)
plt.plot()
plt.show()

""" set_context """
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
sns.set_context("paper", font_scale=0.5)
plt.plot()
plt.show()


