import matplotlib.pyplot as plt
import seaborn as sns

axes = plt.subplot(1, 1, 1)
plt.show()

fig,axes = plt.subplots(2, 2)
#axes[0][0].set_title("axe_0_0")
for i, ax in enumerate(axes.flatten()):
    ax.set_title(f"axe_{i}")
plt.title("title total")
plt.show()

plt.plot()
plt.title("title")
plt.show()