from matplotlib import pyplot as plt
import numpy as np

def plot_with_debug(datas, names, save_path="./pic/debug.png"):
    # 折线图
    fig, ax = plt.subplots()
    for i in range(len(datas)):
        x_values = np.arange(len(datas[i])) 
        ax.plot(x_values, datas[i], label=names[i], marker='o', linestyle='-')
    ax.legend()
    ax.grid()
    plt.savefig(save_path, dpi=500)
    plt.show()