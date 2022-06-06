#!/usr/bin/python3
import json as js
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import cm
import numpy as np
import sys
import os

z_dict = dict()
np.set_printoptions(threshold=sys.maxsize)


@np.vectorize
def f(x, y):
    return z_dict[(x, y)]


def read_settings(input_path):
    with open(input_path) as f:
        settings = js.load(f)
    return settings


def nnm_cost_2(V_row, V_col, W_row, W_col, H_row, H_col, num_iterations):

    return (2 * W_row * H_col * W_col + 5 * V_row * V_col + 3) + num_iterations * (2 * W_row * H_col * W_col + 5 * V_row * V_col +
                                                2 * W_col * V_col * V_row +
                                                2 * W_col * W_col * W_row +
                                                2 * W_col * W_col * H_col +
                                                2 * V_row * H_row * V_col +
                                                1 * H_row * H_col * H_row +
                                                1 * H_row * H_row * W_row +
                                                2 * H_row * H_col +
                                                2 * W_row * W_col + 3)

def plot():
    # sns.set_theme()
    settings = read_settings("./settings3D.json")
    fig, ax = plt.subplots(figsize=(10, 10), dpi=80, subplot_kw={"projection": "3d"})
    X = []
    Y = []
    Z = []
    for line in settings["lines"]:
        for filename in os.listdir(line['input_file']):
            df = pd.read_csv(line['input_file'] + filename)

            X.extend(df["m"].tolist())
            Y.extend(df["r"].tolist())
            for i in range(len(df["m"].tolist())):
                # if settings['title'].endswith('61'):
                #     if df["r"].tolist()[i] == 16:
                #         l = df["performance"].tolist()[i]
                #     else:
                #         real_m = (((df["m"].tolist()[i] - 1) / 16) + 1) * 16
                #         real_n = (((df["n"].tolist()[i] - 1) / 16) + 1) * 16
                #         real_r = (((df["r"].tolist()[i] - 1) / 16) + 1) * 16

                #         new_cost = nnm_cost_2(real_m, real_n, real_m, real_r, real_r, real_n, 100)
                #         new_cost += (5 * real_n * real_m)
                #         new_cost += (5 * real_m * real_r)
                #         new_cost += (5 * real_r * real_n)
                #         l = new_cost / df["cycles"].tolist()[i]
                # else:
                l = df["performance"].tolist()[i]
                z_dict[(df["m"].tolist()[i], df["r"].tolist()[i])] = l

    X = np.array(X)
    Y = np.array(Y)
    p = X.argsort()
    o = Y.argsort()

    X, Y = np.meshgrid(X[p], Y[o])
    Z = f(X, Y)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    plt.title(settings["title"], fontsize=23)

    plt.xlabel(settings["xlabel"], fontsize=12)
    plt.ylabel(settings["ylabel"], fontsize=12)
    ax.set_zlabel(settings["zlabel"], fontsize=12)

    plt.yticks(np.arange(8, 25, 2))
    ax.view_init(30, 200)  
    #ax.view_init(30, 160)
    plt.title(settings["title"])

    if settings["action"] == "show":
        plt.show()
    else:
        plt.savefig(settings["output_file"])


if __name__ == "__main__":
    print("Expected input files structure:")
    print("m,r,n,cycles,performance\n\n")

    print("Use settings.json to change the plots")
    print("Possible actions: ['save', 'show']")

    plot()
