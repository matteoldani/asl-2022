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
                z_dict[(df["m"].tolist()[i], df["r"].tolist()[i])] = df["performance"].tolist()[i]

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
    #ax.view_init(30, 200)  
    ax.view_init(30, 160)
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
