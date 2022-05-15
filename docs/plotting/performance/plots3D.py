#!/usr/bin/python3
import json as js
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import cm
import numpy as np

z_dict = dict()

@np.vectorize
def f(x, y):
    return z_dict[(x,y)]

def read_settings(input_path):
    with open(input_path) as f:
        settings = js.load(f)
    return settings

def plot():
    sns.set_theme()
    settings = read_settings("/home/asl/asl-2022/docs/plotting/performance/settings3D.json")
    fig, ax = plt.subplots(figsize = (9.5,5.5), dpi = 80, subplot_kw={"projection": "3d"})
    X = []
    Y = []
    Z = []
    for line in settings["lines"]:
        df = pd.read_csv(line["input_file"])

        X.extend(df["m"].tolist())
        Y.extend(df["performance"].tolist())
        for i in range(len(df["m"].tolist())):
            print(df["performance"].tolist()[i])
            z_dict[(df["m"].tolist()[i], df["performance"].tolist()[i])] = df["r"].tolist()[i]
        
    X, Y = np.meshgrid(X, Y)
    X = np.array(X)
    Y = np.array(Y)
    Z = f(X,Y)
    
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    plt.title(settings["title"],fontsize = 23)

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel(settings["xlabel"], fontsize = 18)
    plt.ylabel(settings["ylabel"], fontsize = 18)
    plt.zlabel(settings["zlabel"], fontsize = 18)
   
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.zticks(fontsize = 15)
    plt.title(settings["title"])
    fig.colorbar(surf, shrink=0.5, aspect=5)

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
