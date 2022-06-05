#!/usr/bin/python3
import json as js
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def read_settings(input_path):
    with open(input_path) as f:
        settings = js.load(f)
    return settings

def plot():
    sns.set_theme()
    settings = read_settings("./settings.json")
    fig = plt.figure(figsize = (9.5,5.5), dpi = 80)
    ax = plt.subplot(111)
    df_bs1 = pd.read_csv("C:/File system/ETH/2nd Semester/ASL/Project/asl-2022/docs/plotting/speedup/bs1.out")

    for line in settings["lines"]:
        df = pd.read_csv(line["input_file"])

        ax.plot(df["m"], df_bs1["cycles"] / df["cycles"],
                color=line["color"], linewidth = 1.5,
                marker=line["marker"], markerfacecolor=line["color"],
                markersize=4, label = line["label"])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    plt.title(settings["title"],fontsize = 23)

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel(settings["xlabel"], fontsize = 18)
    plt.ylabel(settings["ylabel"], fontsize = 18)
   
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
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
