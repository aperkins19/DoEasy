import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Time course all
def plot_timecourse_mean(plotting_data, product_name):
    
    fig = plt.figure(figsize=(10,5))

    sns.lineplot(x="Time", y= "RFUs", hue = "Condition", data = plotting_data)

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    fig.suptitle(product_name + " "+"Timecourse Fluorescence of Conditions")
    fig.tight_layout()

    plt.savefig("/app/analysis_output/plots/"+product_name+"_timecourse_mean.png")


# bar plot
def show_values(axs, orient="v", space=.05):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.1f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.1f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


def endpoint_barplot(bar_plot_data, product_name):

    #bar_plot_data.to_csv("bar.csv")

    
    fig_barplot = plt.figure(figsize=(10,5))

    #ax = sns.barplot(x="Well", y= "RFUs", hue = "Condition", data = bar_plot_data)
    ax = sns.barplot(x="Condition", y= "RFUs",  data = bar_plot_data)
    # show values above 
    show_values(ax, orient="v", space=0.05)
    # rotate tick labels
    ax.set_xticklabels(np.arange(1,len(bar_plot_data["Condition"].unique())+1, 1))

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    fig_barplot.suptitle(product_name +" Expression at 100 mins")
    fig_barplot.tight_layout()

    plt.savefig("/app/analysis_output/plots/"+product_name+"_barplot_endpoint.png")