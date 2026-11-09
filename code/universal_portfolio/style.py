import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
 "figure.dpi":200, "savefig.dpi":200, "savefig.bbox":"tight", "savefig.pad_inches":0.02,
 "pdf.fonttype":42, "ps.fonttype":42, "font.family":"serif", "font.serif":["CMU Serif","DejaVu Serif"], "mathtext.fontset":"cm",
 "font.size":10, "axes.labelsize":10, "axes.titlesize":10,
 "xtick.labelsize":9, "ytick.labelsize":9, "legend.fontsize":8.5,
 "axes.linewidth":0.7, "xtick.major.width":0.7, "ytick.major.width":0.7,
 "xtick.direction":"out", "ytick.direction":"out",
 "grid.linewidth":0.5, "grid.alpha":0.25, "lines.linewidth":1.2,
 "legend.frameon":False, "legend.handlelength":1.8, "legend.borderaxespad":0.4,
})
W = 7.2          # full span width in inches, matching the IMS textwidth
C = {"blue":"#26456e","red":"#c0562f","green":"#2a8f82","orange":"#d4a11e",
     "purple":"#7b5ea7","grey":"#6b6b6b","light":"#a9bcd6"}
def finish(ax, legend=True, ncol=1, loc="best", above=False):
    ax.grid(True, which="major", color="#aaaaaa", linewidth=0.4, alpha=0.35)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    if legend and ax.get_legend_handles_labels()[0]:
        if above:
            ax.legend(ncol=ncol, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                      borderaxespad=0.0, columnspacing=1.4, handletextpad=0.5)
        else:
            ax.legend(ncol=ncol, loc=loc)
